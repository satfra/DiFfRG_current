#include <DiFfRG/physics/integration/map_scheduler.hh>

// std
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

namespace DiFfRG
{
  namespace
  {
    double env_double(const char *name, const double fallback)
    {
      const char *v = std::getenv(name);
      if (v == nullptr || *v == '\0') return fallback;
      try {
        return std::stod(v);
      } catch (...) {
        std::cerr << "WARNING: could not parse " << name << "='" << v << "' as a number; ignoring." << std::endl;
        return fallback;
      }
    }

    bool env_flag(const char *name)
    {
      const char *v = std::getenv(name);
      return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
    }
  } // namespace

  MapScheduler &MapScheduler::instance()
  {
    static MapScheduler scheduler;
    return scheduler;
  }

  MapScheduler::MapScheduler(const uint rank, const uint n_ranks)
      : m_comm(MPI_COMM_WORLD), m_rank(rank), m_n_ranks(n_ranks), m_simulated(true)
  {
    m_load.assign(m_n_ranks, 0.);
  }

  MapScheduler::MapScheduler() : m_comm(MPI_COMM_WORLD)
  {
    m_rank = MPI::rank(m_comm);
    m_n_ranks = MPI::size(m_comm);
    const double from_env = env_double("DIFFRG_MAP_QUANTUM", 0.);
    if (from_env > 0.) {
      m_quantum = from_env;
      m_quantum_pinned = true;
    }
    m_verbose = env_flag("DIFFRG_MAP_VERBOSE");
    m_load.assign(m_n_ranks, 0.);
  }

  void MapScheduler::set_quantum(double quantum)
  {
    // DIFFRG_MAP_QUANTUM is the knob a scaling sweep turns from the outside, so it wins over the
    // parameter file rather than being silently overwritten by it.
    if (quantum > 0. && !m_quantum_pinned) m_quantum = quantum;
  }

  std::vector<uint> MapScheduler::least_loaded(const size_t r) const
  {
    std::vector<uint> order(m_n_ranks);
    std::iota(order.begin(), order.end(), 0u);
    // Ties broken by rank index, so the choice is identical on every rank.
    std::partial_sort(order.begin(), order.begin() + r, order.end(), [this](const uint a, const uint b) {
      if (m_load[a] != m_load[b]) return m_load[a] < m_load[b];
      return a < b;
    });
    std::vector<uint> owners(order.begin(), order.begin() + r);
    // Ascending, so that slice j of the grid always belongs to the j-th smallest owning rank.
    std::sort(owners.begin(), owners.end());
    return owners;
  }

  MapSlice MapScheduler::schedule(const size_t integrator_id, void *dest, const size_t elem_size,
                                  const size_t grid_size, const size_t quadrature_volume, const bool splittable)
  {
    if (!active() || grid_size == 0) return MapSlice{0, grid_size};

    const double S = double(grid_size) * double(quadrature_volume);

    size_t r = 1;
    if (splittable && grid_size > 1 && m_quantum > 0.) {
      const double raw = S / m_quantum;
      r = raw < 1. ? 1 : static_cast<size_t>(raw);
      r = std::min<size_t>(r, std::min<size_t>(m_n_ranks, grid_size));
    }

    const std::vector<uint> owners = least_loaded(r);

    MapSlice mine{0, 0};
    for (size_t j = 0; j < r; ++j) {
      const size_t begin = part(grid_size, r, j);
      const size_t end = part(grid_size, r, j + 1);
      m_load[owners[j]] += double(end - begin) * double(quadrature_volume);
      if (owners[j] == m_rank) mine = MapSlice{begin, end - begin};
    }

    m_plan.push_back(Entry{integrator_id, dest, elem_size, grid_size, owners});
    return mine;
  }

  bool MapScheduler::plan_contains(const size_t integrator_id) const
  {
    for (const auto &e : m_plan)
      if (e.integrator_id == integrator_id) return true;
    return false;
  }

  uint64_t MapScheduler::plan_hash() const
  {
    // FNV-1a over everything that must agree between ranks. Destination *pointers* are deliberately
    // excluded: they are rank-local and ASLR makes them differ legitimately.
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](const uint64_t v) {
      for (int b = 0; b < 8; ++b) {
        h ^= (v >> (8 * b)) & 0xffull;
        h *= 1099511628211ull;
      }
    };
    mix(m_plan.size());
    for (const auto &e : m_plan) {
      mix(e.integrator_id);
      mix(e.elem_size);
      mix(e.grid_size);
      mix(e.owners.size());
      for (const uint o : e.owners)
        mix(o);
    }
    return h;
  }

  void MapScheduler::log_plan() const
  {
    std::stringstream ss;
    ss << "MapScheduler: " << m_n_ranks << " ranks, quantum " << m_quantum << ", " << m_plan.size()
       << " maps in this batch\n";
    for (size_t e = 0; e < m_plan.size(); ++e) {
      const auto &en = m_plan[e];
      ss << "  map " << e << ": G=" << en.grid_size << " split " << en.owners.size() << "x over ranks {";
      for (size_t j = 0; j < en.owners.size(); ++j)
        ss << en.owners[j] << (j + 1 < en.owners.size() ? "," : "");
      ss << "}\n";
    }
    ss << "  per-rank load (kernel evaluations):";
    for (uint p = 0; p < m_n_ranks; ++p)
      ss << " [" << p << "]=" << m_load[p];
    std::cout << ss.str() << std::endl;
  }

  void MapScheduler::poison(const char *reason)
  {
    m_plan.clear();
    std::fill(m_load.begin(), m_load.end(), 0.);
    if (m_poisoned == nullptr) m_poisoned = reason;
  }

  void MapScheduler::complete()
  {
    if (!prepare_batch()) return;
    MPI::allgatherv_bytes(m_comm, m_send.data(), static_cast<size_t>(m_counts[m_rank]), m_recv.data(), m_counts,
                          m_displs);
    finish_batch();
  }

  bool MapScheduler::prepare_batch()
  {
    if (!active()) {
      m_plan.clear();
      return false;
    }

    if (m_poisoned != nullptr)
      MPI::abort(m_comm, std::string("the map schedule was abandoned mid-batch (") + m_poisoned +
                             "), so the ranks can no longer agree on what to exchange. Continuing would "
                             "deadlock in the next collective.");

    if (m_plan.empty()) return false;

    if (m_verbose && !m_logged) {
      if (m_rank == 0) log_plan();
      m_logged = true;
    }

    // -- 1. Agree that every rank built the same plan. -------------------------------------------
    // A mismatch here would otherwise present as a hang inside the Allgatherv below (mismatched
    // counts), which is close to undiagnosable on a cluster.
    if (!m_simulated && !MPI::agree(m_comm, plan_hash()))
      MPI::abort(m_comm, "ranks disagree about the map schedule. Every rank must issue the same sequence of map() "
                         "calls; a rank-local branch around a flow evaluation is the usual cause.");

    // -- 2. Byte counts per rank, in canonical order. ---------------------------------------------
    m_counts.assign(m_n_ranks, 0);
    for (const auto &e : m_plan)
      for (size_t j = 0; j < e.owners.size(); ++j)
        m_counts[e.owners[j]] +=
            static_cast<int>((part(e.grid_size, e.owners.size(), j + 1) - part(e.grid_size, e.owners.size(), j)) *
                             e.elem_size);

    m_displs.assign(m_n_ranks, 0);
    size_t total = 0;
    for (uint p = 0; p < m_n_ranks; ++p) {
      m_displs[p] = static_cast<int>(total);
      total += static_cast<size_t>(m_counts[p]);
    }
    // MPI_Allgatherv counts and displacements are `int`. A flow batch is kilobytes, so this is a
    // backstop against a future caller, not a live concern -- but silently truncating a
    // displacement would corrupt results rather than fail.
    if (total > static_cast<size_t>(std::numeric_limits<int>::max()))
      MPI::abort(m_comm, "a single map batch exceeds 2 GB, which MPI_Allgatherv cannot address. "
                         "Flush more often (a shorter DeferredMaps scope).");

    // -- 3. Pack this rank's slices. ---------------------------------------------------------------
    const size_t my_bytes = static_cast<size_t>(m_counts[m_rank]);
    if (m_send.size() < my_bytes) m_send.resize(my_bytes);
    if (m_recv.size() < total) m_recv.resize(total);

    size_t off = 0;
    for (const auto &e : m_plan) {
      const size_t r = e.owners.size();
      for (size_t j = 0; j < r; ++j) {
        if (e.owners[j] != m_rank) continue;
        const size_t begin = part(e.grid_size, r, j);
        const size_t bytes = (part(e.grid_size, r, j + 1) - begin) * e.elem_size;
        std::memcpy(m_send.data() + off, static_cast<const char *>(e.dest) + begin * e.elem_size, bytes);
        off += bytes;
      }
    }

    return true;
  }

  void MapScheduler::finish_batch()
  {
    // -- 4. Land every other rank's slices. --------------------------------------------------------
    // Our own slices are already in place, so they are skipped rather than copied back over
    // themselves.
    for (uint p = 0; p < m_n_ranks; ++p) {
      if (p == m_rank) continue;
      size_t block = static_cast<size_t>(m_displs[p]);
      for (const auto &e : m_plan) {
        const size_t r = e.owners.size();
        for (size_t j = 0; j < r; ++j) {
          if (e.owners[j] != p) continue;
          const size_t begin = part(e.grid_size, r, j);
          const size_t bytes = (part(e.grid_size, r, j + 1) - begin) * e.elem_size;
          std::memcpy(static_cast<char *>(e.dest) + begin * e.elem_size, m_recv.data() + block, bytes);
          block += bytes;
        }
      }
    }

    m_plan.clear();
    std::fill(m_load.begin(), m_load.end(), 0.);
  }
} // namespace DiFfRG
