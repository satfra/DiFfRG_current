#pragma once

// DiFfRG
#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/quadrature/matsubara.hh>
#include <DiFfRG/common/quadrature/quadrature.hh>
#include <DiFfRG/common/run_logger.hh>

// standard library
#include <mutex>

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief A class that stores Matsubara quadrature points and weights for a given T, E.
     * Its main purpose is to avoid recomputing the quadrature points and weights for each Matsubara integrator and
     * provide a search algorithm to find previously computed Matsubara quadratures.
     */
    class MatsubaraStorage
    {
    public:
      /**
       * @brief Return the MatsubaraQuadrature object for a given T, E
       *
       */
      template <typename NT = double> MatsubaraQuadrature<NT> &get_matsubara_quadrature(const NT T, const NT E)
      {
        std::lock_guard<std::mutex> lock(m_mutex);

        if constexpr (std::is_same_v<NT, double>) {
          auto T_it = find_T_d(T);
          auto E_it = find_E_d(E, T_it);
          return E_it->second;
        } else if constexpr (std::is_same_v<NT, float>) {
          auto T_it = find_T_f(T);
          auto E_it = find_E_f(E, T_it);
          return E_it->second;
        }
        static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                      "Unknown type requested of MatsubaraStorage::get_matsubara_quadrature");
      }

      void set_verbosity(int v);
      void set_log_port(LogPort port) { log = std::move(port); }

      void set_vacuum_quad_size(const int size);
      void set_min_matsubara_size(const int value);
      void set_max_matsubara_size(const int value);
      void set_matsubara_precision_factor(const double value);

    private:
      MatsubaraQuadrature<double> &get_matsubara_quadrature_d(const double T, const double E);
      MatsubaraQuadrature<float> &get_matsubara_quadrature_f(const float T, const float E);

      template <typename NT = double> using StorageType = std::map<double, std::map<double, MatsubaraQuadrature<NT>>>;
      template <typename NT = double> using SubStorageType = std::map<double, MatsubaraQuadrature<NT>>;

      template <typename NT = double> using TemperatureIterator = typename StorageType<NT>::iterator;
      template <typename NT = double> using EnergyIterator = typename StorageType<NT>::mapped_type::iterator;

      TemperatureIterator<double> find_T_d(const double T);
      TemperatureIterator<float> find_T_f(const float T);

      EnergyIterator<double> find_E_d(const double E, TemperatureIterator<double> T_it);
      EnergyIterator<float> find_E_f(const float E, TemperatureIterator<float> T_it);

      StorageType<double> quadratures_d;
      StorageType<float> quadratures_f;

      int verbosity = 0;
      int vacuum_quad_size = 64;
      double matsubara_precision_factor = 0;
      int min_matsubara_size = 32;
      int max_matsubara_size = 128;

      std::mutex m_mutex;
      LogPort log;
    };

    /**
     * @brief A class that stores Quadrature points and weights for a given type and order
     * Its main purpose is to avoid recomputing the quadrature points and weights for each integrator and
     * provide a search algorithm to find previously computed quadratures.
     */
    class QuadratureStorage
    {
    public:
      template <typename NT = double> Quadrature<NT> &get_quadrature(const size_t order, const QuadratureType type)
      {
        std::lock_guard<std::mutex> lock(m_mutex);

        if constexpr (std::is_same_v<NT, double>) {
          auto type_it = find_type_d(type);
          auto order_it = find_order_d(order, type_it);
          return order_it->second;
        } else if constexpr (std::is_same_v<NT, float>) {
          auto type_it = find_type_f(type);
          auto order_it = find_order_f(order, type_it);
          return order_it->second;
        }
        static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                      "Unknown type requested of QuadratureStorage::get_quadrature");
      }

      void set_verbosity(int v);
      void set_log_port(LogPort port) { log = std::move(port); }

    private:
      Quadrature<double> &get_quadrature_d(const size_t order, const QuadratureType type);
      Quadrature<float> &get_quadrature_f(const size_t order, const QuadratureType type);

      template <typename NT = double> using StorageType = std::map<QuadratureType, std::map<size_t, Quadrature<NT>>>;
      template <typename NT = double> using SubStorageType = std::map<size_t, Quadrature<NT>>;

      template <typename NT = double> using TypeIterator = typename StorageType<NT>::iterator;
      template <typename NT = double> using OrderIterator = typename StorageType<NT>::mapped_type::iterator;

      TypeIterator<double> find_type_d(const QuadratureType type);
      TypeIterator<float> find_type_f(const QuadratureType type);

      OrderIterator<double> find_order_d(const size_t order, TypeIterator<double> type_it);
      OrderIterator<float> find_order_f(const size_t order, TypeIterator<float> type_it);

      StorageType<double> quadratures_d;
      StorageType<float> quadratures_f;

      int verbosity = 0;

      std::mutex m_mutex;
      LogPort log;
    };
  } // namespace internal

  /**
   * @brief A class that provides quadrature points and weights, in host and device memory.
   * The quadrature points and weights are computed either the GSL quadratures or the MatsubaraQuadrature class.
   * This avoids recomputing the quadrature points and weights for each integrator.
   */
  class QuadratureProvider
  {
  public:
    QuadratureProvider();
    /**
     * @brief Construct a provider that reports the quadratures it builds.
     *
     * If no port is passed, the provider opens its own side-channel log at
     * <output folder>/<output name>_quadrature.log. It has to own that logger rather than borrow one from an
     * OutputSession, because integrators request their quadratures inside their constructors -- typically before
     * any output session exists -- and because the provider is a process-level cache that outlives a single run.
     * Set /output/quadrature_log to false to suppress the file.
     */
    explicit QuadratureProvider(const ConfigTree &config, LogPort log = {});

    /**
     * @brief Get the quadrature points for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const std::vector<double>&
     */
    template <typename NT = double, typename MemorySpace = CPU_memory>
    auto nodes(const size_t order, const QuadratureType type = QuadratureType::legendre)
    {
      return quadrature_storage.get_quadrature<NT>(order, type).template nodes<MemorySpace>();
    }

    /**
     * @brief Get the quadrature weights for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const std::vector<double>&
     */
    template <typename NT = double, typename MemorySpace = CPU_memory>
    auto weights(const size_t order, const QuadratureType type = QuadratureType::legendre)
    {
      return quadrature_storage.get_quadrature<NT>(order, type).template weights<MemorySpace>();
    }

    /**
     * @brief Get the quadrature points for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const std::vector<double>&
     */
    template <typename NT = double, typename MemorySpace = CPU_memory>
    auto matsubara_nodes(const NT T, const NT typical_E)
    {
      return matsubara_storage.get_matsubara_quadrature<NT>(T, typical_E).template nodes<MemorySpace>();
    }

    /**
     * @brief Get the quadrature weights for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const std::vector<double>&
     */
    template <typename NT = double, typename MemorySpace = CPU_memory>
    auto matsubara_weights(const NT T, const NT typical_E)
    {
      return matsubara_storage.get_matsubara_quadrature<NT>(T, typical_E).template weights<MemorySpace>();
    }

    /**
     * @brief Get the effective temperature for the Matsubara zero-mode weight.
     * Returns 0 for vacuum (T=0) quadratures and the physical T for Monien quadratures.
     */
    template <typename NT = double> NT matsubara_T(const NT T, const NT typical_E)
    {
      return matsubara_storage.get_matsubara_quadrature<NT>(T, typical_E).get_T();
    }

    /**
     * @brief Hand out one of a small pool of execution space instances, round robin.
     *
     * Every integrator used to default-construct its ExecutionSpace, i.e. they all shared the
     * default stream and no two flows could ever execute concurrently -- even the ones that are
     * mutually independent (the vertex flows of a typical RHS read the same dressings and write
     * disjoint outputs). Handing neighbouring integrators different instances lets those overlap,
     * which matters most for the many small launches that individually leave the device
     * underutilised.
     *
     * Safe to mix with the interpolators' uploads, which fence on the host before returning, so
     * any kernel issued afterwards on any stream already sees the new data.
     *
     * The pool is created lazily and shared by every integrator holding this provider. Host
     * execution spaces have no streams to speak of, so they get a default-constructed instance.
     */
    template <typename ExecutionSpace> ExecutionSpace next_execution_space()
    {
      if constexpr (std::is_same_v<typename ExecutionSpace::memory_space, CPU_memory>) {
        return ExecutionSpace();
      } else {
        static constexpr size_t pool_size = 8;
        static std::vector<ExecutionSpace> pool = [] {
          const std::vector<int> weights(pool_size, 1);
          return Kokkos::Experimental::partition_space(ExecutionSpace(), weights);
        }();
        static size_t next = 0;
        return pool[next++ % pool_size];
      }
    }

  private:
    internal::MatsubaraStorage matsubara_storage;
    internal::QuadratureStorage quadrature_storage;

    /** Only engaged when the provider opens its own quadrature log; empty when a port was handed in. */
    RunLogger own_logger;

    int verbosity;
  };
} // namespace DiFfRG
