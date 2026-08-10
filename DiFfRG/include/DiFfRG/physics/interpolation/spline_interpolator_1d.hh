#pragma once

// DiFfRG
#include <DiFfRG/common/fast_interp.hh>
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>

// std
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief A spline interpolator for 1D data, both on GPU and CPU
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates, typename DefaultMemorySpace = CPU_memory> class SplineInterpolator1D
  {
    static_assert(Coordinates::dim == 1, "SplineInterpolator1D requires 1D coordinates");
    // The spline coefficients come from a non-cyclic tridiagonal solve with boundary conditions at the grid edges,
    // which cannot close a periodic axis. Use LinearInterpolator1D there instead.
    static_assert(!is_periodic_coordinate_v<Coordinates>,
                  "SplineInterpolator1D does not support periodic coordinates; use LinearInterpolator1D.");

  public:
    using memory_space = DefaultMemorySpace;
    using other_memory_space = other_memory_space_t<DefaultMemorySpace>;
    using ctype = typename Coordinates::ctype;
    using value_type = NT;
    static constexpr size_t dim = 1;

    /**
     * @brief Construct a SplineInterpolator1D with internal, zeroed data and a coordinate system.
     *
     * @param coordinates coordinate system of the data
     */
    SplineInterpolator1D(const Coordinates &coordinates) : coordinates(coordinates), size(coordinates.size())
    {
      // Allocate separate views for values and spline coefficients (SoA layout)
      device_values = ValueViewType("SplineInterpolator1D_values", size);
      device_coeffs = CoeffViewType("SplineInterpolator1D_coeffs", size);
      // Create host mirrors
      host_values = Kokkos::create_mirror_view(device_values);
      host_coeffs = Kokkos::create_mirror_view(device_coeffs);
      // Tabulated inverse transform ("fastInterpLookups"): sampled ONCE here; see fast_interp.hh.
      if (fast_interp_lookups() && size >= 2) build_inverse_table();
    }

    /**
     * @brief Copy constructor for SplineInterpolator1D. This is ONLY for usage inside Kokkos parallel loops.
     *
     */
    KOKKOS_FUNCTION
    SplineInterpolator1D(const SplineInterpolator1D &other) : coordinates(other.coordinates), size(other.size)
    {
      // Use the same data (reference-counted)
      device_values = other.device_values;
      device_coeffs = other.device_coeffs;
      device_fastidx = other.device_fastidx;
      m_inv_degree = other.m_inv_degree;
      m_fast_index = other.m_fast_index;
    }

    KOKKOS_FUNCTION ~SplineInterpolator1D() { KOKKOS_IF_ON_HOST((if (owns_other_instance) delete other_instance;)) }

    template <typename NT2>
    void update(const NT2 *in_data, const ctype lower_y1 = std::numeric_limits<ctype>::max(),
                const ctype upper_y1 = std::numeric_limits<ctype>::max())
    {
      // Check if the host data is already allocated
      if (!host_values.is_allocated())
        throw std::runtime_error(
            "SplineInterpolator1D: You probably called update() on a copied instance. This is not allowed. "
            "You need to call update() on the original instance.");

      // Copy values from input data
      for (size_t i = 0; i < size; ++i)
        host_values(i) = in_data[i];

      // Build the spline coefficients
      build_y2(lower_y1, upper_y1);

      // Copy data to device
      Kokkos::deep_copy(device_values, host_values);
      Kokkos::deep_copy(device_coeffs, host_coeffs);
    }

    template <typename View>
      requires(Kokkos::is_view<View>::value && View::rank == 1)
    void update(const View &view, const ctype lower_y1 = std::numeric_limits<ctype>::max(),
                const ctype upper_y1 = std::numeric_limits<ctype>::max())
    {
      // Check if the host data is already allocated
      if (!host_values.is_allocated())
        throw std::runtime_error(
            "SplineInterpolator1D: You probably called update() on a copied instance. This is not allowed. "
            "You need to call update() on the original instance.");

      // Copy values from input view
      Kokkos::deep_copy(host_values, view);

      // Build the spline coefficients
      build_y2(lower_y1, upper_y1);

      // Copy data to device
      Kokkos::deep_copy(device_values, host_values);
      Kokkos::deep_copy(device_coeffs, host_coeffs);
    }

    NT operator[](size_t i) const
    {
      // Check if the host data is already allocated
      if (!host_values.is_allocated())
        throw std::runtime_error(
            "SplineInterpolator1D: You probably called operator[]() on a copied instance. This is not allowed. "
            "You need to call operator[]() on the original instance.");

      return host_values(i);
    }

    /**
     * @brief Interpolate the data at a given point.
     *
     * @param x the point at which to interpolate
     * @return NT the interpolated value
     */
    /**
     * @brief Map a physical coordinate onto the (clamped) grid index.
     *
     * Split out of operator() because it is the expensive half: for a logarithmic axis
     * Coordinates::backward is a fp64 log1p, which costs ~200 fp64 instructions on current NVIDIA
     * parts, against ~10 for the spline evaluation itself. A generated kernel that evaluates many
     * dressings at the SAME momentum pays that log1p once per dressing, because each interpolator
     * owns its own `coordinates` members and the compiler cannot prove they are equal across
     * objects — so it cannot CSE the transform. Hoisting index() lets the caller pay it once.
     *
     * The returned index is only meaningful for interpolators sharing this coordinate system.
     */
    ctype KOKKOS_FUNCTION index(const typename Coordinates::ctype x) const
    {
      // m_fast_index is uniform across a launch (sampled once at construction from the
      // fastInterpLookups parameter), so this branch never diverges.
      if (m_fast_index) return index_tab(x);
      return coordinates.backward(x);
    }

    /**
     * @brief Tabulated inverse transform: binary search over the grid node positions plus a
     * per-cell Chebyshev polynomial (Clenshaw evaluation), ~50 device ops against the ~155-400 of
     * the analytic backward() on the logarithmic grids. Built at construction against the exact
     * analytic transform; reproduces it to <= 1e-12 in index units (verified by a dense sweep in
     * build_inverse_table). The input is clamped to the grid span first — identical post-at()
     * behaviour, since at() clamps the index anyway.
     */
    ctype KOKKOS_FUNCTION index_tab(const ctype y_in) const
    {
      const int n = static_cast<int>(size);
      const ctype y = Kokkos::max(device_fastidx(0), Kokkos::min(y_in, device_fastidx(n - 1)));
      // largest cell lo in [0, n-2] with node(lo) <= y
      int lo = 0, len = n - 1;
      while (len > 1) {
        const int half = len >> 1;
        lo = (device_fastidx(lo + half) <= y) ? lo + half : lo;
        len -= half;
      }
      const int base = n + lo * (m_inv_degree + 2);
      const ctype t = ctype(2) * (y - device_fastidx(lo)) * device_fastidx(base) - ctype(1);
      // Clenshaw over the Chebyshev coefficients (the fit is of backward itself, absolute index)
      ctype b1 = 0, b2 = 0;
      for (int k = m_inv_degree; k >= 1; --k) {
        const ctype b0 = ctype(2) * t * b1 - b2 + device_fastidx(base + 1 + k);
        b2 = b1;
        b1 = b0;
      }
      return t * b1 - b2 + device_fastidx(base + 1);
    }

    /**
     * @brief Interpolate at a grid index previously obtained from index().
     *
     * Clamping lives here, not in index(), so that index() depends ONLY on the coordinate system.
     * That is exactly the precondition under which a caller may share one index across several
     * interpolators; folding a size-dependent clamp into it would silently break sharing between
     * interpolators of different length.
     */
    NT KOKKOS_FUNCTION at(const ctype raw_idx) const
    {
      // Clamp the index to the range [0, size - 1]
      const ctype idx = Kokkos::max(static_cast<ctype>(0),
                                    Kokkos::min(raw_idx, static_cast<ctype>(size - 1)));
      const size_t lidx = Kokkos::min(size_t(Kokkos::floor(idx)), size - 2);
      const size_t uidx = lidx + 1;
      // t is the fractional part of the index
      const ctype t = idx - lidx;

      // SoA layout: separate reads from values and coefficients views for coalesced GPU access
      const NT lower = device_values(lidx);
      const NT upper = device_values(uidx);
      const NT cl = device_coeffs(lidx);
      const NT cu = device_coeffs(uidx);

      const ctype tm1 = t - 1;
      const NT cubic = t * tm1 * ((t + 1) * cl - (t - 2) * cu);

      if constexpr (std::is_arithmetic_v<NT>)
        return Kokkos::fma(t, upper, Kokkos::fma(-t, lower, lower)) + cubic; // linear + cubic
      else
        return t * upper + (1 - t) * lower + cubic; // linear + cubic
    }

    /**
     * @brief Interpolate the data at a given point.
     *
     * @param x the point at which to interpolate
     * @return NT the interpolated value
     */
    NT KOKKOS_FUNCTION operator()(const typename Coordinates::ctype x) const { return at(index(x)); }

    auto &CPU() const { return get_on<CPU_memory>(); }
    auto &GPU() const { return get_on<GPU_memory>(); }

    template <typename MemorySpace> auto &get_on() const
    {
      // Check if the host data is already allocated
      if (!host_values.is_allocated())
        throw std::runtime_error(
            "SplineInterpolator1D: You probably called get_on() on a copied instance. This is not allowed. "
            "You need to call get_on() on the original instance.");

      if constexpr (std::is_same_v<MemorySpace, DefaultMemorySpace>) {
        // remove constness
        return const_cast<SplineInterpolator1D<NT, Coordinates, MemorySpace> &>(*this);
      } else {
        // Create a new instance with the same data but in the requested memory space
        if (other_instance == nullptr) {
          other_instance = new SplineInterpolator1D<NT, Coordinates, MemorySpace>(coordinates);
          owns_other_instance = true;
          other_instance->other_instance = const_cast<std::decay_t<decltype(*this)> *>(this);
        }
        // Copy the data from the current instance to the new one
        other_instance->update(host_values);
        // Return the new instance
        return *other_instance;
      }
    }

    /**
     * @brief Get the coordinate system of the data.
     *
     * @return const Coordinates& the coordinate system
     */
    const Coordinates &get_coordinates() const { return coordinates; }

    NT *data()
    {
      if (!host_values.is_allocated())
        throw std::runtime_error(
            "SplineInterpolator1D: You probably called data() on a copied instance. This is not allowed. "
            "You need to call data() on the original instance.");
      return host_values.data();
    }

    friend class SplineInterpolator1D<NT, Coordinates, other_memory_space>;

  private:
    const Coordinates coordinates;
    const size_t size;

    using ValueViewType = Kokkos::View<NT *, DefaultMemorySpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using CoeffViewType = Kokkos::View<NT *, DefaultMemorySpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using HostValueViewType = typename ValueViewType::host_mirror_type;
    using HostCoeffViewType = typename CoeffViewType::host_mirror_type;

    ValueViewType device_values;
    CoeffViewType device_coeffs;
    HostValueViewType host_values;
    HostCoeffViewType host_coeffs;

    // Tabulated inverse transform ("fastInterpLookups"). Layout of device_fastidx:
    //   [0 .. size-1]                                  node positions forward(i)
    //   [size + i*(deg+2) .. size + (i+1)*(deg+2)-1]   cell i record: { invw, ch0 .. ch<deg> }
    // where p(t) = Clenshaw(ch, 2*(y-node_i)*invw - 1) approximates backward(y) over cell i
    // (backward returns the absolute fractional index, so no per-cell offset is needed). Empty
    // (and m_fast_index false) unless the fastInterpLookups parameter was set when this instance
    // was constructed.
    using FastIdxViewType = Kokkos::View<ctype *, DefaultMemorySpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    FastIdxViewType device_fastidx;
    int m_inv_degree = 0;
    bool m_fast_index = false;

    /**
     * @brief Build the tabulated inverse of coordinates.backward on the host and upload it.
     *
     * Chebyshev interpolation per cell, degree chosen globally as the smallest of
     * {6, 8, 10, 12, 14, 16} for which a dense verification sweep against the analytic backward
     * stays below 1e-12 (index units). Throws if even degree 16 leaves more than 1e-10 — that
     * would mean the transform is not smooth on a cell, which no supported coordinate class
     * produces.
     */
    void build_inverse_table()
    {
      const int n = static_cast<int>(size);
      const int ncells = n - 1;
      std::vector<double> nodes(n);
      for (int i = 0; i < n; ++i)
        nodes[i] = static_cast<double>(coordinates.forward(static_cast<ctype>(i)));

      const double tol_target = 1e-12, tol_hard = 1e-10;
      std::vector<double> coeffs; // ncells * (deg+1)
      int deg = 0;
      double worst = std::numeric_limits<double>::infinity();
      for (const int d : {6, 8, 10, 12, 14, 16}) {
        deg = d;
        coeffs.assign(static_cast<size_t>(ncells) * (d + 1), 0.0);
        worst = 0.0;
        for (int i = 0; i < ncells; ++i) {
          const double a = nodes[i], w = nodes[i + 1] - nodes[i];
          // Chebyshev interpolation coefficients from the d+1 Chebyshev points of the cell
          std::vector<double> fv(d + 1);
          for (int j = 0; j <= d; ++j) {
            const double xj = std::cos(M_PI * (j + 0.5) / (d + 1)); // in (-1, 1)
            const double yj = a + 0.5 * (xj + 1.0) * w;
            fv[j] = static_cast<double>(coordinates.backward(static_cast<ctype>(yj)));
          }
          for (int k = 0; k <= d; ++k) {
            double s = 0.0;
            for (int j = 0; j <= d; ++j)
              s += fv[j] * std::cos(k * M_PI * (j + 0.5) / (d + 1));
            coeffs[static_cast<size_t>(i) * (d + 1) + k] = (k == 0 ? 1.0 : 2.0) / (d + 1) * s;
          }
          // dense verification sweep over the cell
          for (int j = 0; j <= 40; ++j) {
            const double y = a + (j / 40.0) * w;
            const double t = 2.0 * (y - a) / w - 1.0;
            double b1 = 0.0, b2 = 0.0;
            for (int k = d; k >= 1; --k) {
              const double b0 = 2.0 * t * b1 - b2 + coeffs[static_cast<size_t>(i) * (d + 1) + k];
              b2 = b1;
              b1 = b0;
            }
            const double p = t * b1 - b2 + coeffs[static_cast<size_t>(i) * (d + 1)];
            worst = std::max(worst, std::abs(p - static_cast<double>(coordinates.backward(static_cast<ctype>(y)))));
          }
        }
        if (worst < tol_target) break;
      }
      if (worst > tol_hard)
        throw std::runtime_error("SplineInterpolator1D: fastInterpLookups inverse table failed to reach 1e-10 (worst " +
                                 std::to_string(worst) + ") — coordinates.backward is not smooth on a cell?");

      // pack: nodes + per-cell records { invw, ch0 + i, ch1 .. chdeg }
      device_fastidx = FastIdxViewType("SplineInterpolator1D_fastidx",
                                       static_cast<size_t>(n) + static_cast<size_t>(ncells) * (deg + 2));
      auto host_tab = Kokkos::create_mirror_view(device_fastidx);
      for (int i = 0; i < n; ++i)
        host_tab(i) = static_cast<ctype>(nodes[i]);
      for (int i = 0; i < ncells; ++i) {
        const size_t base = static_cast<size_t>(n) + static_cast<size_t>(i) * (deg + 2);
        host_tab(base) = static_cast<ctype>(1.0 / (nodes[i + 1] - nodes[i]));
        for (int k = 0; k <= deg; ++k)
          host_tab(base + 1 + k) = static_cast<ctype>(coeffs[static_cast<size_t>(i) * (deg + 1) + k]);
      }
      Kokkos::deep_copy(device_fastidx, host_tab);
      m_inv_degree = deg;
      m_fast_index = true;
    }

    mutable SplineInterpolator1D<NT, Coordinates, other_memory_space> *other_instance = nullptr;
    mutable bool owns_other_instance = false;

    void build_y2(const ctype lower_y1, const ctype upper_y1)
    {
      NT p, qn, sig, un;
      std::vector<NT> u(size - 1);

      if (!std::isfinite(lower_y1) || lower_y1 >= std::numeric_limits<ctype>::max() / 2)
        host_coeffs(0) = u[0] = 0.0;
      else {
        host_coeffs(0) = -0.5;
        u[0] = 3.0 * ((host_values(1) - host_values(0)) - lower_y1);
      }
      for (size_t i = 1; i < size - 1; i++) {
        sig = 0.5;
        p = sig * host_coeffs(i - 1) + 2.0;
        host_coeffs(i) = (sig - 1.0) / p;
        u[i] = (host_values(i + 1) - host_values(i)) - (host_values(i) - host_values(i - 1));
        u[i] = (6.0 * u[i] / 2. - sig * u[i - 1]) / p;
      }
      if (!std::isfinite(upper_y1) || upper_y1 >= std::numeric_limits<ctype>::max() / 2)
        qn = un = 0.0;
      else {
        qn = 0.5;
        un = 3.0 * (upper_y1 - (host_values(size - 1) - host_values(size - 2)));
      }
      host_coeffs(size - 1) = (un - qn * u[size - 2]) / (qn * host_coeffs(size - 2) + 1);
      for (int k = size - 2; k >= 0; k--)
        host_coeffs(k) = host_coeffs(k) * host_coeffs(k + 1) + u[k];

      // Precompute division by 6 so operator() avoids per-call divides
      for (size_t k = 0; k < size; ++k)
        host_coeffs(k) /= (ctype)6;
    }
  };
} // namespace DiFfRG
