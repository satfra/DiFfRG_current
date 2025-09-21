// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/quadrature/matsubara.hh>
#include <DiFfRG/common/quadrature/quadrature.hh>

// Standard library
#include <cmath>
#include <string>

namespace DiFfRG
{
  template <typename NT> int MatsubaraQuadrature<NT>::predict_size(const NT T, const NT typical_E, const int step)
  {
    const NT relative_distance = abs(typical_E) / abs(2 * M_PI * T + 1e-16);
    // From some testing, switching here to the vacuum quadrature will generate relative errors of order 5e-6 when using
    // the default arguments
    if (is_close(T, NT{}) || relative_distance > 1e+4) return -vacuum_quad_size;

    const NT E_max = 1e4 * precision_factor * std::abs(typical_E);
    int size = 5 + int(std::sqrt(4 * E_max / (M_PI * M_PI * std::abs(T))));
    size = (int)std::ceil(size / (double)step) * step;
    return size;
  }

  template <typename NT>
  MatsubaraQuadrature<NT>::MatsubaraQuadrature(const NT T, const NT typical_E, const int step, const int min_size,
                                               const int max_size, const int vacuum_quad_size,
                                               const int precision_factor)
  {
    reinit(T, typical_E, step, min_size, max_size, vacuum_quad_size, precision_factor);
  }

  template <typename NT> MatsubaraQuadrature<NT>::MatsubaraQuadrature() : m_size(0), vacuum_quad_size(64) {}

  template <typename NT>
  void MatsubaraQuadrature<NT>::reinit(const NT T, const NT typical_E, const int step, const int min_size,
                                       const int max_size, const int vacuum_quad_size, const int precision_factor)
  {
    if (precision_factor <= 0)
      this->precision_factor = 1;
    else
      this->precision_factor = precision_factor;

    // This is ludicrously low, but let's cut somewhere
    if (vacuum_quad_size <= 16)
      this->vacuum_quad_size = 16;
    else
      this->vacuum_quad_size = vacuum_quad_size;

    if (max_size < min_size) throw std::invalid_argument("MatsubaraQuadrature: max_size must be larger than min_size.");

    this->T = T;
    this->typical_E = typical_E;

    // Determine the number of nodes in the quadrature rule.
    m_size = predict_size(T, typical_E, step);
    // If m_size is negative, we use the vacuum quadrature
    if (m_size < 0) {
      m_size = abs(m_size);
      reinit_0();
      return;
    }
    m_size = std::max(min_size, std::min(max_size, m_size));

    // construct the recurrence relation for the quadrature rule from [1]
    std::vector<NT> a(m_size, 0.);
    std::vector<NT> b(m_size, 0.);

    for (int j = 0; j < m_size; ++j) {
      const double j1 = j + 1;
      a[j] = 2 * powr<2>(M_PI) / (4 * j + 1) / (4 * j + 5);
      b[j] = powr<4>(M_PI) / ((4 * j1 - 1) * (4 * j1 + 3)) / powr<2>(4 * j1 + 1);
    }
    a[0] = powr<2>(M_PI) / 15.;

    const NT mu0 = powr<2>(M_PI) / 6.;

    std::vector<NT> x(m_size, 0.);
    std::vector<NT> w(m_size, 0.);

    // compute the nodes and weights of the quadrature rule
    make_quadrature(a, b, mu0, x, w);

    // normalize the weights and scale the nodes
    for (int i = 0; i < m_size; ++i) {
      w[i] = T * w[i] / x[i];
      x[i] = 2. * M_PI * T / std::sqrt(x[i]);
    }

    write_data(x, w);
  }

  // TODO: rewrite this function
  template <typename NT> void MatsubaraQuadrature<NT>::reinit_0()
  {
    this->T = 0;
    m_size = abs(m_size);
    // ensure that m_size is divisible by 3, so that we can divide it into 2/3 and 1/3 parts
    while (m_size % 3 != 0)
      m_size++;
    if (is_close(typical_E, NT(0))) this->typical_E = 1.;

    // obtain a gauss-legendre quadrature rule for the interval [0, 1]
    Quadrature<NT> quad_up(m_size / 3, QuadratureType::legendre);
    Quadrature<NT> quad_down(m_size / 3 * 2, QuadratureType::legendre);

    // resize the nodes and weights
    std::vector<NT> x(m_size, 0.);
    std::vector<NT> w(m_size, 0.);

    // strategy: divide into two parts, one with linear and one with logarithmic scaling
    // the dividing point is somewhat above the typical energy scale
    const long double div = 3 * abs(typical_E);

    // the nodes with a linear scale
    for (int i = 0; i < m_size / 3 * 2; ++i) {
      x[i] = quad_down.template nodes<CPU_memory>()[i] * div;
      w[i] = quad_down.template weights<CPU_memory>()[i] * div / NT(2 * M_PI);
    }

    // the nodes with a logarithmic scale
    using std::log, std::exp, std::abs, std::min;
    const long double extent = 1e6 * abs(typical_E);
    const long double log_start = log(div);
    const long double log_ext = log(extent / div);
    for (int i = 0; i < m_size / 3; ++i) {
      x[i + m_size / 3 * 2] = exp(log_start + log_ext * quad_up.template nodes<CPU_memory>()[i]);
      w[i + m_size / 3 * 2] =
          (quad_up.template weights<CPU_memory>()[i] * log_ext * x[i + m_size / 3 * 2]) / NT(2 * M_PI);
    }

    write_data(x, w);
  }

  template <typename NT> void MatsubaraQuadrature<NT>::write_data(const std::vector<NT> &x, const std::vector<NT> &w)
  {
    std::string name = "matsubara_T" + std::to_string(T) + "_typicalE" + std::to_string(typical_E);

    // copy nodes and weights to std::vector
    device_nodes = Kokkos::View<NT *, Kokkos::DefaultExecutionSpace::memory_space>(
        "device_nodes_" + name + "_" + std::to_string(m_size), m_size);
    device_weights = Kokkos::View<NT *, Kokkos::DefaultExecutionSpace::memory_space>(
        "device_weights_" + name + "_" + std::to_string(m_size), m_size);

    // create a host mirror view
    auto host_mirror_nodes = Kokkos::create_mirror_view(device_nodes);
    auto host_mirror_weights = Kokkos::create_mirror_view(device_weights);

    // copy data from gsl to host mirror view
    for (int i = 0; i < m_size; ++i) {
      host_mirror_nodes(i) = x[i];
      host_mirror_weights(i) = w[i];
    }

    // copy data from host mirror view to device view
    Kokkos::deep_copy(device_nodes, host_mirror_nodes);
    Kokkos::deep_copy(device_weights, host_mirror_weights);

    // copy data from host mirror view to host view
    host_nodes = Kokkos::View<NT *, Kokkos::DefaultHostExecutionSpace::memory_space>(
        "host_nodes_" + name + "_" + std::to_string(m_size), m_size);
    host_weights = Kokkos::View<NT *, Kokkos::DefaultHostExecutionSpace::memory_space>(
        "host_weights_" + name + "_" + std::to_string(m_size), m_size);
    Kokkos::deep_copy(host_nodes, host_mirror_nodes);
    Kokkos::deep_copy(host_weights, host_mirror_weights);
  }

  template <typename NT> size_t MatsubaraQuadrature<NT>::size() const { return m_size; }
  template <typename NT> NT MatsubaraQuadrature<NT>::get_T() const { return T; }
  template <typename NT> NT MatsubaraQuadrature<NT>::get_typical_E() const { return typical_E; }

  // explicit instantiation
  template class MatsubaraQuadrature<double>;
  template class MatsubaraQuadrature<float>;
} // namespace DiFfRG