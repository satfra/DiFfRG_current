// DiFfRG
#include <DiFfRG/common/quadrature/quadrature.hh>

// external libraries
#include <gsl/gsl_integration.h>

namespace DiFfRG
{
  template <typename NT> Quadrature<NT>::Quadrature(const size_t order, const QuadratureType _t) { reinit(order, _t); }
  template <typename NT> Quadrature<NT>::Quadrature() {}

  template <typename NT> void Quadrature<NT>::reinit(const size_t order, const QuadratureType _t)
  {
    this->order = order;
    this->_t = _t;

    gsl_integration_fixed_workspace *w;
    const gsl_integration_fixed_type *T;

    // create gsl workspace and set quadrature type
    std::string name;
    switch (_t) {
    case QuadratureType::legendre:
      T = gsl_integration_fixed_legendre;
      name = "legendre";
      break;
    case QuadratureType::chebyshev:
      T = gsl_integration_fixed_chebyshev;
      name = "chebyshev";
      break;
    case QuadratureType::hermite:
      T = gsl_integration_fixed_hermite;
      name = "hermite";
      break;
    case QuadratureType::jacobi:
      T = gsl_integration_fixed_jacobi;
      name = "jacobi";
      break;
    case QuadratureType::laguerre:
      T = gsl_integration_fixed_laguerre;
      name = "laguerre";
      break;
    default:
      throw std::runtime_error("Unknown quadrature type");
    }
    w = gsl_integration_fixed_alloc(T, order, 0.0, 1.0, 0.0, 0.0);

    // get nodes and weights
    const double *gsl_nodes = gsl_integration_fixed_nodes(w);
    const double *gsl_weights = gsl_integration_fixed_weights(w);

    device_nodes = Kokkos::View<NT *, Kokkos::DefaultExecutionSpace::memory_space>(
        "device_nodes_" + name + "_" + std::to_string(order), order);
    device_weights = Kokkos::View<NT *, Kokkos::DefaultExecutionSpace::memory_space>(
        "device_weights_" + name + "_" + std::to_string(order), order);

    // create a host mirror view
    auto host_mirror_nodes = Kokkos::create_mirror_view(device_nodes);
    auto host_mirror_weights = Kokkos::create_mirror_view(device_weights);

    // copy data from gsl to host mirror view
    for (size_t i = 0; i < order; ++i) {
      host_mirror_nodes(i) = gsl_nodes[i];
      host_mirror_weights(i) = gsl_weights[i];
    }

    // copy data from host mirror view to device view
    Kokkos::deep_copy(device_nodes, host_mirror_nodes);
    Kokkos::deep_copy(device_weights, host_mirror_weights);

    // copy data from host mirror view to host view
    host_nodes = Kokkos::View<NT *, Kokkos::DefaultHostExecutionSpace::memory_space>(
        "host_nodes_" + name + "_" + std::to_string(order), order);
    host_weights = Kokkos::View<NT *, Kokkos::DefaultHostExecutionSpace::memory_space>(
        "host_weights_" + name + "_" + std::to_string(order), order);
    Kokkos::deep_copy(host_nodes, host_mirror_nodes);
    Kokkos::deep_copy(host_weights, host_mirror_weights);

    // free gsl workspace
    gsl_integration_fixed_free(w);
  }

  template <typename NT> size_t Quadrature<NT>::size() const { return order; }
  template <typename NT> QuadratureType Quadrature<NT>::get_type() const { return _t; }

  // explicit instantiation
  template class Quadrature<double>;
  template class Quadrature<float>;
} // namespace DiFfRG