#include <DiFfRG/model/model.hh>

#include <catch2/catch_all.hpp>

#include <array>
#include <tuple>

namespace
{
  using namespace DiFfRG;
  using namespace dealii;

  constexpr uint dim = 2;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;

  class O2_Model_VI_B_I : public def::AbstractModel<O2_Model_VI_B_I, Components>,
                          public def::LLFFlux<O2_Model_VI_B_I>,
                          public def::FlowBoundaries<O2_Model_VI_B_I>,
                          public def::OriginOddLinearExtrapolationBoundaries<O2_Model_VI_B_I>,
                          public def::AD<O2_Model_VI_B_I>
  {
  public:
    template <typename Vector> void initial_condition([[maybe_unused]] const Point<dim> &x, Vector &values) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
    }

    template <typename NumberType, typename Solutions>
    void KurganovTadmor_advection_flux(std::array<Tensor<1, dim, NumberType>, 2> &F_i,
                                       [[maybe_unused]] const Point<dim> &x,
                                       [[maybe_unused]] const Solutions &sol) const
    {
      for (auto &flux : F_i)
        flux = 0.0;
    }

    template <typename NumberType, typename Solutions>
    void flux(std::array<Tensor<1, dim, NumberType>, 2> &F_i, [[maybe_unused]] const Point<dim> &x,
              [[maybe_unused]] const Solutions &sol) const
    {
      for (auto &flux : F_i)
        flux = 0.0;
    }

    template <typename NumberType, typename Solutions>
    void source(std::array<NumberType, 2> &s_i, [[maybe_unused]] const Point<dim> &x,
                [[maybe_unused]] const Solutions &sol) const
    {
      s_i[0] = 0.0;
      s_i[1] = 0.0;
    }
  };
} // namespace

TEST_CASE("O2 Model VI-B-I exposes zero hooks", "[O2 Model][FV][KT][2d]")
{
  O2_Model_VI_B_I model;

  std::array<double, 2> values{{1.0, -1.0}};
  model.initial_condition(Point<dim>(0.25, 0.5), values);
  CHECK(values[0] == Catch::Approx(0.0));
  CHECK(values[1] == Catch::Approx(0.0));

  std::array<Tensor<1, dim, double>, 2> advection_fluxes;
  std::array<Tensor<1, dim, double>, 2> fluxes;
  std::array<double, 2> sources{{1.0, -1.0}};

  model.KurganovTadmor_advection_flux(advection_fluxes, Point<dim>(0.25, 0.5), std::tuple<>{});
  model.flux(fluxes, Point<dim>(0.25, 0.5), std::tuple<>{});
  model.source(sources, Point<dim>(0.25, 0.5), std::tuple<>{});

  for (const auto &flux : advection_fluxes) {
    CHECK(flux[0] == Catch::Approx(0.0));
    CHECK(flux[1] == Catch::Approx(0.0));
  }
  for (const auto &flux : fluxes) {
    CHECK(flux[0] == Catch::Approx(0.0));
    CHECK(flux[1] == Catch::Approx(0.0));
  }
  CHECK(sources[0] == Catch::Approx(0.0));
  CHECK(sources[1] == Catch::Approx(0.0));
}

TEST_CASE("O2 Model VI-B-I uses origin-odd lower boundaries", "[O2 Model][FV][KT][2d]")
{
  using namespace def::BoundaryStencilIndex;

  O2_Model_VI_B_I model;
  def::BoundaryStencilValues<dim, double, 2> u_stencil{
      {{{11.0, 21.0}}, {{12.0, 22.0}}, {{5.0, 7.0}}, {{2.0, 13.0}}, {{4.0, 17.0}}, {{0.0, 0.0}}, {{0.0, 0.0}}}};
  def::BoundaryStencilPoints<dim> x_stencil{{Point<dim>(8.0, 8.0),
                                             Point<dim>(9.0, 9.0),
                                             Point<dim>(0.5, 3.0),
                                             Point<dim>(1.5, 3.0),
                                             Point<dim>(2.5, 3.0),
                                             Point<dim>(),
                                             Point<dim>()}};

  REQUIRE(model.apply_boundary_stencil(u_stencil, x_stencil, Point<dim>(0.0, 3.0)));

  CHECK(u_stencil[physical_cell][0] == Catch::Approx(0.0));
  CHECK(u_stencil[lower_inner][0] == Catch::Approx(-2.0));
  CHECK(u_stencil[lower_outer][0] == Catch::Approx(-4.0));
  CHECK(u_stencil[physical_cell][1] == Catch::Approx(7.0));
  CHECK(u_stencil[lower_inner][1] == Catch::Approx(7.0));
  CHECK(u_stencil[lower_outer][1] == Catch::Approx(13.0));
}
