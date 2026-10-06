#pragma once

// DiFfRG
#include <DiFfRG/common/tuples.hh>
#include <DiFfRG/model/ad.hh>
#include <DiFfRG/physics/integration/point_arg.hh>

// external libraries
#include <autodiff/forward/real.hpp>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <tbb/tbb.h>

// standard library
#include <algorithm>
#include <array>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief Per-point solution tuple with the names of CG::fe_tie, so the per-point model callbacks
   * (flux, source, boundary_numflux and their jacobians) accept it.
   */
  template <typename... T> auto batch_tie(T &&...t)
  {
    return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "fe_derivatives", "fe_hessians", "extractors",
                                                     "variables", "cell_width">>(std::tie(t...));
  }

  namespace internal
  {
    /// v = n copies of value, filled in parallel: these buffers hold up to millions of entries.
    template <typename T> void parallel_assign(std::vector<T> &v, const size_t n, const T &value)
    {
      v.resize(n);
      tbb::parallel_for(tbb::blocked_range<size_t>(0, n, 1 << 14), [&](const tbb::blocked_range<size_t> &r) {
        std::fill(v.begin() + r.begin(), v.begin() + r.end(), value);
      });
    }
  } // namespace internal

  /// Copy of the solution at a single point of a PointBatch.
  template <int dim, typename NT, size_t n_fe> struct PointState {
    std::array<NT, n_fe> values;
    std::array<dealii::Tensor<1, dim, NT>, n_fe> derivatives;
    std::array<dealii::Tensor<2, dim, NT>, n_fe> hessians;
    double cell_width;
  };

  /**
   * @brief The FE solution at a set of points: values, derivatives and (optionally) hessians of every
   * FE function, plus the position and cell width of each point. Extractors and variables are shared by
   * all points.
   *
   * Storage is component-major with the points fastest, so values(c), derivatives(c, d) and
   * hessians(c, d1, d2) are contiguous columns that can be passed directly as per-point arguments to
   * an integrator's map_points().
   *
   * Point i means the same point in every column, and in every column of the output FluxSourceBatch.
   * The order of the points carries no meaning for the model.
   */
  template <int dim_, typename NT, size_t n_fe, typename Extractors, typename Variables> class PointBatch
  {
  public:
    static constexpr int dim = dim_;
    static constexpr size_t n_fe_functions = n_fe;
    using number_type = NT;
    using variables_type = Variables;

    void reinit(const size_t n_points, const bool with_hessians)
    {
      n = n_points;
      m_hessians = with_hessians;
      state.resize(n * n_columns());
      positions.resize(n * dim);
      widths.resize(n);
    }

    void set_shared(const Extractors &extractors, const Variables &variables)
    {
      m_extractors = &extractors;
      m_variables = &variables;
    }

    size_t size() const { return n; }
    bool has_hessians() const { return m_hessians; }

    PointArray<NT> values(const size_t c) const { return {column(value_column(c)), n}; }
    PointArray<NT> derivatives(const size_t c, const size_t d) const { return {column(derivative_column(c, d)), n}; }
    PointArray<NT> hessians(const size_t c, const size_t d1, const size_t d2) const
    {
      if (!m_hessians)
        throw std::logic_error("PointBatch: hessians were not gathered; the model sets batch_reads_hessians = false.");
      return {column(hessian_column(c, d1, d2)), n};
    }
    PointArray<double> coordinates(const size_t d) const { return {positions.data() + d * n, n}; }

    dealii::Point<dim> x(const size_t i) const
    {
      dealii::Point<dim> p;
      for (int d = 0; d < dim; ++d)
        p[d] = positions[d * n + i];
      return p;
    }
    double cell_width(const size_t i) const { return widths[i]; }

    const Extractors &extractors() const { return *m_extractors; }
    const Variables &variables() const { return *m_variables; }

    void load(const size_t i, PointState<dim, NT, n_fe> &s) const
    {
      for (size_t c = 0; c < n_fe; ++c) {
        s.values[c] = column(value_column(c))[i];
        for (int d1 = 0; d1 < dim; ++d1) {
          s.derivatives[c][d1] = column(derivative_column(c, d1))[i];
          for (int d2 = 0; d2 < dim; ++d2)
            s.hessians[c][d1][d2] = m_hessians ? column(hessian_column(c, d1, d2))[i] : NT(0);
        }
      }
      s.cell_width = widths[i];
    }

    // Writable access for whoever fills the batch.
    NT &value(const size_t c, const size_t i) { return column(value_column(c))[i]; }
    NT &derivative(const size_t c, const size_t d, const size_t i) { return column(derivative_column(c, d))[i]; }
    NT &hessian(const size_t c, const size_t d1, const size_t d2, const size_t i)
    {
      return column(hessian_column(c, d1, d2))[i];
    }
    double &coordinate(const size_t d, const size_t i) { return positions[d * n + i]; }
    double &width(const size_t i) { return widths[i]; }

  private:
    size_t n_columns() const { return n_fe * (1 + dim + (m_hessians ? dim * dim : 0)); }
    static constexpr size_t value_column(const size_t c) { return c; }
    static constexpr size_t derivative_column(const size_t c, const size_t d) { return n_fe + c * dim + d; }
    static constexpr size_t hessian_column(const size_t c, const size_t d1, const size_t d2)
    {
      return n_fe * (1 + dim) + (c * dim + d1) * dim + d2;
    }
    const NT *column(const size_t col) const { return state.data() + col * n; }
    NT *column(const size_t col) { return state.data() + col * n; }

    size_t n = 0;
    bool m_hessians = true;
    std::vector<NT> state;
    std::vector<double> positions;
    std::vector<double> widths;
    const Extractors *m_extractors = nullptr;
    const Variables *m_variables = nullptr;
  };

  /**
   * @brief Flux and source of every FE function at the points of a PointBatch, as contiguous columns.
   * reinit() zeroes everything, so a model only writes what it computes.
   */
  template <int dim, typename NT, size_t n_fe> class FluxSourceBatch
  {
  public:
    void reinit(const size_t n_points)
    {
      n = n_points;
      internal::parallel_assign(data, n * n_fe * (dim + 1), NT(0));
    }
    size_t size() const { return n; }

    NT *flux(const size_t c, const size_t d) { return data.data() + (c * dim + d) * n; }
    const NT *flux(const size_t c, const size_t d) const { return data.data() + (c * dim + d) * n; }
    NT *source(const size_t c) { return data.data() + (n_fe * dim + c) * n; }
    const NT *source(const size_t c) const { return data.data() + (n_fe * dim + c) * n; }

    void store(const size_t i, const std::array<dealii::Tensor<1, dim, NT>, n_fe> &F, const std::array<NT, n_fe> &S)
    {
      for (size_t c = 0; c < n_fe; ++c) {
        for (int d = 0; d < dim; ++d)
          flux(c, d)[i] = F[c][d];
        source(c)[i] = S[c];
      }
    }

  private:
    size_t n = 0;
    std::vector<NT> data;
  };

  /**
   * @brief Derivatives of flux and source at one point with respect to the FE values, derivatives,
   * hessians and the extractors. Same layout as the per-point jacobian_flux_source* callbacks; at
   * boundary points the flux blocks hold the boundary numflux and the source blocks stay zero.
   */
  template <int dim, size_t n_fe, size_t n_extr> struct PointJacobian {
    using T1 = dealii::Tensor<1, dim, double>;
    SimpleMatrix<T1, n_fe> j_flux;
    SimpleMatrix<dealii::Tensor<1, dim, T1>, n_fe> j_grad_flux;
    SimpleMatrix<dealii::Tensor<1, dim, dealii::Tensor<2, dim, double>>, n_fe> j_hess_flux;
    SimpleMatrix<T1, n_fe, n_extr> j_extr_flux;
    SimpleMatrix<double, n_fe> j_source;
    SimpleMatrix<T1, n_fe> j_grad_source;
    SimpleMatrix<dealii::Tensor<2, dim, double>, n_fe> j_hess_source;
    SimpleMatrix<double, n_fe, n_extr> j_extr_source;
  };

  /**
   * @brief Buffers of the seed-stacked AD jacobian (internal::seed_stacked_jacobian), kept by the caller
   * across calls: they hold n_seeds copies of the batch, and allocating them anew every time costs as
   * much as the evaluation itself.
   */
  template <int dim, size_t n_fe, size_t n_extr, typename Variables> struct SeedStackWorkspace {
    PointBatch<dim, autodiff::real, n_fe, std::array<autodiff::real, n_extr>, Variables> batch;
    FluxSourceBatch<dim, autodiff::real, n_fe> out;
    std::array<autodiff::real, n_extr> extractors;
    std::vector<dealii::Tensor<1, dim>> normals;
  };

  /**
   * @brief Model traits read by the batched assemblers. A model declares
   * `static constexpr bool batch_reads_derivatives = false` (or `batch_reads_hessians`) when neither its
   * flux nor its source reads that input: the batch then skips it, and so do the AD jacobian seeds.
   */
  template <typename Model> constexpr bool batch_reads_derivatives()
  {
    if constexpr (requires { Model::batch_reads_derivatives; })
      return Model::batch_reads_derivatives;
    else
      return true;
  }
  template <typename Model> constexpr bool batch_reads_hessians()
  {
    if constexpr (requires { Model::batch_reads_hessians; })
      return Model::batch_reads_hessians;
    else
      return true;
  }

  namespace internal
  {
    /// f(i, x, sol) at every point i of the batch, in a flat parallel loop; sol is the point's solution tuple.
    template <typename Batch, typename F> void for_each_point(const Batch &batch, const F &f)
    {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, batch.size()), [&](const tbb::blocked_range<size_t> &r) {
        PointState<Batch::dim, typename Batch::number_type, Batch::n_fe_functions> s;
        for (size_t i = r.begin(); i != r.end(); ++i) {
          batch.load(i, s);
          f(i, batch.x(i),
            batch_tie(s.values, s.derivatives, s.hessians, batch.extractors(), batch.variables(), s.cell_width));
        }
      });
    }

    /**
     * @brief Flux and source at all points of a batch from the model's per-point flux and source. This is
     * def::AbstractModel::flux_source_batch's default.
     *
     * The per-point callbacks run on several threads at once; a model whose per-point flux is not thread
     * safe (e.g. it calls a GPU integrator's get()) must override flux_source_batch.
     */
    template <typename Model, typename Out, typename Batch>
    void flux_source_per_point(const Model &model, Out &out, const Batch &batch)
    {
      using NT = typename Batch::number_type;
      for_each_point(batch, [&](const size_t i, const auto &x, const auto &sol) {
        std::array<dealii::Tensor<1, Batch::dim, NT>, Batch::n_fe_functions> F{};
        std::array<NT, Batch::n_fe_functions> S{};
        model.flux(F, x, sol);
        model.source(S, x, sol);
        out.store(i, F, S);
      });
    }
  } // namespace internal

  /**
   * @brief Boundary numflux at all points of a boundary batch, into the flux columns of @p out: the
   * model's `boundary_numflux_batch(out, normals, batch)` if it has one (def::FlowBoundaries does),
   * otherwise its per-point boundary_numflux. normals[i] is the outward normal at point i.
   */
  template <typename Model, typename Out, typename Batch>
  void evaluate_boundary_numflux(const Model &model, Out &out,
                                 const std::vector<dealii::Tensor<1, Batch::dim>> &normals, const Batch &batch)
  {
    if constexpr (requires { model.boundary_numflux_batch(out, normals, batch); })
      model.boundary_numflux_batch(out, normals, batch);
    else {
      using NT = typename Batch::number_type;
      internal::for_each_point(batch, [&](const size_t i, const auto &x, const auto &sol) {
        std::array<dealii::Tensor<1, Batch::dim, NT>, Batch::n_fe_functions> F{};
        model.boundary_numflux(F, normals[i], x, sol);
        out.store(i, F, {});
      });
    }
  }

  namespace internal
  {
    /// One forward-mode AD seed direction among the FE inputs of a point.
    struct BatchSeed {
      enum Kind { value, derivative, hessian } kind;
      uint c, d1, d2;
    };

    // Forward-mode AD on autodiff::real: component 1 carries the derivative along the seeded direction.
    inline void seed_direction(autodiff::real &x) { x[1] = 1.; }
    inline void clear_direction(autodiff::real &x) { x[1] = 0.; }
    inline double along_direction(const autodiff::real &x) { return x[1]; }

    template <int dim, size_t n_fe>
    std::vector<BatchSeed> batch_seeds(const bool reads_derivatives, const bool reads_hessians)
    {
      std::vector<BatchSeed> seeds;
      for (uint c = 0; c < n_fe; ++c)
        seeds.push_back({BatchSeed::value, c, 0, 0});
      if (reads_derivatives)
        for (uint c = 0; c < n_fe; ++c)
          for (uint d = 0; d < dim; ++d)
            seeds.push_back({BatchSeed::derivative, c, d, 0});
      // A hessian is symmetric, so (d1, d2) and (d2, d1) are seeded together.
      if (reads_hessians)
        for (uint c = 0; c < n_fe; ++c)
          for (uint d1 = 0; d1 < dim; ++d1)
            for (uint d2 = d1; d2 < dim; ++d2)
              seeds.push_back({BatchSeed::hessian, c, d1, d2});
      return seeds;
    }

    /**
     * @brief Forward-mode AD jacobian of a batch evaluation, with all seed directions stacked along
     * the point axis: block b of the AD batch is a copy of the input batch with seed b set at every
     * point, so a single evaluation yields the jacobian column of seed b at all points. Seeds are
     * stacked up to max_stacked points per evaluation. Extractor seeds are one evaluation each, as the
     * extractors are shared by all points of a batch.
     *
     * @param evaluate callable(FluxSourceBatch<AD> &out, const PointBatch<AD> &batch, size_t n_blocks)
     * @param with_source whether the source columns carry a result (false at the boundary)
     */
    template <typename Evaluate, typename Batch, size_t n_extr>
    void seed_stacked_jacobian(
        const Evaluate &evaluate, std::vector<PointJacobian<Batch::dim, Batch::n_fe_functions, n_extr>> &J,
        const Batch &batch, const bool reads_derivatives, const size_t max_stacked, const bool with_source,
        SeedStackWorkspace<Batch::dim, Batch::n_fe_functions, n_extr, typename Batch::variables_type> &workspace)
    {
      constexpr int dim = Batch::dim;
      constexpr size_t n_fe = Batch::n_fe_functions;
      const size_t n = batch.size();

      parallel_assign(J, n, PointJacobian<dim, n_fe, n_extr>{});
      if (n == 0) return;

      auto &ad_extractors = workspace.extractors;
      for (size_t e = 0; e < n_extr; ++e)
        ad_extractors[e] = batch.extractors()[e];

      auto &ad_batch = workspace.batch;
      auto &ad_out = workspace.out;

      // Copy point i of the input into slot b * n + i of the AD batch, unseeded.
      const auto copy_block_point = [&](const size_t b, const size_t i) {
        const size_t j = b * n + i;
        for (size_t c = 0; c < n_fe; ++c) {
          ad_batch.value(c, j) = batch.values(c).data[i];
          for (int d1 = 0; d1 < dim; ++d1) {
            ad_batch.derivative(c, d1, j) = batch.derivatives(c, d1).data[i];
            if (batch.has_hessians())
              for (int d2 = 0; d2 < dim; ++d2)
                ad_batch.hessian(c, d1, d2, j) = batch.hessians(c, d1, d2).data[i];
          }
        }
        for (int d = 0; d < dim; ++d)
          ad_batch.coordinate(d, j) = batch.coordinates(d).data[i];
        ad_batch.width(j) = batch.cell_width(i);
      };
      const auto prepare = [&](const size_t n_blocks) {
        ad_batch.reinit(n_blocks * n, batch.has_hessians());
        ad_batch.set_shared(ad_extractors, batch.variables());
        ad_out.reinit(n_blocks * n);
      };

      const auto seeds = batch_seeds<dim, n_fe>(reads_derivatives, batch.has_hessians());
      const size_t per_group = std::max<size_t>(1, max_stacked / n);
      for (size_t g0 = 0; g0 < seeds.size(); g0 += per_group) {
        const size_t G = std::min(per_group, seeds.size() - g0);
        prepare(G);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
          for (size_t i = r.begin(); i != r.end(); ++i)
            for (size_t b = 0; b < G; ++b) {
              copy_block_point(b, i);
              const auto &s = seeds[g0 + b];
              const size_t j = b * n + i;
              if (s.kind == BatchSeed::value)
                seed_direction(ad_batch.value(s.c, j));
              else if (s.kind == BatchSeed::derivative)
                seed_direction(ad_batch.derivative(s.c, s.d1, j));
              else {
                seed_direction(ad_batch.hessian(s.c, s.d1, s.d2, j));
                if (s.d1 != s.d2) seed_direction(ad_batch.hessian(s.c, s.d2, s.d1, j));
              }
            }
        });

        evaluate(ad_out, ad_batch, G);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
          for (size_t i = r.begin(); i != r.end(); ++i)
            for (size_t b = 0; b < G; ++b) {
              const auto &s = seeds[g0 + b];
              const size_t j = b * n + i;
              auto &Ji = J[i];
              for (uint ci = 0; ci < n_fe; ++ci) {
                const double gs = with_source ? along_direction(ad_out.source(ci)[j]) : 0.;
                for (int d = 0; d < dim; ++d) {
                  const double gf = along_direction(ad_out.flux(ci, d)[j]);
                  if (s.kind == BatchSeed::value)
                    Ji.j_flux(ci, s.c)[d] = gf;
                  else if (s.kind == BatchSeed::derivative)
                    Ji.j_grad_flux(ci, s.c)[d][s.d1] = gf;
                  else if (s.d1 == s.d2)
                    Ji.j_hess_flux(ci, s.c)[d][s.d1][s.d1] = gf;
                  else {
                    // The joint seed yields dF/dH_12 + dF/dH_21; only the symmetric part is contracted.
                    Ji.j_hess_flux(ci, s.c)[d][s.d1][s.d2] = gf / 2;
                    Ji.j_hess_flux(ci, s.c)[d][s.d2][s.d1] = gf / 2;
                  }
                }
                if (s.kind == BatchSeed::value)
                  Ji.j_source(ci, s.c) = gs;
                else if (s.kind == BatchSeed::derivative)
                  Ji.j_grad_source(ci, s.c)[s.d1] = gs;
                else if (s.d1 == s.d2)
                  Ji.j_hess_source(ci, s.c)[s.d1][s.d1] = gs;
                else {
                  Ji.j_hess_source(ci, s.c)[s.d1][s.d2] = gs / 2;
                  Ji.j_hess_source(ci, s.c)[s.d2][s.d1] = gs / 2;
                }
              }
            }
        });
      }

      for (size_t e = 0; e < n_extr; ++e) {
        prepare(1);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
          for (size_t i = r.begin(); i != r.end(); ++i)
            copy_block_point(0, i);
        });
        seed_direction(ad_extractors[e]);
        evaluate(ad_out, ad_batch, 1);
        clear_direction(ad_extractors[e]);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
          for (size_t i = r.begin(); i != r.end(); ++i)
            for (uint ci = 0; ci < n_fe; ++ci) {
              for (int d = 0; d < dim; ++d)
                J[i].j_extr_flux(ci, e)[d] = along_direction(ad_out.flux(ci, d)[i]);
              if (with_source) J[i].j_extr_source(ci, e) = along_direction(ad_out.source(ci)[i]);
            }
        });
      }
    }

    template <typename Model>
    constexpr bool has_ad_flux_source_jacobians =
        std::is_base_of_v<def::ADjacobian_flux_source<Model, autodiff::real>, Model>;
    template <typename Model>
    constexpr bool has_ad_boundary_jacobians =
        std::is_base_of_v<def::ADjacobian_boundary_numflux<Model, autodiff::real>, Model>;
  } // namespace internal

  /**
   * @brief Jacobian of flux and source at all points of a batch: for a model with AD jacobians
   * (def::AD) the seed-stacked AD of flux_source_batch, see internal::seed_stacked_jacobian; otherwise
   * the model's per-point jacobian_flux_source* callbacks.
   */
  template <typename Model, typename Batch, size_t n_extr>
  void evaluate_flux_source_jacobian(
      const Model &model, std::vector<PointJacobian<Batch::dim, Batch::n_fe_functions, n_extr>> &J, const Batch &batch,
      const size_t max_stacked,
      SeedStackWorkspace<Batch::dim, Batch::n_fe_functions, n_extr, typename Batch::variables_type> &workspace)
  {
    if constexpr (internal::has_ad_flux_source_jacobians<Model>)
      internal::seed_stacked_jacobian(
          [&](auto &out, const auto &ad_batch, size_t) { model.flux_source_batch(out, ad_batch); }, J, batch,
          batch_reads_derivatives<Model>(), max_stacked, true, workspace);
    else {
      internal::parallel_assign(J, batch.size(), {});
      internal::for_each_point(batch, [&](const size_t i, const auto &x, const auto &sol) {
        auto &Ji = J[i];
        model.template jacobian_flux_source<0, 0>(Ji.j_flux, Ji.j_source, x, sol);
        model.template jacobian_flux_source_grad<1>(Ji.j_grad_flux, Ji.j_grad_source, x, sol);
        model.template jacobian_flux_source_hess<2>(Ji.j_hess_flux, Ji.j_hess_source, x, sol);
        if constexpr (n_extr > 0) model.template jacobian_flux_source_extr<3>(Ji.j_extr_flux, Ji.j_extr_source, x, sol);
      });
    }
  }

  /**
   * @brief Jacobian of the boundary numflux at all points of a boundary batch, into the flux blocks of
   * J. Same dispatch as evaluate_flux_source_jacobian, with the per-point jacobian_boundary_numflux*.
   */
  template <typename Model, typename Batch, size_t n_extr>
  void evaluate_boundary_numflux_jacobian(
      const Model &model, std::vector<PointJacobian<Batch::dim, Batch::n_fe_functions, n_extr>> &J,
      const std::vector<dealii::Tensor<1, Batch::dim>> &normals, const Batch &batch, const size_t max_stacked,
      SeedStackWorkspace<Batch::dim, Batch::n_fe_functions, n_extr, typename Batch::variables_type> &workspace)
  {
    if constexpr (internal::has_ad_boundary_jacobians<Model>) {
      // The AD batch stacks n_blocks copies of the points, so it needs the normals repeated as often.
      auto &stacked_normals = workspace.normals;
      internal::seed_stacked_jacobian(
          [&](auto &out, const auto &ad_batch, const size_t n_blocks) {
            stacked_normals.resize(n_blocks * normals.size());
            for (size_t b = 0; b < n_blocks; ++b)
              std::copy(normals.begin(), normals.end(), stacked_normals.begin() + b * normals.size());
            evaluate_boundary_numflux(model, out, stacked_normals, ad_batch);
          },
          J, batch, batch_reads_derivatives<Model>(), max_stacked, false, workspace);
    } else {
      internal::parallel_assign(J, batch.size(), {});
      internal::for_each_point(batch, [&](const size_t i, const auto &x, const auto &sol) {
        auto &Ji = J[i];
        model.template jacobian_boundary_numflux<0, 0>(Ji.j_flux, normals[i], x, sol);
        model.template jacobian_boundary_numflux_grad<1>(Ji.j_grad_flux, normals[i], x, sol);
        model.template jacobian_boundary_numflux_hess<2>(Ji.j_hess_flux, normals[i], x, sol);
        if constexpr (n_extr > 0) model.template jacobian_boundary_numflux_extr<3>(Ji.j_extr_flux, normals[i], x, sol);
      });
    }
  }
} // namespace DiFfRG
