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
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief Per-point solution tuple with the names of CG::fe_tie, so the per-point model callbacks
   * (flux, source, numflux, boundary_numflux and their jacobians) accept it.
   */
  template <typename... T> auto batch_tie(T &&...t)
  {
    return named_tuple<std::tuple<T &...>, StringSet<"fe_functions", "fe_derivatives", "fe_hessians", "extractors",
                                                     "variables", "cell_width">>(std::tie(t...));
  }

  /**
   * @brief The terms a batch evaluation (Model::evaluate_batch) can produce. The assembler requests only
   * those it uses, e.g. `Term::flux | Term::source` at cell points and `Term::flux` at faces.
   */
  enum class Term : unsigned { flux = 1, source = 2, diffusion_flux = 4 };
  constexpr Term operator|(const Term a, const Term b) { return Term(unsigned(a) | unsigned(b)); }
  /// Whether the set of terms @p set contains @p t.
  constexpr bool contains(const Term set, const Term t) { return (unsigned(set) & unsigned(t)) != 0; }

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
   * @brief The FE solution at a set of points: values and (optionally) derivatives and hessians of every
   * FE function, plus the position and cell width of each point. Extractors and variables are shared by
   * all points.
   *
   * Storage is component-major with the points fastest, so values(c), derivatives(c, d) and
   * hessians(c, d1, d2) are contiguous columns that can be passed directly as per-point arguments to
   * an integrator's map_points().
   *
   * Point i means the same point in every column, and in every column of the output BatchOutput.
   * The order of the points carries no meaning for the model.
   */
  template <int dim_, typename NT, size_t n_fe, typename Extractors, typename Variables> class PointBatch
  {
  public:
    static constexpr int dim = dim_;
    static constexpr size_t n_fe_functions = n_fe;
    using number_type = NT;
    using extractors_type = Extractors;
    using variables_type = Variables;

    void reinit(const size_t n_points, const bool with_derivatives, const bool with_hessians)
    {
      n = n_points;
      m_derivatives = with_derivatives;
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
    bool has_derivatives() const { return m_derivatives; }
    bool has_hessians() const { return m_hessians; }

    PointArray<NT> values(const size_t c) const { return {column(value_column(c)), n}; }
    PointArray<NT> derivatives(const size_t c, const size_t d) const
    {
      if (!m_derivatives)
        throw std::logic_error("PointBatch: derivatives were not gathered (DG, or batch_reads_derivatives = false).");
      return {column(derivative_column(c, d)), n};
    }
    PointArray<NT> hessians(const size_t c, const size_t d1, const size_t d2) const
    {
      if (!m_hessians)
        throw std::logic_error("PointBatch: hessians were not gathered (DG, or batch_reads_hessians = false).");
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

    /// Point i as a PointState; inputs that were not gathered are zero.
    void load(const size_t i, PointState<dim, NT, n_fe> &s) const
    {
      for (size_t c = 0; c < n_fe; ++c) {
        s.values[c] = column(value_column(c))[i];
        for (int d1 = 0; d1 < dim; ++d1) {
          s.derivatives[c][d1] = m_derivatives ? column(derivative_column(c, d1))[i] : NT(0);
          for (int d2 = 0; d2 < dim; ++d2)
            s.hessians[c][d1][d2] = m_hessians ? column(hessian_column(c, d1, d2))[i] : NT(0);
        }
      }
      s.cell_width = widths[i];
    }

    /// Point j of this batch = point i of @p src, which has the same inputs but may differ in number type.
    template <typename Src> void copy_point(const size_t j, const Src &src, const size_t i)
    {
      for (size_t c = 0; c < n_fe; ++c) {
        value(c, j) = src.values(c).data[i];
        for (int d1 = 0; d1 < dim; ++d1) {
          if (m_derivatives) derivative(c, d1, j) = src.derivatives(c, d1).data[i];
          if (m_hessians)
            for (int d2 = 0; d2 < dim; ++d2)
              hessian(c, d1, d2, j) = src.hessians(c, d1, d2).data[i];
        }
      }
      for (int d = 0; d < dim; ++d)
        coordinate(d, j) = src.coordinates(d).data[i];
      width(j) = src.cell_width(i);
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
    size_t n_columns() const { return n_fe * (1 + (m_derivatives ? dim : 0) + (m_hessians ? dim * dim : 0)); }
    static constexpr size_t value_column(const size_t c) { return c; }
    static constexpr size_t derivative_column(const size_t c, const size_t d) { return n_fe + c * dim + d; }
    size_t hessian_column(const size_t c, const size_t d1, const size_t d2) const
    {
      return n_fe * (1 + (m_derivatives ? dim : 0)) + (c * dim + d1) * dim + d2;
    }
    const NT *column(const size_t col) const { return state.data() + col * n; }
    NT *column(const size_t col) { return state.data() + col * n; }

    size_t n = 0;
    bool m_derivatives = true, m_hessians = true;
    std::vector<NT> state;
    std::vector<double> positions;
    std::vector<double> widths;
    const Extractors *m_extractors = nullptr;
    const Variables *m_variables = nullptr;
  };

  /**
   * @brief The requested terms (flux, source, diffusion flux) of every FE function at the points of a
   * PointBatch, as contiguous columns. Only requested terms are stored; reinit() zeroes them, so a model
   * only writes what it computes.
   */
  template <int dim, typename NT, size_t n_fe> class BatchOutput
  {
  public:
    void reinit(const size_t n_points, const Term terms)
    {
      n = n_points;
      m_terms = terms;
      size_t columns = 0;
      const auto place = [&](const Term t, const size_t width) {
        if (!contains(terms, t)) return no_column;
        const size_t offset = columns;
        columns += width;
        return offset;
      };
      flux_offset = place(Term::flux, n_fe * dim);
      source_offset = place(Term::source, n_fe);
      diffusion_offset = place(Term::diffusion_flux, n_fe * dim);
      internal::parallel_assign(data, n * columns, NT(0));
    }
    size_t size() const { return n; }
    /// Whether the assembler asked for @p t; a model may skip computing anything else.
    bool requested(const Term t) const { return contains(m_terms, t); }
    Term terms() const { return m_terms; }

    NT *flux(const size_t c, const size_t d) { return column(flux_offset, c * dim + d, "flux"); }
    const NT *flux(const size_t c, const size_t d) const { return column(flux_offset, c * dim + d, "flux"); }
    NT *source(const size_t c) { return column(source_offset, c, "source"); }
    const NT *source(const size_t c) const { return column(source_offset, c, "source"); }
    NT *diffusion_flux(const size_t c, const size_t d) { return column(diffusion_offset, c * dim + d, "diffusion"); }
    const NT *diffusion_flux(const size_t c, const size_t d) const
    {
      return column(diffusion_offset, c * dim + d, "diffusion");
    }

    void store_flux(const size_t i, const std::array<dealii::Tensor<1, dim, NT>, n_fe> &F)
    {
      for (size_t c = 0; c < n_fe; ++c)
        for (int d = 0; d < dim; ++d)
          flux(c, d)[i] = F[c][d];
    }
    void store_source(const size_t i, const std::array<NT, n_fe> &S)
    {
      for (size_t c = 0; c < n_fe; ++c)
        source(c)[i] = S[c];
    }
    void store_diffusion_flux(const size_t i, const std::array<dealii::Tensor<1, dim, NT>, n_fe> &D)
    {
      for (size_t c = 0; c < n_fe; ++c)
        for (int d = 0; d < dim; ++d)
          diffusion_flux(c, d)[i] = D[c][d];
    }

  private:
    static constexpr size_t no_column = size_t(-1);
    NT *column(const size_t offset, const size_t col, const char *term)
    {
      return const_cast<NT *>(std::as_const(*this).column(offset, col, term));
    }
    const NT *column(const size_t offset, const size_t col, const char *term) const
    {
      if (offset == no_column)
        throw std::logic_error(std::string("BatchOutput: the ") + term + " was not requested by the assembler.");
      return data.data() + (offset + col) * n;
    }

    size_t n = 0;
    Term m_terms = Term::flux;
    size_t flux_offset = no_column, source_offset = no_column, diffusion_offset = no_column;
    std::vector<NT> data;
  };

  /**
   * @brief Derivatives of flux and source at one point with respect to the FE values, derivatives,
   * hessians and the extractors. Same layout as the per-point jacobian_flux_source* callbacks. At
   * face points the flux blocks hold the numerical flux and the source blocks stay zero; an interior
   * face point has one of these per trace, and the extractor blocks of trace 0 hold the total.
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
  /// The jacobian at an interior face point: one block set per trace.
  template <int dim, size_t n_fe, size_t n_extr> using FaceJacobian = std::array<PointJacobian<dim, n_fe, n_extr>, 2>;

  /**
   * @brief Buffers of the seed-stacked AD jacobian (internal::seed_stacked_jacobian), kept by the caller
   * across calls: they hold n_seeds copies of the batch (of each of its n_sides traces), and allocating
   * them anew every time costs as much as the evaluation itself.
   */
  template <int dim, size_t n_fe, size_t n_extr, typename Variables, size_t n_sides = 1> struct SeedStackWorkspace {
    using ADBatch = PointBatch<dim, autodiff::real, n_fe, std::array<autodiff::real, n_extr>, Variables>;
    std::array<ADBatch, n_sides> batches;
    BatchOutput<dim, autodiff::real, n_fe> out;
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

    /// f(i, x, sol_s, sol_n) at every point i of a pair of trace batches, in a flat parallel loop.
    template <typename Batch, typename F>
    void for_each_point_pair(const Batch &batch_s, const Batch &batch_n, const F &f)
    {
      tbb::parallel_for(tbb::blocked_range<size_t>(0, batch_s.size()), [&](const tbb::blocked_range<size_t> &r) {
        PointState<Batch::dim, typename Batch::number_type, Batch::n_fe_functions> s, n;
        for (size_t i = r.begin(); i != r.end(); ++i) {
          batch_s.load(i, s);
          batch_n.load(i, n);
          f(i, batch_s.x(i),
            batch_tie(s.values, s.derivatives, s.hessians, batch_s.extractors(), batch_s.variables(), s.cell_width),
            batch_tie(n.values, n.derivatives, n.hessians, batch_n.extractors(), batch_n.variables(), n.cell_width));
        }
      });
    }

    /**
     * @brief The requested terms at all points of a batch from the model's per-point flux, source and
     * diffusion_flux. This is def::AbstractModel::evaluate_batch's default.
     *
     * The per-point callbacks run on several threads at once; a model whose per-point flux is not thread
     * safe (e.g. it calls a GPU integrator's get()) must override evaluate_batch.
     */
    template <typename Model, typename Out, typename Batch>
    void evaluate_per_point(const Model &model, Out &out, const Batch &batch)
    {
      using NT = typename Batch::number_type;
      constexpr int dim = Batch::dim;
      constexpr size_t n_fe = Batch::n_fe_functions;
      for_each_point(batch, [&](const size_t i, const auto &x, const auto &sol) {
        if (out.requested(Term::flux)) {
          std::array<dealii::Tensor<1, dim, NT>, n_fe> F{};
          model.flux(F, x, sol);
          out.store_flux(i, F);
        }
        if (out.requested(Term::source)) {
          std::array<NT, n_fe> S{};
          model.source(S, x, sol);
          out.store_source(i, S);
        }
        if (out.requested(Term::diffusion_flux)) {
          std::array<dealii::Tensor<1, dim, NT>, n_fe> D{};
          model.diffusion_flux(D, x, sol);
          out.store_diffusion_flux(i, D);
        }
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
        out.store_flux(i, F);
      });
    }
  }

  /**
   * @brief Numerical flux at all points of an interior-face batch, into the flux columns of @p out: the
   * model's `numflux_batch(out, normals, batch_s, batch_n)` if it has one (def::LLFFlux does), otherwise its
   * per-point numflux. Point i of batch_s and batch_n are the two traces at the same face point, and
   * normals[i] points out of the cell of batch_s.
   */
  template <typename Model, typename Out, typename Batch>
  void evaluate_numflux(const Model &model, Out &out, const std::vector<dealii::Tensor<1, Batch::dim>> &normals,
                        const Batch &batch_s, const Batch &batch_n)
  {
    if constexpr (requires { model.numflux_batch(out, normals, batch_s, batch_n); })
      model.numflux_batch(out, normals, batch_s, batch_n);
    else {
      using NT = typename Batch::number_type;
      internal::for_each_point_pair(batch_s, batch_n,
                                    [&](const size_t i, const auto &x, const auto &sol_s, const auto &sol_n) {
                                      std::array<dealii::Tensor<1, Batch::dim, NT>, Batch::n_fe_functions> F{};
                                      model.numflux(F, normals[i], x, sol_s, sol_n);
                                      out.store_flux(i, F);
                                    });
    }
  }

  namespace internal
  {
    /// One forward-mode AD seed direction among the FE inputs of one trace of a point.
    struct BatchSeed {
      enum Kind { value, derivative, hessian } kind;
      uint c, d1, d2;
      uint side;
    };

    // Forward-mode AD on autodiff::real: component 1 carries the derivative along the seeded direction.
    inline void seed_direction(autodiff::real &x) { x[1] = 1.; }
    inline void clear_direction(autodiff::real &x) { x[1] = 0.; }
    inline double along_direction(const autodiff::real &x) { return x[1]; }

    template <int dim, size_t n_fe>
    std::vector<BatchSeed> batch_seeds(const uint n_sides, const bool with_derivatives, const bool with_hessians)
    {
      std::vector<BatchSeed> seeds;
      for (uint side = 0; side < n_sides; ++side) {
        for (uint c = 0; c < n_fe; ++c)
          seeds.push_back({BatchSeed::value, c, 0, 0, side});
        if (with_derivatives)
          for (uint c = 0; c < n_fe; ++c)
            for (uint d = 0; d < dim; ++d)
              seeds.push_back({BatchSeed::derivative, c, d, 0, side});
        // A hessian is symmetric, so (d1, d2) and (d2, d1) are seeded together.
        if (with_hessians)
          for (uint c = 0; c < n_fe; ++c)
            for (uint d1 = 0; d1 < dim; ++d1)
              for (uint d2 = d1; d2 < dim; ++d2)
                seeds.push_back({BatchSeed::hessian, c, d1, d2, side});
      }
      return seeds;
    }

    /**
     * @brief Forward-mode AD jacobian of a batch evaluation, with all seed directions stacked along
     * the point axis: block b of the AD batches is a copy of the input batches with seed b set at every
     * point, so a single evaluation yields the jacobian column of seed b at all points. Seeds are
     * stacked up to max_stacked points per evaluation. Extractor seeds are one evaluation each, as the
     * extractors are shared by all points (and traces) of a batch; their total derivative goes to side 0.
     * Derivatives and hessians are seeded if the batches hold them.
     *
     * @param evaluate callable(BatchOutput<AD> &out, const std::array<ADBatch, n_sides> &batches, size_t n_blocks)
     * @param jacobian_at callable(i, side) -> PointJacobian & of point i with respect to trace `side`
     * @param batches the n_sides traces of the points (one for cells and boundary faces, two for interior faces)
     * @param terms the terms evaluate produces; only those are read
     */
    template <size_t n_sides, typename Evaluate, typename JacobianAt, typename Batch, size_t n_extr>
    void seed_stacked_jacobian(const Evaluate &evaluate, const JacobianAt &jacobian_at,
                               const std::array<const Batch *, n_sides> &batches, const size_t max_stacked,
                               const Term terms,
                               SeedStackWorkspace<Batch::dim, Batch::n_fe_functions, n_extr,
                                                  typename Batch::variables_type, n_sides> &workspace)
    {
      constexpr int dim = Batch::dim;
      constexpr size_t n_fe = Batch::n_fe_functions;
      const size_t n = batches[0]->size();
      if (n == 0) return;
      const bool with_flux = contains(terms, Term::flux), with_source = contains(terms, Term::source);

      auto &ad_extractors = workspace.extractors;
      for (size_t e = 0; e < n_extr; ++e)
        ad_extractors[e] = batches[0]->extractors()[e];

      auto &ad_batches = workspace.batches;
      auto &ad_out = workspace.out;
      const auto prepare = [&](const size_t n_blocks) {
        for (size_t s = 0; s < n_sides; ++s) {
          ad_batches[s].reinit(n_blocks * n, batches[s]->has_derivatives(), batches[s]->has_hessians());
          ad_batches[s].set_shared(ad_extractors, batches[s]->variables());
        }
        ad_out.reinit(n_blocks * n, terms);
      };
      // Slot b * n + i of every AD trace := point i of that trace, unseeded.
      const auto copy_block_point = [&](const size_t b, const size_t i) {
        for (size_t s = 0; s < n_sides; ++s)
          ad_batches[s].copy_point(b * n + i, *batches[s], i);
      };

      const auto seeds = batch_seeds<dim, n_fe>(n_sides, batches[0]->has_derivatives(), batches[0]->has_hessians());
      const size_t per_group = std::max<size_t>(1, max_stacked / n);
      for (size_t g0 = 0; g0 < seeds.size(); g0 += per_group) {
        const size_t G = std::min(per_group, seeds.size() - g0);
        prepare(G);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
          for (size_t i = r.begin(); i != r.end(); ++i)
            for (size_t b = 0; b < G; ++b) {
              copy_block_point(b, i);
              const auto &s = seeds[g0 + b];
              auto &ad = ad_batches[s.side];
              const size_t j = b * n + i;
              if (s.kind == BatchSeed::value)
                seed_direction(ad.value(s.c, j));
              else if (s.kind == BatchSeed::derivative)
                seed_direction(ad.derivative(s.c, s.d1, j));
              else {
                seed_direction(ad.hessian(s.c, s.d1, s.d2, j));
                if (s.d1 != s.d2) seed_direction(ad.hessian(s.c, s.d2, s.d1, j));
              }
            }
        });

        evaluate(ad_out, std::as_const(ad_batches), G);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
          for (size_t i = r.begin(); i != r.end(); ++i)
            for (size_t b = 0; b < G; ++b) {
              const auto &s = seeds[g0 + b];
              const size_t j = b * n + i;
              auto &Ji = jacobian_at(i, s.side);
              for (uint ci = 0; ci < n_fe; ++ci) {
                if (with_flux)
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
                if (with_source) {
                  const double gs = along_direction(ad_out.source(ci)[j]);
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
        evaluate(ad_out, std::as_const(ad_batches), 1);
        clear_direction(ad_extractors[e]);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &r) {
          for (size_t i = r.begin(); i != r.end(); ++i) {
            auto &Ji = jacobian_at(i, 0);
            for (uint ci = 0; ci < n_fe; ++ci) {
              if (with_flux)
                for (int d = 0; d < dim; ++d)
                  Ji.j_extr_flux(ci, e)[d] = along_direction(ad_out.flux(ci, d)[i]);
              if (with_source) Ji.j_extr_source(ci, e) = along_direction(ad_out.source(ci)[i]);
            }
          }
        });
      }
    }

    /// n_blocks copies of @p normals, one per block of a seed-stacked AD batch.
    template <typename Normal>
    const std::vector<Normal> &stack_normals(std::vector<Normal> &stacked, const std::vector<Normal> &normals,
                                             const size_t n_blocks)
    {
      stacked.resize(n_blocks * normals.size());
      for (size_t b = 0; b < n_blocks; ++b)
        std::copy(normals.begin(), normals.end(), stacked.begin() + b * normals.size());
      return stacked;
    }

    template <typename Model>
    constexpr bool has_ad_flux_source_jacobians =
        std::is_base_of_v<def::ADjacobian_flux_source<Model, autodiff::real>, Model>;
    template <typename Model>
    constexpr bool has_ad_boundary_jacobians =
        std::is_base_of_v<def::ADjacobian_boundary_numflux<Model, autodiff::real>, Model>;
    template <typename Model>
    constexpr bool has_ad_numflux_jacobians = std::is_base_of_v<def::ADjacobian_numflux<Model, autodiff::real>, Model>;
  } // namespace internal

  /**
   * @brief Jacobian of flux and source at all points of a batch: for a model with AD jacobians
   * (def::AD) the seed-stacked AD of evaluate_batch, see internal::seed_stacked_jacobian; otherwise
   * the model's per-point jacobian_flux_source* callbacks.
   */
  template <typename Model, typename Batch, size_t n_extr>
  void evaluate_flux_source_jacobian(
      const Model &model, std::vector<PointJacobian<Batch::dim, Batch::n_fe_functions, n_extr>> &J, const Batch &batch,
      const size_t max_stacked,
      SeedStackWorkspace<Batch::dim, Batch::n_fe_functions, n_extr, typename Batch::variables_type> &workspace)
  {
    internal::parallel_assign(J, batch.size(), {});
    if constexpr (internal::has_ad_flux_source_jacobians<Model>)
      internal::seed_stacked_jacobian<1>([&](auto &out, const auto &ad, size_t) { model.evaluate_batch(out, ad[0]); },
                                         [&](const size_t i, uint) -> auto & { return J[i]; }, std::array{&batch},
                                         max_stacked, Term::flux | Term::source, workspace);
    else
      internal::for_each_point(batch, [&](const size_t i, const auto &x, const auto &sol) {
        auto &Ji = J[i];
        model.template jacobian_flux_source<0, 0>(Ji.j_flux, Ji.j_source, x, sol);
        model.template jacobian_flux_source_grad<1>(Ji.j_grad_flux, Ji.j_grad_source, x, sol);
        model.template jacobian_flux_source_hess<2>(Ji.j_hess_flux, Ji.j_hess_source, x, sol);
        if constexpr (n_extr > 0) model.template jacobian_flux_source_extr<3>(Ji.j_extr_flux, Ji.j_extr_source, x, sol);
      });
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
    internal::parallel_assign(J, batch.size(), {});
    if constexpr (internal::has_ad_boundary_jacobians<Model>)
      internal::seed_stacked_jacobian<1>(
          [&](auto &out, const auto &ad, const size_t n_blocks) {
            evaluate_boundary_numflux(model, out, internal::stack_normals(workspace.normals, normals, n_blocks), ad[0]);
          },
          [&](const size_t i, uint) -> auto & { return J[i]; }, std::array{&batch}, max_stacked, Term::flux, workspace);
    else
      internal::for_each_point(batch, [&](const size_t i, const auto &x, const auto &sol) {
        auto &Ji = J[i];
        model.template jacobian_boundary_numflux<0, 0>(Ji.j_flux, normals[i], x, sol);
        model.template jacobian_boundary_numflux_grad<1>(Ji.j_grad_flux, normals[i], x, sol);
        model.template jacobian_boundary_numflux_hess<2>(Ji.j_hess_flux, normals[i], x, sol);
        if constexpr (n_extr > 0) model.template jacobian_boundary_numflux_extr<3>(Ji.j_extr_flux, normals[i], x, sol);
      });
  }

  /**
   * @brief Jacobian of the numerical flux at all points of an interior-face batch with respect to both
   * traces, into the flux blocks of J[i][0] (trace s) and J[i][1] (trace n); the extractor blocks of
   * J[i][0] hold the total. Same dispatch as evaluate_flux_source_jacobian, with the per-point
   * jacobian_numflux*.
   */
  template <typename Model, typename Batch, size_t n_extr>
  void evaluate_numflux_jacobian(
      const Model &model, std::vector<FaceJacobian<Batch::dim, Batch::n_fe_functions, n_extr>> &J,
      const std::vector<dealii::Tensor<1, Batch::dim>> &normals, const Batch &batch_s, const Batch &batch_n,
      const size_t max_stacked,
      SeedStackWorkspace<Batch::dim, Batch::n_fe_functions, n_extr, typename Batch::variables_type, 2> &workspace)
  {
    constexpr int dim = Batch::dim;
    constexpr size_t n_fe = Batch::n_fe_functions;
    internal::parallel_assign(J, batch_s.size(), {});
    if constexpr (internal::has_ad_numflux_jacobians<Model>)
      internal::seed_stacked_jacobian<2>(
          [&](auto &out, const auto &ad, const size_t n_blocks) {
            evaluate_numflux(model, out, internal::stack_normals(workspace.normals, normals, n_blocks), ad[0], ad[1]);
          },
          [&](const size_t i, const uint side) -> auto & { return J[i][side]; }, std::array{&batch_s, &batch_n},
          max_stacked, Term::flux, workspace);
    else
      internal::for_each_point_pair(
          batch_s, batch_n, [&](const size_t i, const auto &x, const auto &sol_s, const auto &sol_n) {
            using T1 = dealii::Tensor<1, dim>;
            std::array<SimpleMatrix<T1, n_fe>, 2> j_value;
            std::array<SimpleMatrix<dealii::Tensor<1, dim, T1>, n_fe>, 2> j_grad;
            std::array<SimpleMatrix<dealii::Tensor<1, dim, dealii::Tensor<2, dim>>, n_fe>, 2> j_hess;
            model.template jacobian_numflux<0, 0>(j_value, normals[i], x, sol_s, sol_n);
            model.template jacobian_numflux_grad<1>(j_grad, normals[i], x, sol_s, sol_n);
            model.template jacobian_numflux_hess<2>(j_hess, normals[i], x, sol_s, sol_n);
            for (uint side = 0; side < 2; ++side) {
              J[i][side].j_flux = j_value[side];
              J[i][side].j_grad_flux = j_grad[side];
              J[i][side].j_hess_flux = j_hess[side];
            }
            if constexpr (n_extr > 0) {
              std::array<SimpleMatrix<T1, n_fe, n_extr>, 2> j_extr;
              model.template jacobian_numflux_extr<3>(j_extr, normals[i], x, sol_s, sol_n);
              for (uint c = 0; c < n_fe; ++c)
                for (uint e = 0; e < n_extr; ++e)
                  J[i][0].j_extr_flux(c, e) = j_extr[0](c, e) + j_extr[1](c, e);
            }
          });
  }
} // namespace DiFfRG
