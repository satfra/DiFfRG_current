#pragma once

// DiFfRG
#include <DiFfRG/discretization/FV/assembler/flux_ties.hh>
#include <DiFfRG/discretization/FV/reconstructor/types.hh>
#include <DiFfRG/model/batch.hh>

// external libraries
#include <autodiff/forward/real/real.hpp>
#include <tbb/tbb.h>

// standard library
#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

namespace DiFfRG
{
  namespace FV
  {
    namespace KurganovTadmor
    {
      namespace internal
      {
        /// Which per-point tuple a TraceBatch hands the model.
        enum class TraceKind {
          /// flux_tie: "fe_functions", "fe_derivatives", "extractors", "variables", "cell_width"
          flux,
          /// diffusion_flux_tie: flux_tie plus "fe_third_derivatives" after the derivatives
          diffusion,
          /// the same names as flux_tie; "fe_derivatives" is the reconstructor's limited slope at the cell point.
          /// There is no "fe_hessians" slot, so a model that reads it without opting in via source_uses_hessians
          /// fails to compile rather than silently receiving zeros.
          source,
          /// source plus "fe_hessians" (the unlimited diagonal curvature) after the derivatives
          source_hessians
        };

        /**
         * @brief The states the KT assembler evaluates the model at, as a batch: the face traces (u^-, u^+ with
         * their reconstructed gradients, and for the diffusion flux the corrected gradients and third derivatives)
         * or the cell quadrature points (limited gradient, and for source_uses_hessians the diagonal curvature).
         *
         * A PointBatch with optional third-derivative columns, `third_derivatives(c, d0, d1, d2)`. Its per-point
         * tuple (tie()) has the names and positions of flux_tie / diffusion_flux_tie (see TraceKind).
         */
        template <int dim_, typename NT, size_t n, typename Extractors, typename Variables, TraceKind kind>
        class TraceBatch : public PointBatch<dim_, NT, n, Extractors, Variables>
        {
          using Base = PointBatch<dim_, NT, n, Extractors, Variables>;

        public:
          static constexpr int dim = dim_;
          static constexpr bool with_third = kind == TraceKind::diffusion;
          /// Third derivatives are only reconstructed in 1D: elsewhere the tuple carries zeros and nothing is stored.
          static constexpr bool stores_third = with_third && dim == 1;
          static constexpr bool with_hessians = kind == TraceKind::source_hessians;
          static constexpr uint extractor_index = with_third || with_hessians ? 3 : 2;
          /// The term this tuple is made for (see DiFfRG::batch_terms).
          static constexpr Term terms = kind == TraceKind::flux        ? Term::flux
                                        : kind == TraceKind::diffusion ? Term::diffusion_flux
                                                                       : Term::source;
          /// Extractors stay double: KT freezes them within a Newton step.
          template <typename NT2> using rebind = TraceBatch<dim, NT2, n, Extractors, Variables, kind>;

          struct State : PointState<dim, NT, n> {
            def::ThirdDerivativeType<dim, NT, n> third_derivatives{};
          };

          void reinit(const size_t n_points) { reinit(n_points, true, with_hessians); }
          void reinit(const size_t n_points, const bool with_derivatives, const bool with_hessians_)
          {
            Base::reinit(n_points, with_derivatives, with_hessians_);
            if constexpr (stores_third)
              third.resize(n_points * n_third_columns);
            else if constexpr (with_third)
              third.assign(n_points, NT(0)); // one zero column, the value of every third derivative
          }

          PointArray<NT> third_derivatives(const size_t c, const size_t d0, const size_t d1, const size_t d2) const
          {
            static_assert(with_third, "Only the diffusion traces carry third derivatives.");
            if constexpr (!stores_third) return {third.data(), this->size()};
            return {third.data() + third_column(c, d0, d1, d2) * this->size(), this->size()};
          }
          NT &third_derivative(const size_t c, const size_t d0, const size_t d1, const size_t d2, const size_t i)
          {
            static_assert(stores_third, "Third derivatives are only stored in 1D.");
            return third[third_column(c, d0, d1, d2) * this->size() + i];
          }

          void load(const size_t i, State &s) const
          {
            Base::load(i, s);
            if constexpr (stores_third)
              for (size_t c = 0; c < n; ++c)
                for (int d0 = 0; d0 < dim; ++d0)
                  for (int d1 = 0; d1 < dim; ++d1)
                    for (int d2 = 0; d2 < dim; ++d2)
                      s.third_derivatives[c][d0][d1][d2] = third_derivatives(c, d0, d1, d2).data[i];
          }

          auto tie(State &s) const
          {
            if constexpr (with_third)
              return diffusion_flux_tie(s.values, s.derivatives, s.third_derivatives, this->extractors(),
                                        this->variables(), s.cell_width);
            else if constexpr (with_hessians)
              return named_tuple<
                  std::tuple<decltype(s.values) &, decltype(s.derivatives) &, decltype(s.hessians) &,
                             const Extractors &, const Variables &, double &>,
                  StringSet<"fe_functions", "fe_derivatives", "fe_hessians", "extractors", "variables", "cell_width">>(
                  std::tie(s.values, s.derivatives, s.hessians, this->extractors(), this->variables(), s.cell_width));
            else
              return flux_tie(s.values, s.derivatives, this->extractors(), this->variables(), s.cell_width);
          }

          template <typename Src> void copy_point(const size_t j, const Src &src, const size_t i)
          {
            Base::copy_point(j, src, i);
            if constexpr (stores_third)
              for (size_t c = 0; c < n; ++c)
                for (int d0 = 0; d0 < dim; ++d0)
                  for (int d1 = 0; d1 < dim; ++d1)
                    for (int d2 = 0; d2 < dim; ++d2)
                      third_derivative(c, d0, d1, d2, j) = src.third_derivatives(c, d0, d1, d2).data[i];
          }

        private:
          static constexpr size_t n_third_columns = n * dim * dim * dim;
          static constexpr size_t third_column(const size_t c, const size_t d0, const size_t d1, const size_t d2)
          {
            return ((c * dim + d0) * dim + d1) * dim + d2;
          }
          std::vector<NT> third;
        };

        /// The AD batch and output of stacked_directions, kept by the caller across calls: allocating them anew
        /// for every call costs more than the evaluation when the calls are small (face chunks).
        template <typename ADNumber, typename Batch, size_t n_out> struct StackedWorkspace {
          typename Batch::template rebind<ADNumber> ad;
          BatchOutput<Batch::dim, ADNumber, n_out> out;
        };

        /// The number of points stacked_directions evaluates together.
        inline size_t stacked_chunk_size(const size_t n_points, const size_t n_directions, const size_t max_stacked)
        {
          return std::min(n_points, std::max<size_t>(1, max_stacked / n_directions));
        }

        /**
         * @brief Forward-mode derivatives of a batch evaluation along n_directions input directions at every point,
         * with the directions stacked along the point axis as in DiFfRG::internal::seed_stacked_jacobian.
         *
         * The points are taken in chunks of at most max_stacked / n_directions; block b of the AD batch of a chunk
         * is a copy of its points in which seed(ad_batch, direction, slot) has seeded direction b, so one
         * evaluation of at most max_stacked points covers every direction of the chunk. Small chunks keep the AD
         * batch in cache; a chunk is never smaller than one point, nor a direction group than one direction.
         *
         * Only the points [begin, end) of @p batch are evaluated.
         *
         * @tparam ADNumber the AD type, e.g. autodiff::Real<1, double> or autodiff::Real<2, double>
         * @tparam n_out the number of output components of the evaluation
         * @param evaluate callable(BatchOutput<ADNumber> &out, const ADBatch &batch)
         * @param seed callable(ADBatch &batch, size_t direction, size_t slot)
         * @param read callable(const BatchOutput<ADNumber> &out, size_t direction, size_t point, size_t slot)
         * @param chunk_done callable(size_t first_point, size_t n_points), after every direction of a chunk was read;
         * chunks start at begin + multiples of stacked_chunk_size()
         */
        template <typename ADNumber, size_t n_out, typename Batch, typename Evaluate, typename Seed, typename Read,
                  typename ChunkDone = decltype([](size_t, size_t) {})>
        void stacked_directions(StackedWorkspace<ADNumber, Batch, n_out> &workspace, const Batch &batch,
                                const size_t begin, const size_t end, const size_t n_directions,
                                const size_t max_stacked, const Term terms, const Evaluate &evaluate, const Seed &seed,
                                const Read &read, const ChunkDone &chunk_done = {})
        {
          using ADBatch = typename Batch::template rebind<ADNumber>;
          static_assert(std::is_same_v<typename ADBatch::extractors_type, typename Batch::extractors_type>,
                        "stacked_directions shares the batch's extractors with the AD batch, so they must stay frozen "
                        "(as in TraceBatch).");
          const size_t n = end - begin;
          if (n == 0 || n_directions == 0) return;

          auto &ad = workspace.ad;
          auto &out = workspace.out;
          const size_t chunk = stacked_chunk_size(n, n_directions, max_stacked);
          const size_t per_group = std::clamp<size_t>(max_stacked / chunk, 1, n_directions);
          for (size_t p0 = begin; p0 < end; p0 += chunk) {
            const size_t m = std::min(chunk, end - p0);
            for (size_t g0 = 0; g0 < n_directions; g0 += per_group) {
              const size_t G = std::min(per_group, n_directions - g0);
              ad.reinit(G * m, batch.has_derivatives(), batch.has_hessians());
              ad.set_shared(batch.extractors(), batch.variables());
              out.reinit(G * m, terms);
              tbb::parallel_for(tbb::blocked_range<size_t>(0, m), [&](const tbb::blocked_range<size_t> &r) {
                for (size_t i = r.begin(); i != r.end(); ++i)
                  for (size_t b = 0; b < G; ++b) {
                    ad.copy_point(b * m + i, batch, p0 + i);
                    seed(ad, g0 + b, b * m + i);
                  }
              });
              evaluate(out, std::as_const(ad));
              tbb::parallel_for(tbb::blocked_range<size_t>(0, m), [&](const tbb::blocked_range<size_t> &r) {
                for (size_t i = r.begin(); i != r.end(); ++i)
                  for (size_t b = 0; b < G; ++b)
                    read(std::as_const(out), g0 + b, p0 + i, b * m + i);
              });
            }
            chunk_done(p0, m);
          }
        }
      } // namespace internal
    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG
