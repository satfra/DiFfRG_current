#pragma once

// DiFfRG
#include <DiFfRG/discretization/FEM/assembler/dg.hh>

namespace DiFfRG
{
  namespace dDG
  {
    /**
     * @brief The DG assembler whose model also sees the derivatives and hessians of the FE functions, in
     * flux, source and numerical flux. See DG::internal::BatchedAssembler.
     */
    template <typename Discretization,
              typename Model = typename DiFfRG::internal::assembler_model_of<Discretization>::type>
    class Assembler : public DG::internal::BatchedAssembler<Discretization, Model, true>
    {
    public:
      using DG::internal::BatchedAssembler<Discretization, Model, true>::BatchedAssembler;
    };
  } // namespace dDG
} // namespace DiFfRG
