#include <DiFfRG/discretization/data/output_session.hh>

void probe(DiFfRG::OutputFrame<1, dealii::Vector<double>> &output)
{
  auto fields = output.fe_output();
  (void)fields;
}
