#include <DiFfRG/common/utils.hh>
#include <DiFfRG/physics/flow_equations.hh>

void probe(DiFfRG::FlowEquations &flow)
{
  DiFfRG::build_logger("legacy", "legacy.log");
  flow.print_parameters(std::string("legacy"));
}
