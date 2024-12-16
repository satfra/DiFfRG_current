// external libraries
#include <deal.II/lac/block_vector.h>

// DiFfRG
#include <DiFfRG/timestepping/solver/newton.hh>

template class DiFfRG::Newton<dealii::BlockVector<double>>;
template class DiFfRG::Newton<dealii::Vector<double>>;