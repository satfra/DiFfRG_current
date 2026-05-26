#pragma once

#include <DiFfRG/common/utils.hh>
#include <deal.II/base/mpi.h>

namespace DiFfRG
{
#ifndef DEAL_II_WITH_MPI
  // Without MPI, deal.II (>= 9.7) declares the MPI types and constants in
  // namespace dealii rather than at global scope. Pull them into namespace
  // DiFfRG so that the API below resolves identically with and without MPI.
  using dealii::MPI_Comm;
  using dealii::MPI_Datatype;
  using dealii::MPI_Op;
  using dealii::MPI_Request;

  using dealii::MPI_COMM_NULL;
  using dealii::MPI_COMM_SELF;
  using dealii::MPI_COMM_WORLD;
  using dealii::MPI_LOR;
  using dealii::MPI_MAX;
  using dealii::MPI_MIN;
  using dealii::MPI_REQUEST_NULL;
  using dealii::MPI_SUM;
#endif

  namespace MPI
  {
    uint rank(MPI_Comm comm);
    uint size(MPI_Comm comm);
    void barrier(MPI_Comm comm);

    int sum_reduce(MPI_Comm comm, int *data, int size);
    int max_reduce(MPI_Comm comm, int *data, int size);
    int min_reduce(MPI_Comm comm, int *data, int size);

    int sum_reduce(MPI_Comm comm, int value);
    int max_reduce(MPI_Comm comm, int value);
    int min_reduce(MPI_Comm comm, int value);

    double sum_reduce(MPI_Comm comm, double *data, int size);
    double max_reduce(MPI_Comm comm, double *data, int size);
    double min_reduce(MPI_Comm comm, double *data, int size);

    double sum_reduce(MPI_Comm comm, double value);
    double max_reduce(MPI_Comm comm, double value);
    double min_reduce(MPI_Comm comm, double value);
  } // namespace MPI
} // namespace DiFfRG
