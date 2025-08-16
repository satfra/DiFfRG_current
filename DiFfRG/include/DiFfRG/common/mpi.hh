#pragma once

#include <DiFfRG/common/utils.hh>

#ifdef HAVE_MPI
#include <mpi.h>
#else
#ifndef MPI_Comm
#define MPI_Comm int
#define MPI_COMM_WORLD 1
#endif
#endif

namespace DiFfRG
{
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