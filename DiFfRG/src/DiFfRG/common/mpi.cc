#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/kokkos.hh>

namespace DiFfRG
{
  namespace MPI
  {
#ifdef HAVE_MPI

    uint rank(MPI_Comm comm)
    {
      int rank;
      MPI_Comm_rank(comm, &rank);
      return rank;
    }
    uint size(MPI_Comm comm)
    {
      int size;
      MPI_Comm_size(comm, &size);
      return size;
    }
    void barrier(MPI_Comm comm)
    {
      Kokkos::fence();
      MPI_Barrier(comm);
    }

    int sum_reduce(MPI_Comm comm, int *data, int size)
    {
      int result;
      MPI_Allreduce(data, &result, size, MPI_INT, MPI_SUM, comm);
      return result;
    }
    int max_reduce(MPI_Comm comm, int *data, int size)
    {
      int result;
      MPI_Allreduce(data, &result, size, MPI_INT, MPI_MAX, comm);
      return result;
    }
    int min_reduce(MPI_Comm comm, int *data, int size)
    {
      int result;
      MPI_Allreduce(data, &result, size, MPI_INT, MPI_MIN, comm);
      return result;
    }

    int sum_reduce(MPI_Comm comm, int value)
    {
      int result;
      MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_SUM, comm);
      return result;
    }
    int max_reduce(MPI_Comm comm, int value)
    {
      int result;
      MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_MAX, comm);
      return result;
    }
    int min_reduce(MPI_Comm comm, int value)
    {
      int result;
      MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_MIN, comm);
      return result;
    }

    double sum_reduce(MPI_Comm comm, double *data, int size)
    {
      double result;
      MPI_Allreduce(data, &result, size, MPI_DOUBLE, MPI_SUM, comm);
      return result;
    }
    double max_reduce(MPI_Comm comm, double *data, int size)
    {
      double result;
      MPI_Allreduce(data, &result, size, MPI_DOUBLE, MPI_MAX, comm);
      return result;
    }
    double min_reduce(MPI_Comm comm, double *data, int size)
    {
      double result;
      MPI_Allreduce(data, &result, size, MPI_DOUBLE, MPI_MIN, comm);
      return result;
    }

    double sum_reduce(MPI_Comm comm, double value)
    {
      double result;
      MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_SUM, comm);
      return result;
    }
    double max_reduce(MPI_Comm comm, double value)
    {
      double result;
      MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, comm);
      return result;
    }
    double min_reduce(MPI_Comm comm, double value)
    {
      double result;
      MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MIN, comm);
      return result;
    }

#else

    uint rank(MPI_Comm) { return 0; }
    uint size(MPI_Comm) { return 1; }
    void barrier(MPI_Comm) { Kokkos::fence(); }

    int sum_reduce(MPI_Comm, int *data, int size)
    {
      int result = 0;
      for (int i = 0; i < size; ++i)
        result += data[i];
      return result;
    }
    int max_reduce(MPI_Comm, int *data, int size)
    {
      int result = data[0];
      for (int i = 1; i < size; ++i)
        result = std::max(result, data[i]);
      return result;
    }
    int min_reduce(MPI_Comm, int *data, int size)
    {
      int result = data[0];
      for (int i = 1; i < size; ++i)
        result = std::min(result, data[i]);
      return result;
    }

    int sum_reduce(MPI_Comm, int value) { return value; }
    int max_reduce(MPI_Comm, int value) { return value; }
    int min_reduce(MPI_Comm, int value) { return value; }

    double sum_reduce(MPI_Comm, double *data, int size)
    {
      double result = 0;
      for (int i = 0; i < size; ++i)
        result += data[i];
      return result;
    }
    double max_reduce(MPI_Comm, double *data, int size)
    {
      double result = data[0];
      for (int i = 1; i < size; ++i)
        result = std::max(result, data[i]);
      return result;
    }
    double min_reduce(MPI_Comm, double *data, int size)
    {
      double result = data[0];
      for (int i = 1; i < size; ++i)
        result = std::min(result, data[i]);
      return result;
    }

    double sum_reduce(MPI_Comm, double value) { return value; }
    double max_reduce(MPI_Comm, double value) { return value; }
    double min_reduce(MPI_Comm, double value) { return value; }
#endif
  } // namespace MPI
} // namespace DiFfRG