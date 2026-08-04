// standard library
#include <fstream>

// external libraries
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/data/fe_output.hh>

#include <hdf5lib/hdf5.hh>

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace DiFfRG
{
  using namespace dealii;

  template <uint dim, typename VectorType>
  FEOutput<dim, VectorType>::FEOutput(std::string top_folder, std::string output_name, std::string output_folder,
                                      const OutputSettings &settings, const bool active)
      : top_folder(make_folder(top_folder)), output_name(output_name), output_folder(make_folder(output_folder)),
        filename_pvd(output_name + ".pvd"), buffer_size([&]() {
          if (!settings.asynchronous) return 1u;
          const uint read_size = settings.max_pending_frames;
          return std::max(2u, read_size);
        }()),
        max_pending_bytes(settings.max_pending_bytes), series_number(0), subdivisions(settings.subdivisions),
        data_outs(buffer_size), save_vtk(active && settings.write_vtk)
  {
    if (active) create_folder(this->top_folder);
    if (save_vtk) create_folder(this->top_folder + this->output_folder);

    // Initialize the attached_solutions list.
    attached_solutions.emplace_back();
    attached_bytes.push_back(0);
  }

  template <uint dim, typename VectorType> void FEOutput<dim, VectorType>::set_hdf5_output(HDF5Output *output)
  {
    this->hdf5_output = output;
  }

  template <uint dim, typename VectorType> void FEOutput<dim, VectorType>::update_buffers()
  {
    if (buffer_size == 1) return;

    if (output_threads.size() + 1 != attached_solutions.size())
      throw std::logic_error(
          "FEOutput::flush: The number of output threads does not match the size of the attached_solutions lisea.");

    // If the number of attached solutions exceeds the buffer size, we clear the buffers.
    while (attached_solutions.size() > buffer_size) {
      retire_oldest();
    }
  }

  template <uint dim, typename VectorType> void FEOutput<dim, VectorType>::retire_oldest()
  {
    if (output_threads.empty()) return;
    if (output_threads.front().joinable()) output_threads.front().join();
    data_outs[output_thread_series.front() % buffer_size].clear();
    output_threads.pop_front();
    output_thread_series.pop_front();
    attached_solutions.pop_front();
    pending_bytes -= attached_bytes.front();
    attached_bytes.pop_front();
  }

  template <uint dim, typename VectorType> void FEOutput<dim, VectorType>::reserve_bytes(const std::size_t bytes)
  {
    while (max_pending_bytes > 0 && pending_bytes + bytes > max_pending_bytes && !output_threads.empty())
      retire_oldest();
    pending_bytes += bytes;
    attached_bytes.back() += bytes;
  }

  template <uint dim, typename VectorType> FEOutput<dim, VectorType>::~FEOutput() noexcept
  {
    try {
      finish();
    } catch (...) {
      // Destructors must not terminate the process. Timestepper run boundaries
      // call drain() to observe worker errors before teardown.
    }
  }

  template <uint dim, typename VectorType> void FEOutput<dim, VectorType>::drain()
  {
    if (finished) {
      std::exception_ptr to_throw;
      {
        std::lock_guard<std::mutex> lk(exception_mutex);
        to_throw = stored_exception;
        stored_exception = nullptr;
      }
      if (to_throw) std::rethrow_exception(to_throw);
      return;
    }

    for (auto &thread : output_threads)
      if (thread.joinable()) thread.join();

    // Workers retain their DataOut contents until the main thread joins them.
    for (const uint series : output_thread_series)
      data_outs[series % buffer_size].clear();

    output_threads.clear();
    output_thread_series.clear();
    attached_solutions.clear();
    attached_solutions.emplace_back();
    attached_bytes.clear();
    attached_bytes.push_back(0);
    pending_bytes = 0;

    // Surface a captured worker exception at the explicit run boundary.
    std::exception_ptr to_throw;
    {
      std::lock_guard<std::mutex> lk(exception_mutex);
      to_throw = stored_exception;
      stored_exception = nullptr;
    }
    if (to_throw) std::rethrow_exception(to_throw);
  }

  template <uint dim, typename VectorType> void FEOutput<dim, VectorType>::finish()
  {
    if (finished) {
      std::exception_ptr to_throw;
      {
        std::lock_guard<std::mutex> lk(exception_mutex);
        to_throw = stored_exception;
        stored_exception = nullptr;
      }
      if (to_throw) std::rethrow_exception(to_throw);
      return;
    }

    std::exception_ptr drain_error;
    try {
      drain();
    } catch (...) {
      drain_error = std::current_exception();
    }

    // Free the DataOut copies of the solution vectors and drain GrowingVectorMemory's shared pool
    // now, while we are on the main thread and Kokkos is still alive, rather than leaving these
    // (Kokkos-backed) deallocations for program teardown. NB: a rare exit-time crash can still
    // occur deeper down, inside deal.II's GrowingVectorMemory Kokkos finalize hook, when that
    // hook registry is corrupted by concurrent lazy registration during parallel vector
    // allocation -- that is an upstream deal.II/Kokkos teardown race, tracked separately.
    data_outs.clear();
    GrowingVectorMemory<VectorType>::release_unused_memory();

    finished = true;
    if (drain_error) std::rethrow_exception(drain_error);
  }

  template <uint dim, typename VectorType> void FEOutput<dim, VectorType>::flush(double time)
  {
    if (finished) throw std::logic_error("FEOutput::flush: output has already been finished.");
    // Surface any exception captured from a previous worker thread before doing more work.
    {
      std::exception_ptr to_throw;
      {
        std::lock_guard<std::mutex> lk(exception_mutex);
        if (stored_exception) {
          to_throw = stored_exception;
          stored_exception = nullptr;
        }
      }
      if (to_throw) std::rethrow_exception(to_throw);
    }

    update_buffers();

    if (!save_vtk && hdf5_output == nullptr) {
      data_outs[series_number % buffer_size].clear();
      attached_solutions.back().clear();
      pending_bytes -= attached_bytes.back();
      attached_bytes.back() = 0;
      series_number++;
      return;
    }

    // The .vtu file will be named like output_name_000001.vtu, where the number is the
    // series number. This imposes a limit of 1 million series, which is more than enough for most applications.
    const std::string filename_vtu =
        output_folder + output_name + "_" + Utilities::int_to_string(series_number, 6) + ".vtu";

    auto &m_data_out = data_outs[series_number % buffer_size];

    m_data_out.build_patches(subdivisions);

    if (hdf5_output != nullptr) {
      auto h5_file = hdf5_output->get_file();
      auto h5_group = h5_file.root().open_group("FE");
      {
        auto cur_group = h5_group.create_group(Utilities::int_to_string(series_number, 6));
        cur_group.write_attribute("time", time);
        cur_group.write_attribute("series_number", static_cast<int>(series_number));
        cur_group.write_attribute("output_name", output_name);

        DataOutBase::DataOutFilterFlags mflags(false, false);
        DataOutBase::DataOutFilter data_filter(mflags);
        m_data_out.write_filtered_data(data_filter);
        std::vector<double> node_data;
        data_filter.fill_node_data(node_data);

        auto nodes_space = DiFfRG::hdf5::Dataspace::simple({data_filter.n_nodes(), dim});
        auto nodes = cur_group.create_dataset("nodes", DiFfRG::hdf5::type_of<double>(), nodes_space);
        nodes.write(node_data);

        for (uint i = 0; i < data_filter.n_data_sets(); ++i) {
          auto data_space = DiFfRG::hdf5::Dataspace::simple({data_filter.n_nodes()});
          const std::string name = data_filter.get_data_set_name(i);
          auto dataset = cur_group.create_dataset(name, DiFfRG::hdf5::type_of<double>(), data_space);
          const double *data_set_data = data_filter.get_data_set(i);
          dataset.write(data_set_data, data_filter.n_nodes());
        }
      }
      hdf5_output->close_file();
    }

    auto output_func = [=, this](const uint m_series_number, const double m_time) {
      try {
        auto &m_data_out = data_outs[m_series_number % buffer_size];
        if (save_vtk) {
          auto flags = DataOutBase::VtkFlags(m_time, m_series_number);
          m_data_out.set_flags(flags);

          std::ofstream output_vtu(top_folder + filename_vtu);
          m_data_out.write_vtu(output_vtu);
          if (!output_vtu) throw std::runtime_error("FEOutput::flush: Could not write vtu file.");

          // Publish the PVD entry only after its VTU exists completely.
          {
            std::unique_lock<std::mutex> lk(output_mutex);
            output_condition.wait(lk, [&]() { return publication_failed || m_series_number == next_published_series; });
            if (publication_failed)
              throw std::runtime_error("FEOutput::flush: An earlier VTK frame could not be published.");
            time_series.emplace_back(m_time, filename_vtu);
            try {
              std::ofstream output_pvd(top_folder + filename_pvd);
              DataOutBase::write_pvd_record(output_pvd, time_series);
              if (!output_pvd) throw std::runtime_error("write failed");
            } catch (const std::exception &e) {
              throw std::runtime_error("FEOutput::flush: Could not write pvd file.");
            }
            ++next_published_series;
            lk.unlock();
            output_condition.notify_all();
          }
        }
        // NB: m_data_out.clear() is intentionally NOT called here -- it frees Kokkos-backed
        // vectors and is performed on the main thread (synchronous path below, or update_buffers
        // after this worker is joined) to avoid concurrent Kokkos dispatch from multiple threads.
      } catch (...) {
        // Never let an exception escape the worker thread (that would call
        // std::terminate). Capture the first one; later workers' exceptions are
        // dropped because the first is enough to fail the main-thread caller.
        {
          std::lock_guard<std::mutex> lk(exception_mutex);
          if (!stored_exception) stored_exception = std::current_exception();
        }
        {
          std::lock_guard<std::mutex> lk(output_mutex);
          publication_failed = true;
        }
        output_condition.notify_all();
      }
    };

    // If the buffer size is 1, we save ourselves the cost of spawning a thread
    if (buffer_size == 1) {
      output_func(series_number, time);
      // synchronous path: clear the DataOut here on the main thread (see note in output_func)
      data_outs[series_number % buffer_size].clear();
      attached_solutions.back().clear();
      pending_bytes -= attached_bytes.back();
      attached_bytes.back() = 0;
    } else {
      output_threads.emplace_back(output_func, series_number, time);
      output_thread_series.push_back(series_number);
      // We need to prepare the next attached_solutions list
      attached_solutions.emplace_back();
      attached_bytes.push_back(0);
    }

    series_number++;
  }

  template <uint dim, typename VectorType>
  void FEOutput<dim, VectorType>::attach(const DoFHandler<dim> &dof_handler, const VectorType &solution,
                                         const std::string &name)
  {
    update_buffers();
    reserve_bytes(solution.size() * sizeof(typename VectorType::value_type));

    attached_solutions.back().emplace_back(mem);
    *(attached_solutions.back().back()) = solution;

    auto &m_data_out = data_outs[series_number % buffer_size];
    m_data_out.add_data_vector(dof_handler, *(attached_solutions.back().back()), name);
  }

  template <uint dim, typename VectorType>
  void FEOutput<dim, VectorType>::attach(const DoFHandler<dim> &dof_handler, const VectorType &solution,
                                         const std::vector<std::string> &names)
  {
    update_buffers();
    reserve_bytes(solution.size() * sizeof(typename VectorType::value_type));

    attached_solutions.back().emplace_back(mem);
    *(attached_solutions.back().back()) = solution;

    auto &m_data_out = data_outs[series_number % buffer_size];
    m_data_out.add_data_vector(dof_handler, *(attached_solutions.back().back()), names);
  }

  template <typename VectorType>
  FEOutput<0, VectorType>::FEOutput(std::string, std::string, std::string, const OutputSettings &, bool)
  {
  }

  template class FEOutput<0, dealii::Vector<double>>;
  template class FEOutput<1, dealii::Vector<double>>;
  template class FEOutput<2, dealii::Vector<double>>;
  template class FEOutput<3, dealii::Vector<double>>;
  template class FEOutput<0, dealii::BlockVector<double>>;
  template class FEOutput<1, dealii::BlockVector<double>>;
  template class FEOutput<2, dealii::BlockVector<double>>;
  template class FEOutput<3, dealii::BlockVector<double>>;
} // namespace DiFfRG
