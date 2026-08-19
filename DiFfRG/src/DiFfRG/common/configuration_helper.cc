// DiFfRG
#include <DiFfRG/common/configuration_helper.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/eom_config.hh>
#include <DiFfRG/discretization/data/output_path.hh>

// standard library
#include <filesystem>
#include <fstream>

namespace DiFfRG
{
  ConfigurationHelper::ConfigurationHelper(int argc, char *argv[], const std::string parameter_file)
      : parsed(false), parameter_file(parameter_file)
  {
    for (int i = 1; i < argc; ++i)
      args.emplace_back(argv[i]);
    try {
      parse();
    } catch (const std::exception &e) {
      std::cerr << "While reading the CLI arguments an error occurred:\n    " << e.what() << "\n" << std::endl;
      print_usage_message();
      exit(1);
    }
  }

  ConfigurationHelper::ConfigurationHelper(const ConfigTree &config) : config(config), parsed(true) {}

  ConfigurationHelper::ConfigurationHelper(const ConfigurationHelper &other)
  {
    if (other.parsed) throw std::runtime_error("A parsed ConfigurationHelper can not be copied!");

    parameter_file = other.parameter_file;

    args = other.args;
    cli_parameters = other.cli_parameters;
  }

  ConfigTree &ConfigurationHelper::get_config() { return config; }
  const ConfigTree &ConfigurationHelper::get_config() const { return config; }

  ConfigTree &ConfigurationHelper::get_json() { return config; }
  const ConfigTree &ConfigurationHelper::get_json() const { return config; }

  ConfigTree ConfigurationHelper::probe(int argc, char *argv[], const std::string parameter_file)
  {
    ConfigurationHelper helper;
    helper.parameter_file = parameter_file;
    helper.quiet = true;
    for (int i = 1; i < argc; ++i)
      helper.args.emplace_back(argv[i]);
    try {
      helper.parse();
    } catch (const std::exception &) {
      // A malformed or absent parameter file is not this function's problem to report: the
      // application's own ConfigurationHelper will hit the same error and diagnose it properly.
      return ConfigTree();
    }
    return helper.config;
  }

  void ConfigurationHelper::parse()
  {
    parsed = true;

    try {
      parse_cli();
    } catch (const std::exception &e) {
      if (quiet) throw;
      std::cerr << "While reading the CLI arguments an error occurred:\n    " << e.what() << "\n" << std::endl;
      print_usage_message();
      exit(1);
    }

    try {
      if (parameter_file.empty()) throw std::runtime_error("No parameter file specified.");
      // A default parameter file name is only a suggestion: if it is not on disk, but the
      // same name with a .toml extension is, take that. A file named explicitly with -p is
      // never second-guessed.
      if (!parameter_file_explicit && !std::filesystem::exists(parameter_file)) {
        const auto toml_alternative = std::filesystem::path(parameter_file).replace_extension(".toml");
        if (std::filesystem::exists(toml_alternative)) parameter_file = toml_alternative.string();
      }
      // parse parameter file and CLI, log it to file
      config = ConfigTree(parameter_file);

      for (const auto &param : cli_parameters) {
        std::string path, value, buf;

        // separate the str into a path to an entry and a value
        std::istringstream ss1(param.first);
        std::getline(ss1, path, '=');
        std::getline(ss1, value, '=');

        try {
          if (param.second == "double")
            config().at_pointer(path) = std::stod(value);
          else if (param.second == "bool")
            config().at_pointer(path) = (value == "true");
          else if (param.second == "int")
            config().at_pointer(path) = std::stoi(value);
          else if (param.second == "string")
            config().at_pointer(path) = value;
        } catch (const std::exception &e) {
          throw std::runtime_error("Failed to parse CLI parameter '" + path + "=" + value + "' as " + param.second +
                                   ": " + e.what());
        }
      }
    } catch (const std::exception &e) {
      if (quiet) throw;
      std::cerr << "While reading the parameter file an error occurred:\n    " << e.what() << "\n" << std::endl;
      print_usage_message();
      exit(1);
    }
  }

  void ConfigurationHelper::append(std::string str)
  {
    std::istringstream ss(str);
    std::string argv;
    while (std::getline(ss, argv, ' '))
      args.emplace_back(argv);
    try {
      parse_cli();
    } catch (const std::exception &e) {
      std::cerr << "While reading the CLI arguments an error occurred:\n    " << e.what() << "\n" << std::endl;
      print_usage_message();
      exit(1);
    }
  }

  void ConfigurationHelper::print_usage_message()
  {
    std::string help_text = R"(
  ╭━━━━╮ ╭━━━╮╭━┳━━━━┳━━━━╮
  ╰╮╭╮ ┃ ┃╭━━╯┃╭┫╭━━╮┃╭━╮ ┃
   ┃┃┃ ┣━┫╰━━┳╯╰┫╰━━╯┃┃ ╰━╯
   ┃┃┃ ┣━┫╭━━┻╮╭┫╭╮ ╭┫┃╭━━╮
  ╭╯╰╯ ┃ ┃┃for┃┃┃┃┃ ╰┫╰┻━━┃
  ╰━━━━┻━┻╯   ╰╯╰╯╰━━┻━━━━╯
    The Discretisation Framework for functional Renormalisation Group flows.

This is a DiFfRG simulation. You can pass the following optional parameters to this executable:
  --help                      shows this text
  --generate-parameter-file   generates a parameter file with some default values, in the
                              format implied by the parameter file name (JSON or TOML)
  -p                          specifiy a parameter file other than the standard parameter.json/toml
                              Files ending in .toml are read as TOML, everything else as JSON.
                              If no parameter file is given and parameter.json is absent, but
                              parameter.toml is present, the latter is used.
  -sd                         overwrite a double parameter. This should be in the format '-sd /physical/T=0.1'
  -si                         overwrite an integer parameter. This should be in the format '-si /physical/Nc=1'
  -sb                         overwrite a boolean parameter. This should be in the format '-sb /physical/use_sth=true'
  -ss                         overwrite a string parameter. This should be in the format '-ss /physical/a=hello'
)";
    std::cout << help_text << std::endl;
  }

  void ConfigurationHelper::parse_cli()
  {
    while (args.size()) {
      if (args.front() == std::string("--generate-parameter-file")) {
        // Both of these terminate the program, which a probe() must never do; the application's
        // own parse hits them a moment later and acts on them.
        if (quiet) {
          args.pop_front();
          continue;
        }
        generate_parameter_file();
        exit(0);
      } else if (args.front() == std::string("--help")) {
        if (quiet) {
          args.pop_front();
          continue;
        }
        print_usage_message();
        exit(0);
      } else if (args.front() == std::string("-p")) {
        if (args.size() == 1)
          throw std::runtime_error("Error: flag '-p' must be followed by the name of a parameter file.");
        args.pop_front();
        parameter_file = args.front();
        parameter_file_explicit = true;
        args.pop_front();
      } else if (args.front() == std::string("-sd")) {
        if (args.size() == 1) throw std::runtime_error("Error: flag '-sd' must be followed by a valid parameter.");
        args.pop_front();
        cli_parameters.push_back({args.front(), "double"});
        args.pop_front();
      } else if (args.front() == std::string("-si")) {
        if (args.size() == 1) throw std::runtime_error("Error: flag '-si' must be followed by a valid parameter.");
        args.pop_front();
        cli_parameters.push_back({args.front(), "int"});
        args.pop_front();
      } else if (args.front() == std::string("-sb")) {
        if (args.size() == 1) throw std::runtime_error("Error: flag '-sb' must be followed by a valid parameter.");
        args.pop_front();
        cli_parameters.push_back({args.front(), "bool"});
        args.pop_front();
      } else if (args.front() == std::string("-ss")) {
        if (args.size() == 1) throw std::runtime_error("Error: flag '-ss' must be followed by a valid parameter.");
        args.pop_front();
        cli_parameters.push_back({args.front(), "string"});
        args.pop_front();
      } else {
        if (!quiet)
          std::cerr << "WARNING: Unrecognized CLI argument '" << args.front() << "' was ignored. Use --help for usage."
                    << std::endl;
        args.pop_front();
      }
    }
  }

  void ConfigurationHelper::generate_parameter_file()
  {
    ConfigTree config = json::value(
        // json::object() rather than {}: an empty braced list is deduced as an array, so
        // the generated file used to contain "physical": [{}] instead of an empty section.
        {{"physical", json::object()},
         {"integration",
          // These names must match what internal::make_int_grid() and optimize_x_extent() read;
          // the file used to advertise x_quadrature_order/angle_quadrature_order, which nothing
          // ever looked up, so a freshly generated file threw on the first integrator.
          {{"x_order", 32},
           {"cos1_order", 8},
           {"cos2_order", 8},
           {"phi_order", 8},
           {"x_extent_tolerance", 1e-5},
           {"jacobian_quadrature_factor", 0.5},
           {"vacuum_quad_size", 64},
           {"matsubara_precision_factor", 1},
           {"min_matsubara_size", 8},
           {"max_matsubara_size", 128}}},
         {"discretization",
          {{"fe_order", 3},
           {"threads", 0},
           {"kokkos_threads", 0},
           {"mesh_workers", 8},
           {"batch_size", 16},
           {"overintegration", 0},
           {"output_subdivisions", 2},

           {"EoM_abs_tol", Config::EoMConfig::default_abs_tol},
           {"EoM_max_iter", Config::EoMConfig::default_max_iter},
           {"EoM_smoothing_length", Config::EoMConfig::default_smoothing_length},
           {"EoM_bound_tolerance", Config::EoMConfig::default_bound_tolerance},
           {"EoM_armijo_coefficient", Config::EoMConfig::default_armijo_coefficient},
           {"EoM_max_backtracks", Config::EoMConfig::default_max_backtracks},

           {"grid", {{"x_grid", "0:0.1:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}},
           {"adaptivity",
            {{"start_adapt_at", 0.},
             {"adapt_dt", 1e-1},
             {"level", 0},
             {"refine_percent", 1e-1},
             {"coarsen_percent", 5e-2}}}}},
         {"timestepping",
          {{"final_time", 1.},
           {"output_dt", 1e-1},
           {"explicit",
            {{"dt", 1e-2},
             {"minimal_dt", 1e-6},
             {"maximal_dt", 1e-1},
             {"abs_tol", 1e-4},
             {"rel_tol", 1e-4},
             {"coupling_mode", 1}}},
           {"implicit",
            {{"dt", 1e-4},
             {"minimal_dt", 1e-6},
             {"maximal_dt", 1e-1},
             {"abs_tol", 1e-13},
             {"rel_tol", 1e-7},
             {"max_steps", 1000000}}}}},
         {"output", {{"verbosity", 0}, {"folder", "output/"}, {"name", "output"}, {"max_pending_frames", 2}}}});

    const auto temporary = parameter_file + ".tmp";
    {
      std::ofstream file(temporary, std::ofstream::trunc);
      if (!file) throw std::runtime_error("Could not create parameter file '" + parameter_file + "'.");
      // Write in whichever format the requested file name implies.
      if (internal::has_toml_extension(parameter_file))
        config.print_toml(file);
      else
        config.print(file);
      if (!file) throw std::runtime_error("Failed while writing parameter file '" + parameter_file + "'.");
    }
    std::filesystem::rename(temporary, parameter_file);

    std::cout << "\nGenerated parameter file: " << parameter_file << "\n\n"
              << "Parameter sections:\n"
              << "  /physical          Physical parameters (e.g. temperature, couplings, cutoff scale)\n"
              << "  /integration       Quadrature settings for momentum-space loop integrals\n"
              << "    x_order                       Order of radial momentum quadrature\n"
              << "    cos1/cos2/phi_order           Order of angular quadrature\n"
              << "    x_extent_tolerance            Tolerance for integration domain extent\n"
              << "    jacobian_quadrature_factor    Quadrature reduction factor for Jacobian assembly\n"
              << "    vacuum_quad_size                                                               \n"
              << "    matsubara_precision_factor                                                     \n"
              << "    min_matsubara_size                                                             \n"
              << "    max_matsubara_size                                                             \n"
              << "  /discretization    Spatial discretization settings\n"
              << "    fe_order                      Finite element polynomial order\n"
              << "    threads                       TBB/deal.II application thread limit. 0 = all\n"
              << "                                  available cores\n"
              << "    kokkos_threads                Kokkos host-backend thread count. 0 or absent =\n"
              << "                                  resolved application thread count\n"
              << "                                  DEAL_II_NUM_THREADS, if set, caps both counts.\n"
              << "    mesh_workers                  Number of in-flight cells in the assembly\n"
              << "                                  pipeline (MeshWorker queue length)\n"
              << "    batch_size                    Batch size for GPU kernel launches\n"
              << "    overintegration               Extra quadrature points beyond FE order\n"
              << "    output_subdivisions           VTK output subdivisions per cell\n"
              << "    EoM_abs_tol / EoM_max_iter    In-cell EoM minimizer tolerance and iteration limit\n"
              << "    EoM_smoothing_length           Physical EoM-potential smoothing length (-1: automatic)\n"
              << "    EoM_bound_tolerance            Reference-cell active-bound tolerance\n"
              << "    EoM_armijo_coefficient         Armijo sufficient-decrease coefficient\n"
              << "    EoM_max_backtracks             Maximum line-search backtracking steps\n"
              << "    grid/*                        Grid specification (format: \"start:step:stop\")\n"
              << "    adaptivity/*                  h-adaptive mesh refinement settings\n"
              << "  /timestepping      Time integration settings\n"
              << "    final_time                    End time for the RG flow\n"
              << "    output_dt                     Time interval between output snapshots\n"
              << "    explicit/*                    Explicit timestepper (dt, tolerances)\n"
              << "    implicit/*                    Implicit timestepper (dt, tolerances)\n"
              << "    explicit/coupling_mode        0: Lag, 1: Stagger (default), 2: Predict\n"
              << "  /output            Output settings\n"
              << "    verbosity                     Log verbosity level (0 = minimal)\n"
              << "    folder                        Output directory\n"
              << "    name                          Base name for output files\n"
              << "    max_pending_frames            Lossless output queue frame limit\n"
              << std::endl;
  }

  std::string ConfigurationHelper::get_log_file() const
  {
    return OutputPath(config).run_file(".log").filename().string();
  }
  std::string ConfigurationHelper::get_parameter_file() const { return parameter_file; }
  std::string ConfigurationHelper::get_output_name() const { return OutputPath(config).run_name(); }
  std::string ConfigurationHelper::get_output_folder() const
  {
    return make_folder(OutputPath(config).field_directory().generic_string());
  }
  std::string ConfigurationHelper::get_top_folder() const
  {
    return make_folder(OutputPath(config).root().generic_string());
  }

} // namespace DiFfRG
