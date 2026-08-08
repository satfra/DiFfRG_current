function(add_migration_probe TARGET SOURCE)
  add_library(${TARGET} OBJECT EXCLUDE_FROM_ALL ${SOURCE})
  target_link_libraries(${TARGET} PRIVATE DiFfRG::DiFfRG)
endfunction()

function(add_migration_contract_test NAME TARGET EXPECT_SUCCESS EXPECTED_DIAGNOSTICS)
  get_target_property(probe_sources ${TARGET} SOURCES)
  list(GET probe_sources 0 probe_source)
  add_test(
    NAME ${NAME}
    COMMAND
      ${CMAKE_COMMAND} -DBUILD_DIR=${CMAKE_BINARY_DIR} -DTARGET_NAME=${TARGET}
      -DSOURCE_TO_TOUCH=${CMAKE_CURRENT_SOURCE_DIR}/${probe_source} -DEXPECT_SUCCESS=${EXPECT_SUCCESS}
      "-DEXPECTED_DIAGNOSTICS=${EXPECTED_DIAGNOSTICS}" -P
      ${CMAKE_CURRENT_SOURCE_DIR}/check_compile_contract.cmake)
  set_tests_properties(${NAME} PROPERTIES LABELS "api;migration" RUN_SERIAL TRUE)
endfunction()

function(add_clean_migration_contract_test NAME TARGET FORBIDDEN_DIAGNOSTICS)
  get_target_property(probe_sources ${TARGET} SOURCES)
  list(GET probe_sources 0 probe_source)
  add_test(
    NAME ${NAME}
    COMMAND
      ${CMAKE_COMMAND} -DBUILD_DIR=${CMAKE_BINARY_DIR} -DTARGET_NAME=${TARGET}
      -DSOURCE_TO_TOUCH=${CMAKE_CURRENT_SOURCE_DIR}/${probe_source} -DEXPECT_SUCCESS=TRUE
      "-DFORBIDDEN_DIAGNOSTICS=${FORBIDDEN_DIAGNOSTICS}" -P
      ${CMAKE_CURRENT_SOURCE_DIR}/check_compile_contract.cmake)
  set_tests_properties(${NAME} PROPERTIES LABELS "api;migration" RUN_SERIAL TRUE)
endfunction()

add_migration_probe(migration_callable_probe callable_probe.cc)
add_migration_contract_test(
  migration_callable_diagnostics migration_callable_probe TRUE
  "OutputPath(json).run_file|OutputPath(json).run_name|OutputPath(json).field_directory|OutputPath(json).root|CsvOutput(top_folder, output_name)|OutputSettings(json), active|OutputSettings(json).configuration_json|ConfigurationMesh<dim>")

add_migration_probe(migration_data_output_failure data_output_failure.cc)
add_migration_contract_test(
  migration_data_output_diagnostic migration_data_output_failure FALSE
  "Construct OutputPath, then OutputSession<dim, VectorType>, and pass the session to the timestepper|attach_data_output(OutputFrame<dim, VectorType>&, ...)|auto fields = output.fields(); fields.attach(...)")

add_migration_probe(migration_affine_failure affine_failure.cc)
add_migration_contract_test(
  migration_affine_diagnostic migration_affine_failure FALSE
  "apply_boundary_affine_constraints|apply_affine_constraints|AffineConstraintContext")

add_migration_probe(migration_assembler_log_failure assembler_log_failure.cc)
add_migration_contract_test(
  migration_assembler_log_diagnostic migration_assembler_log_failure FALSE
  "Construct the assembler with output.log_port() and call log() instead")

add_migration_probe(migration_readout_failure readout_failure.cc)
add_migration_contract_test(
  migration_readout_diagnostic migration_readout_failure FALSE
  "stable readout ID: helper(id, EoM, outputter)")

add_migration_probe(migration_output_callback_failure output_callback_failure.cc)
add_migration_contract_test(
  migration_output_callback_diagnostic migration_output_callback_failure FALSE
  "attach_data_output(OutputFrame<dim, VectorType>&, ...) instead")

add_migration_probe(migration_output_frame_csv_failure output_frame_csv_failure.cc)
add_migration_contract_test(
  migration_output_frame_csv_diagnostic migration_output_frame_csv_failure FALSE
  "auto table = output.table(name)")

add_migration_probe(migration_output_frame_lambda_failure output_frame_lambda_failure.cc)
add_migration_contract_test(
  migration_output_frame_lambda_diagnostic migration_output_frame_lambda_failure FALSE
  "TableRecord returned by output.table(name)")

add_migration_probe(migration_output_frame_fields_failure output_frame_fields_failure.cc)
add_migration_contract_test(
  migration_output_frame_fields_diagnostic migration_output_frame_fields_failure FALSE
  "auto fields = output.fields()|fields.attach(...)")

add_migration_probe(migration_inherited_log_port_constructor inherited_log_port_constructor.cc)
add_migration_contract_test(
  migration_inherited_log_port_constructor_diagnostic migration_inherited_log_port_constructor TRUE
  "Pass output.log_port() or an intentional LogPort{}")

add_migration_probe(migration_direct_log_port_constructor direct_log_port_constructor.cc)
add_migration_contract_test(
  migration_direct_log_port_constructor_diagnostic migration_direct_log_port_constructor TRUE
  "Pass output.log_port() or an intentional LogPort{}")

add_migration_probe(migration_explicit_log_port_constructor explicit_log_port_constructor.cc)
add_clean_migration_contract_test(
  migration_explicit_log_port_constructor_clean migration_explicit_log_port_constructor
  "deprecated-declarations|warning: 'legacy_default_log_port")

add_migration_probe(migration_json_value_alias json_value_alias.cc)
add_migration_contract_test(
  migration_json_value_alias_diagnostic migration_json_value_alias TRUE
  "JSONValue was renamed. Use DiFfRG::ConfigTree instead")
