# ##############################################################################
# O(2) two-dimensional KT regression scaffold
# ##############################################################################

add_executable(o2_model_kt_2D_regression regression.cc)
target_link_libraries(o2_model_kt_2D_regression PRIVATE DiFfRG::DiFfRG Catch2::Catch2WithMain)
catch_discover_tests(o2_model_kt_2D_regression
  ADD_TAGS_AS_LABELS
  PROPERTIES TIMEOUT 3600)
