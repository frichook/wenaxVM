add_executable(wenaxvm-cli main.cpp)

target_link_libraries(wenaxvm-cli PRIVATE wenaxvm-core)
target_compile_features(wenaxvm-cli PRIVATE cxx_std_23)

set_target_properties(wenaxvm-cli PROPERTIES FOLDER "wenaxvm")
