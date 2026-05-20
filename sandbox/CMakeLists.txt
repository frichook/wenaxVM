# Sandbox: tiny console target binary used as M1 virtualization target.
# Built with full PDB so wenaxvm-cli can resolve symbols.

add_executable(sandbox sandbox.cpp)

target_compile_features(sandbox PRIVATE cxx_std_20)

if(MSVC)
    # /O2 keeps the test functions register-only (no stack spills) -- the
    # lifter only handles reg/imm operands. /Ob1 + __declspec(noinline) in
    # source keep symbols intact so PDB can resolve them.
    target_compile_options(sandbox PRIVATE /Zi /O2 /Ob1)
    target_link_options(sandbox PRIVATE /DEBUG:FULL /INCREMENTAL:NO /OPT:NOREF /OPT:NOICF)
endif()
target_link_libraries(sandbox PRIVATE user32)  # for MessageBeep

set_target_properties(sandbox PROPERTIES
    FOLDER "wenaxvm"
    # Make compute() non-inlined and reachable by symbol name.
    COMPILE_PDB_NAME "sandbox"
    PDB_NAME         "sandbox")
