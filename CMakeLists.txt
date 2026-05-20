cmake_minimum_required(VERSION 3.25)

project(wenaxVM
    VERSION 0.1.0
    DESCRIPTION "x64 code virtualizer (VMProtect-style)"
    LANGUAGES C CXX ASM_MASM)

# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive- /Zc:preprocessor /utf-8 /EHsc)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(WENAXVM_BUILD_TESTS   "Build wenaxvm unit tests"           OFF)
option(WENAXVM_BUILD_SANDBOX "Build M1 sandbox target binary"     ON)

# -----------------------------------------------------------------------------
# Dependencies via FetchContent
# -----------------------------------------------------------------------------
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# Zydis — disassembler + encoder. MANDATORY.
FetchContent_Declare(
    Zydis
    GIT_REPOSITORY https://github.com/zyantific/zydis.git
    GIT_TAG        v4.1.0
    GIT_SHALLOW    TRUE
)
set(ZYDIS_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_DOXYGEN  OFF CACHE BOOL "" FORCE)
set(ZYDIS_FEATURE_ENCODER ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Zydis)

# linux-pe — header-only x64 PE parser used by EagleVM. Lightweight.
# We use it for M1-M3 to keep iteration fast; LIEF can be added later for M4.
FetchContent_Declare(
    linux-pe
    GIT_REPOSITORY https://github.com/can1357/linux-pe.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(linux-pe)

add_library(linux_pe INTERFACE)
target_include_directories(linux_pe INTERFACE
    "${linux-pe_SOURCE_DIR}/includes"
    "${linux-pe_SOURCE_DIR}")

# raw-pdb — header-only PDB parser from Molecular Matters.
FetchContent_Declare(
    raw_pdb
    GIT_REPOSITORY https://github.com/MolecularMatters/raw_pdb.git
    GIT_TAG        main
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(raw_pdb)

# spdlog — logging.
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.14.1
    GIT_SHALLOW    TRUE
)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL       OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(spdlog)

# nlohmann_json — JSON config parsing.
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlohmann_json)

if(WENAXVM_BUILD_TESTS)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.6.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Catch2)
    enable_testing()
endif()

# -----------------------------------------------------------------------------
# Sub-projects
# -----------------------------------------------------------------------------
add_subdirectory(src)
add_subdirectory(tools/wenaxvm-cli)

if(WENAXVM_BUILD_SANDBOX)
    add_subdirectory(sandbox)
endif()

if(WENAXVM_BUILD_TESTS)
    add_subdirectory(tests)
endif()
