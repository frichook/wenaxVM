# wenaxvm-core: static library with all virtualizer guts.
# Sources are added incrementally as each module lands.

add_library(wenaxvm-core STATIC
    # M1.3 codec/
    codec/disasm.cpp
    codec/encoder.cpp
    codec/polycrypt.cpp

    # M1.4 disasm/
    disasm/cfg.cpp

    # M1.5 ir/
    ir/commands.cpp

    # M1.6 ir/translator
    ir/translator.cpp

    # M1.7 vm/register_manager
    vm/register_manager.cpp

    # M1.8 vm/machine
    vm/machine.cpp

    # M1.9 pdb/pdb_resolver
    pdb/pdb_resolver.cpp

    # M3 pdb/crt_blacklist: hardcoded CRT/SEH/CFG/RTC helper deny names.
    pdb/crt_blacklist.cpp

    # M1.10 pe/pe_patcher
    pe/pe_patcher.cpp

    # M3 pe/pe_deny: auto-deny RVA collector (entry point, TLS callbacks,
    # EH handlers, LoadConfig SecurityCookie/GuardCF*).
    pe/pe_deny.cpp
)

target_include_directories(wenaxvm-core
    PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(wenaxvm-core
    PUBLIC
        Zydis
        linux_pe
        raw_pdb
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

target_compile_features(wenaxvm-core PUBLIC cxx_std_23)

set_target_properties(wenaxvm-core PROPERTIES FOLDER "wenaxvm")
