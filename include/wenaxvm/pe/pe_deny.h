#pragma once

// AUTO-DENY RVA collector. Equivalent in spirit to the parts of
// VMProtect's PEFile.cc that flag certain functions ctNone:
//   - AddressOfEntryPoint
//   - TLS Directory AddressOfCallBacks
//   - Exception Directory entries with UNW_FLAG_EHANDLER / UNW_FLAG_UHANDLER
//     (recursively chasing UNW_FLAG_CHAININFO)
//   - LoadConfig: SecurityCookie, GuardCFCheckFunctionPointer,
//     GuardCFDispatchFunctionPointer, GuardCFFunctionTable entries
//
// Touching any of these from inside the VM blob breaks the
// process startup (mainCRTStartup -> __scrt_common_main_seh ->
// __security_init_cookie) and exception unwinding.

#include <cstdint>
#include <span>
#include <unordered_set>

namespace wenaxvm::pe {

struct deny_set {
    std::unordered_set<std::uint32_t> rvas;
    // Diagnostic counters (logged by CLI).
    std::size_t entry_point{0};
    std::size_t tls_callbacks{0};
    std::size_t eh_handlers{0};
    std::size_t cfg_entries{0};
    std::size_t load_config_singletons{0};
};

// Walks PE directories on the in-memory image and returns a set of RVAs
// that must NEVER be virtualised. Pure read-only; does not modify image.
deny_set collect_pe_denies(std::span<const std::uint8_t> image_bytes);

}  // namespace wenaxvm::pe
