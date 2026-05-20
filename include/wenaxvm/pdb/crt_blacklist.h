#pragma once

// Hardcoded blacklist of MSVC CRT / SEH / CFG / RTC helpers that must NEVER
// be virtualised. Touching any of these from inside the VM blob breaks
// either process startup (mainCRTStartup -> __scrt_common_main_seh ->
// __security_init_cookie) or per-call SEH/CFG integrity.
//
// Conceptually equivalent to VMProtect's hardcoded import tables
// (kernel32_info[], user32_info[], default_info[]) -- the difference is
// VMProtect tags imports by name from those tables while we tag PDB symbols
// by name. Same idea: a curated allow-by-exception set.

#include <string_view>

namespace wenaxvm::pdb {

// True if `name` belongs to the hardcoded CRT/security helper blacklist.
// Match is case-sensitive and uses substring / prefix / exact rules
// chosen to match MSVC mangling conventions.
bool is_crt_blacklisted(std::string_view name) noexcept;

}  // namespace wenaxvm::pdb
