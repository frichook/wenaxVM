#include "wenaxvm/pdb/crt_blacklist.h"

#include <string_view>

namespace wenaxvm::pdb {

namespace {

using sv = std::string_view;

// Prefix list. Anything starting with these is rejected.
constexpr sv kPrefix[] = {
    // ---- MSVC CRT initialisation / shutdown plumbing -----------------------
    "__scrt_",
    "__acrt_",
    "__crt_",
    "__vcrt_",
    "_scrt_",
    "_acrt_",
    "_vcrt_",
    "_crt_",
    "pre_cpp_init",
    "post_pgo_init",
    "_get_startup_",
    "_initialize_",
    "_register_thread_local_exe_atexit_callback",

    // ---- Stack probes / chkstk family --------------------------------------
    "__chkstk",            // stack probe — touched by every >4KB-frame fn
    "_chkstk",
    "__alloca_probe",
    "_alloca_probe",

    // ---- Stack/security cookies + CFG --------------------------------------
    "__security_",         // __security_init_cookie / __security_check_cookie
    "_security_",
    "__guard_",            // __guard_check_icall_fptr / __guard_dispatch_*
    "_guard_",
    "__GS",                // __GSHandlerCheck*, __GSCheckHandler*
    "_GS",

    // ---- Runtime error reporters -------------------------------------------
    "__report_",           // __report_gsfailure / __report_rangecheckfailure
    "_report_",
    "__raise_securityfailure",

    // ---- /RTCx runtime checks ----------------------------------------------
    "_RTC_",
    "RTC_",                // some PDBs drop the underscore on demangle

    // ---- C++ STL / type_info / RTTI ABI helpers ----------------------------
    "__std_",              // STL ABI helpers (init_once, calloc_crt, ...)
    "__cxa_",              // Itanium-ABI helpers if any
    "__std_terminate",
    "__std_type_info_",
    "__type_info_",
    "_TypeDescriptor_",
    "__RTDynamicCast",
    "__RTtypeid",
    "__RTCastToVoid",
    "__local_unwind",
    "_local_unwind",
    "__global_unwind",
    "_global_unwind",
    "__except_handler",
    "_except_handler",

    // ---- Exception / SEH handlers ------------------------------------------
    "__CxxFrameHandler",
    "__GSHandlerCheck",
    "__C_specific_handler",
    "_C_specific_handler",
    "__CxxThrowException",
    "_CxxThrowException",
    "_CxxFrameHandler",
    "__FrameHandler",

    // ---- Pure-virtual / vcall thunks ---------------------------------------
    "_purecall",
    "__purecall",

    // ---- Linker stubs (DllImport / lazy thunks) ----------------------------
    "__imp_",
    "__imp_load_",
    "_imp_",

    // ---- Intel/CPU feature dispatch ----------------------------------------
    "__intel_",
    "__cpu_",
    "_cpu_",
    "__isa_",
    "_isa_",

    // ---- setjmp/longjmp/coroutine helpers ----------------------------------
    "_setjmp",
    "__setjmp",
    "longjmp",
    "_longjmp",
    "__coro_",

    // ---- TLS callbacks emitted by /Zc:threadSafeInit -----------------------
    "_Init_thread_",
    "__Init_thread_",
    "Init_thread_",
    "_Tls_",
    "__tlregdtor",

    // ---- Misc startup / atexit machinery -----------------------------------
    "_onexit",
    "__onexit",
    "_atexit",
    "__atexit",
    "_invoke_watson",
    "_invalid_parameter",
    "__invalid_parameter",
};

// Exact-name list. Whole symbol must match.
constexpr sv kExact[] = {
    // Program/DLL entry stubs (synthesized by MSVC) -------------------------
    "mainCRTStartup",
    "wmainCRTStartup",
    "WinMainCRTStartup",
    "wWinMainCRTStartup",
    "_DllMainCRTStartup",
    "DllMainCRTStartup",
    // User entry stubs (these run before the CRT has fully spun up its TLS
    // and exception state; safer left native)
    "main",
    "wmain",
    "WinMain",
    "wWinMain",
    "DllMain",

    // ISA-dispatch / cpu-features --------------------------------------------
    "__isa_available_init",
    "__isa_available",
    "__isa_enabled",
    "__favor",
    "__cpu_features_init",
    "__cpu_features",

    // atexit machinery -------------------------------------------------------
    "atexit",
    "_onexit",
    "_register_onexit_function",
    "_execute_onexit_table",
    "_initialize_onexit_table",
    "_crt_atexit",
    "_crt_at_quick_exit",

    // initterm / exit / abort ------------------------------------------------
    "_initterm",
    "_initterm_e",
    "exit",
    "_exit",
    "quick_exit",
    "_cexit",
    "_c_exit",
    "abort",
    "_abort",
    "terminate",
    "_terminate",
    "unexpected",
    "_unexpected",

    // Security / cookie / GS reporters ---------------------------------------
    "__report_gsfailure",
    "__report_rangecheckfailure",
    "__GSHandlerCheck",
    "__security_init_cookie",
    "__security_check_cookie",
    "__security_check_cookie_seh",

    // /RTCs whole table ------------------------------------------------------
    "_RTC_Initialize",
    "_RTC_Terminate",
    "_RTC_CheckEsp",
    "_RTC_CheckStackVars",
    "_RTC_CheckStackVars2",
    "_RTC_AllocaHelper",
    "_RTC_UninitUse",
    "_RTC_Shutdown",
    "_RTC_GetErrDesc",
    "_RTC_NumErrors",
    "_RTC_SetErrorFunc",
    "_RTC_SetErrorType",
    "_RTC_StackFailure",

    // SEH thunks emitted by MSVC for any function with try/__try -----------
    // They run *during* unwinding — never call them as VM.
    "__GSHandlerCheck_SEH",
    "__GSHandlerCheck_EH",
    "__GSHandlerCheck_EH4",
    "__CxxFrameHandler3",
    "__CxxFrameHandler4",
    "__C_specific_handler",
    "__except_validate_context_record",
    "__except_validate_jump_buffer",

    // long-jump / setjmp -----------------------------------------------------
    "_RtlGuardCheckLongJumpTarget",
    "_setjmp",
    "_setjmpex",
    "longjmp",

    // Pure virtual / fastfail ------------------------------------------------
    "_purecall",
    "__fastfail",

    // memcpy/memset family (small inline helpers MSVC sometimes emits as
    // standalone IAT-like functions; their bodies often rely on vector
    // instructions we don't lift, and they're hammered by *every* virtualised
    // function -- safer left native)
    "memcpy",
    "memmove",
    "memset",
    "memcmp",
    "strlen",
    "strcpy",
    "strncpy",
    "wcslen",
    "wcscpy",
    "wcsncpy",

    // Floating-point helpers MSVC emits for /fp:precise -----------------------
    "_fltused",
    "_ftol",
    "_ftol2",
    "_ftol2_sse",
    "_dtol3",
    "_ltod3",
    "_dtoui3",
    "_ultod3",
    "_alldiv",
    "_aulldiv",
    "_allmul",
    "_allrem",
    "_aullrem",
    "_allshl",
    "_allshr",
    "_aullshr",

    // Pure CRT internals that escaped the prefix sieve
    "_tmainCRTStartup",
};

// Substring list. If any of these appears anywhere in the name, reject.
constexpr sv kSubstr[] = {
    "EagleVMBegin",        // marker stubs (not relevant here but cheap)
    "EagleVMEnd",
    "scrt_common_main",    // catches all 4 variants used by MSVC
    "common_main_seh",
    "common_main_dispatch",
    "_initialize_onexit_table",
    "_register_onexit_function",
    "_execute_onexit_table",
    "ExceptionHandler",     // any *ExceptionHandler / __ExceptionHandler
    "FrameHandler",         // any *FrameHandler thunk (Cxx/C/GS)
    "thread_safe_init",
    "Init_thread_",
    // Mangled type_info / typeid descriptors -- their backing data lives in
    // .rdata but MSVC sometimes emits .text helpers prefixed like this.
    "??_R0",                // RTTI Type Descriptor
    "??_R1",                // RTTI Base Class Descriptor
    "??_R2",                // RTTI Base Class Array
    "??_R3",                // RTTI Class Hierarchy Descriptor
    "??_R4",                // RTTI Complete Object Locator
};

bool starts_with(sv s, sv pre) {
    return s.size() >= pre.size() &&
           s.substr(0, pre.size()) == pre;
}

bool contains(sv s, sv needle) {
    return s.find(needle) != sv::npos;
}

}  // namespace

bool is_crt_blacklisted(std::string_view name) noexcept {
    if (name.empty()) return false;
    for (auto e : kExact)  if (name == e)      return true;
    for (auto p : kPrefix) if (starts_with(name, p)) return true;
    for (auto s : kSubstr) if (contains(name, s))    return true;
    return false;
}

}  // namespace wenaxvm::pdb
