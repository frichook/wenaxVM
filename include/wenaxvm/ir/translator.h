#pragma once

// x86 -> IR translator.

#include "wenaxvm/disasm/cfg.h"
#include "wenaxvm/ir/commands.h"

#include <Zydis/Zydis.h>

#include <atomic>
#include <cstdint>

namespace wenaxvm::ir {

// Diagnostic counters incremented every time the CFG-fptr peepholes fire
// inside the lifter. Exposed so the CLI can print a one-line summary after
// batch virtualisation -- a non-zero `dispatch` count is the empirical
// confirmation that we actually rewrote `call [__guard_dispatch_icall_fptr]`
// into the `call rax` IR; a zero count means either the binary has no CFG
// dispatcher idioms (impossible for MSVC /guard:cf builds) or our
// `rip_rel_matches_rva` predicate failed to recognise the slot.
extern std::atomic<std::uint64_t> g_cfg_dispatch_peephole_hits;
extern std::atomic<std::uint64_t> g_cfg_check_peephole_hits;

// Reason a lift bailed out. Valid only when lift() returned an empty block.
struct fail_info {
    ZydisMnemonic mnemonic{ZYDIS_MNEMONIC_INVALID};
    std::uint64_t rva{0};
    enum reason_t : std::uint8_t {
        none,                // not a failure
        undecodable,         // decode_function did not complete
        empty_function,      // function_view had zero instructions
        unsupported_insn,    // mnemonic or operand shape not implemented
    } reason{none};
};

// Lifter configuration. Resolved once per binary; threaded into every lift
// call. Currently used to short-circuit MSVC's Control Flow Guard indirect-call
// idiom so that `call qword ptr [__guard_dispatch_icall_fptr]` is lifted as
// `call rax` (preserving the guest's RAX = real callee target), rather than as
// "load the dispatcher's address into RAX and jmp to it" -- the latter spins
// in the dispatcher's `jmp rax` body forever.
//
// All RVAs default to 0 meaning "not present / not detected"; the lifter
// only special-cases an indirect call/jmp when the resolved [rip+disp] target
// matches a non-zero RVA here.
struct translator_config {
    // RVA of the IAT slot `__guard_check_icall_fptr` (LoadConfig field
    // guard_cf_check_function_ptr). With CFG disabled this slot holds the
    // address of `_guard_check_icall_nop` (a bare `ret`); a `call [slot]`
    // therefore validates RAX and returns without changing it. Virtualising
    // that call as exit_to_top would clobber RAX, so we lift it as a NOP.
    std::uint32_t cfg_check_fptr_rva{0};

    // RVA of the IAT slot `__guard_dispatch_icall_fptr` (LoadConfig field
    // guard_cf_dispatch_function_ptr). The slot holds the address of
    // `_guard_dispatch_icall_nop` whose body is `jmp rax`. The MSVC idiom is:
    //     mov  rax, real_target
    //     call qword ptr [__guard_dispatch_icall_fptr]
    // which is semantically equivalent to `call rax`. We lift it that way.
    std::uint32_t cfg_dispatch_fptr_rva{0};
};

class translator {
 public:
    translator() = default;
    explicit translator(translator_config cfg) noexcept : cfg_(cfg) {}

    void set_config(translator_config cfg) noexcept { cfg_ = cfg; }
    const translator_config& config() const noexcept { return cfg_; }

    // Lift a single linear basic block. Returns empty vector if any
    // instruction is unsupported -- caller should then skip the function.
    block lift(const disasm::basic_block& bb);

    // Lift a whole-function view (multi-block: jcc/jmp, intra-fn labels).
    block lift(const disasm::function_view& fv);

    // Last failure recorded by either lift() overload. Cleared at the start
    // of each call.
    const fail_info& last_failure() const noexcept { return fail_; }

 private:
    fail_info         fail_{};
    translator_config cfg_{};
};

}  // namespace wenaxvm::ir
