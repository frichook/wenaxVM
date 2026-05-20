#pragma once

// Register role manager. Filled in M1.7.
// In M1 the mapping is fixed; shuffle is introduced in M3.

#include "wenaxvm/ir/commands.h"

#include <Zydis/Zydis.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace wenaxvm::vm {

// VM role enum -- ATTN: kept under the original `vsp/vregs/...` member names
// only for backward-compat with the existing call-sites; alias members below
// provide the new neutral names (`stk/ctx/...`) and are gradually replacing the
// old ones. Prefer the neutral names for any new code.
enum class role : std::uint8_t {
    vip = 0,
    vsp,        // alias: stk     -- VM stack pointer
    vregs,      // alias: ctx     -- context/save-area base
    vcs,
    vcsret,
    vbase,      // alias: anchor  -- runtime image-base anchor
    vflags,     // alias: pstat   -- preserved status (RFLAGS stash)
    vtemp0,     // alias: aux0    -- scratch 0
    vtemp1,     // alias: aux1    -- scratch 1
    vtempf,    // dedicated flag-scratch GPR (must be RAX: lahf/sahf use AH)
    count,

    // Neutral aliases for new call-sites. Same numeric value -> same slot.
    stk     = vsp,
    ctx     = vregs,
    anchor  = vbase,
    pstat   = vflags,
    aux0    = vtemp0,
    aux1    = vtemp1,
    auxf    = vtempf,
};

class register_manager {
 public:
    // Default ctor uses the canonical M1 mapping (deterministic, no shuffle).
    register_manager();
    // Per-build shuffle: keeps VTEMPF=RAX and never assigns RSP. Other roles
    // are randomly mapped over {RCX,RDX,RBX,RBP,RSI,RDI,R8..R15}.
    explicit register_manager(std::uint64_t seed);

    // 64-bit physical GPR holding the given role.
    ZydisRegister gpr(role r) const noexcept {
        return mapping_[static_cast<std::size_t>(r)];
    }

    // True if `r64` is currently bound to any VM role.
    bool is_role_reg(ZydisRegister r64) const noexcept;

    // Canonicalise any GPR (eax / ax / al -> rax). Returns r unchanged if
    // not a GPR.
    static ZydisRegister widen_to_64(ZydisRegister r) noexcept;

    // Narrow a 64-bit GPR to the form matching `sz`. rax + bit_32 -> eax.
    static ZydisRegister narrow(ZydisRegister r64, ir::ir_size sz) noexcept;

    // Index of a guest GPR in the VREGS save area. rax->0, rcx->1, ...
    // Returns SIZE_MAX if `r` is not a GPR. The save area is laid out in
    // ZydisRegister GPR64 order: RAX, RCX, RDX, RBX, RSP, RBP, RSI, RDI,
    // R8, R9, R10, R11, R12, R13, R14, R15.
    static std::size_t guest_slot_index(ZydisRegister r) noexcept;

    // Number of guest GPR slots (16) and their byte stride (8).
    static constexpr std::size_t guest_gpr_count() noexcept { return 16; }
    static constexpr std::size_t guest_gpr_stride() noexcept { return 8; }

    // Byte size of the entire guest save area (16 GPRs * 8 bytes).
    static constexpr std::size_t guest_save_area_bytes() noexcept {
        return guest_gpr_count() * guest_gpr_stride();
    }

 private:
    std::array<ZydisRegister, static_cast<std::size_t>(role::count)> mapping_{};
};

}  // namespace wenaxvm::vm
