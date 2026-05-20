#include "wenaxvm/vm/register_manager.h"

#include <algorithm>
#include <array>
#include <random>

namespace wenaxvm::vm {

namespace {

// Canonical guest-GPR slot order inside the per-function context save area.
// Index = slot, value = ZydisRegister. Slot 4 is RSP (special: never restored
// blindly via mov, the exit body switches native rsp through it instead).
constexpr ZydisRegister kGpr64Order[16] = {
    ZYDIS_REGISTER_RAX, ZYDIS_REGISTER_RCX, ZYDIS_REGISTER_RDX, ZYDIS_REGISTER_RBX,
    ZYDIS_REGISTER_RSP, ZYDIS_REGISTER_RBP, ZYDIS_REGISTER_RSI, ZYDIS_REGISTER_RDI,
    ZYDIS_REGISTER_R8,  ZYDIS_REGISTER_R9,  ZYDIS_REGISTER_R10, ZYDIS_REGISTER_R11,
    ZYDIS_REGISTER_R12, ZYDIS_REGISTER_R13, ZYDIS_REGISTER_R14, ZYDIS_REGISTER_R15,
};

}  // namespace

register_manager::register_manager() {
    // Default mapping. Avoids RSP for every role (RSP is the native stack,
    // hands-off for VM logic) and pins the flag-scratch role to RAX because
    // LAHF/SAHF implicitly encode AH. The seeded ctor below randomises the
    // remaining 9 role->GPR assignments.
    mapping_[static_cast<std::size_t>(role::vip)]    = ZYDIS_REGISTER_R15;
    mapping_[static_cast<std::size_t>(role::vsp)]    = ZYDIS_REGISTER_R14;
    mapping_[static_cast<std::size_t>(role::vregs)]  = ZYDIS_REGISTER_R13;
    mapping_[static_cast<std::size_t>(role::vcs)]    = ZYDIS_REGISTER_R12;
    mapping_[static_cast<std::size_t>(role::vcsret)] = ZYDIS_REGISTER_R11;
    mapping_[static_cast<std::size_t>(role::vbase)]  = ZYDIS_REGISTER_R10;
    mapping_[static_cast<std::size_t>(role::vflags)] = ZYDIS_REGISTER_R9;
    mapping_[static_cast<std::size_t>(role::vtemp0)] = ZYDIS_REGISTER_RBX;
    mapping_[static_cast<std::size_t>(role::vtemp1)] = ZYDIS_REGISTER_RCX;
    mapping_[static_cast<std::size_t>(role::vtempf)] = ZYDIS_REGISTER_RAX;  // lahf/sahf use AH
}

register_manager::register_manager(std::uint64_t seed) {
    // VTEMPF is structurally pinned to RAX -- lahf/sahf encode AH implicitly.
    mapping_[static_cast<std::size_t>(role::vtempf)] = ZYDIS_REGISTER_RAX;

    // 14 candidates: all 16 GPRs minus RSP (host stack) and RAX (VTEMPF).
    std::array<ZydisRegister, 14> pool = {
        ZYDIS_REGISTER_RCX, ZYDIS_REGISTER_RDX, ZYDIS_REGISTER_RBX,
        ZYDIS_REGISTER_RBP, ZYDIS_REGISTER_RSI, ZYDIS_REGISTER_RDI,
        ZYDIS_REGISTER_R8,  ZYDIS_REGISTER_R9,  ZYDIS_REGISTER_R10,
        ZYDIS_REGISTER_R11, ZYDIS_REGISTER_R12, ZYDIS_REGISTER_R13,
        ZYDIS_REGISTER_R14, ZYDIS_REGISTER_R15,
    };
    std::mt19937_64 rng(seed);
    std::shuffle(pool.begin(), pool.end(), rng);

    // 9 roles to assign over the 14-candidate pool, in enum order.
    constexpr role to_assign[] = {
        role::vip,    role::vsp,    role::vregs,
        role::vcs,    role::vcsret, role::vbase,
        role::vflags, role::vtemp0, role::vtemp1,
    };
    for (std::size_t i = 0; i < std::size(to_assign); ++i) {
        mapping_[static_cast<std::size_t>(to_assign[i])] = pool[i];
    }
}

bool register_manager::is_role_reg(ZydisRegister r64) const noexcept {
    return std::find(mapping_.begin(), mapping_.end(), r64) != mapping_.end();
}

ZydisRegister register_manager::widen_to_64(ZydisRegister r) noexcept {
    auto cls = ZydisRegisterGetClass(r);
    if (cls != ZYDIS_REGCLASS_GPR8 &&
        cls != ZYDIS_REGCLASS_GPR16 &&
        cls != ZYDIS_REGCLASS_GPR32 &&
        cls != ZYDIS_REGCLASS_GPR64) {
        return r;
    }
    // GPR8 enum ordering interleaves AH/CH/DH/BH at slots 4..7 before
    // SPL/BPL/SIL/DIL/R8B..R15B at 8..19, so ZydisRegisterGetId() returns
    // an enum-relative offset, NOT the x64 register encoding. Translate
    // explicitly via the same REX-aware table used by narrow(), going from
    // GPR8 register -> 4-bit x64 id -> kGpr64Order slot.
    if (cls == ZYDIS_REGCLASS_GPR8) {
        switch (r) {
            case ZYDIS_REGISTER_AL:   return kGpr64Order[0];
            case ZYDIS_REGISTER_CL:   return kGpr64Order[1];
            case ZYDIS_REGISTER_DL:   return kGpr64Order[2];
            case ZYDIS_REGISTER_BL:   return kGpr64Order[3];
            case ZYDIS_REGISTER_AH:   return kGpr64Order[0];
            case ZYDIS_REGISTER_CH:   return kGpr64Order[1];
            case ZYDIS_REGISTER_DH:   return kGpr64Order[2];
            case ZYDIS_REGISTER_BH:   return kGpr64Order[3];
            case ZYDIS_REGISTER_SPL:  return kGpr64Order[4];
            case ZYDIS_REGISTER_BPL:  return kGpr64Order[5];
            case ZYDIS_REGISTER_SIL:  return kGpr64Order[6];
            case ZYDIS_REGISTER_DIL:  return kGpr64Order[7];
            case ZYDIS_REGISTER_R8B:  return kGpr64Order[8];
            case ZYDIS_REGISTER_R9B:  return kGpr64Order[9];
            case ZYDIS_REGISTER_R10B: return kGpr64Order[10];
            case ZYDIS_REGISTER_R11B: return kGpr64Order[11];
            case ZYDIS_REGISTER_R12B: return kGpr64Order[12];
            case ZYDIS_REGISTER_R13B: return kGpr64Order[13];
            case ZYDIS_REGISTER_R14B: return kGpr64Order[14];
            case ZYDIS_REGISTER_R15B: return kGpr64Order[15];
            default:                  return r;
        }
    }
    auto id = ZydisRegisterGetId(r);
    if (id < 0 || id >= 16) return r;
    return kGpr64Order[id];
}

ZydisRegister register_manager::narrow(ZydisRegister r64, ir::ir_size sz) noexcept {
    if (ZydisRegisterGetClass(r64) != ZYDIS_REGCLASS_GPR64) return r64;
    auto id = ZydisRegisterGetId(r64);
    if (id < 0 || id >= 16) return r64;

    // GPR8 enum ordering is NOT linear by x64 register ID: it interleaves
    // AH/CH/DH/BH at slots 4..7 before SPL/BPL/SIL/DIL/R8B..R15B at 8..19.
    // ZydisRegisterEncode(GPR8, n) uses this enum-relative offset, so e.g.
    // n=9 yields BPL not R9B. Use explicit REX-aware tables for sub-32 widths.
    if (sz == ir::ir_size::bit_8) {
        constexpr ZydisRegister rex8[16] = {
            ZYDIS_REGISTER_AL,  ZYDIS_REGISTER_CL,  ZYDIS_REGISTER_DL,  ZYDIS_REGISTER_BL,
            ZYDIS_REGISTER_SPL, ZYDIS_REGISTER_BPL, ZYDIS_REGISTER_SIL, ZYDIS_REGISTER_DIL,
            ZYDIS_REGISTER_R8B, ZYDIS_REGISTER_R9B, ZYDIS_REGISTER_R10B, ZYDIS_REGISTER_R11B,
            ZYDIS_REGISTER_R12B, ZYDIS_REGISTER_R13B, ZYDIS_REGISTER_R14B, ZYDIS_REGISTER_R15B,
        };
        return rex8[id];
    }

    ZydisRegisterClass want;
    switch (sz) {
        case ir::ir_size::bit_16: want = ZYDIS_REGCLASS_GPR16; break;
        case ir::ir_size::bit_32: want = ZYDIS_REGCLASS_GPR32; break;
        case ir::ir_size::bit_64: return r64;
        default: return r64;
    }
    return ZydisRegisterEncode(want, static_cast<std::uint8_t>(id));
}

std::size_t register_manager::guest_slot_index(ZydisRegister r) noexcept {
    auto r64 = widen_to_64(r);
    for (std::size_t i = 0; i < 16; ++i) {
        if (kGpr64Order[i] == r64) return i;
    }
    return static_cast<std::size_t>(-1);
}

}  // namespace wenaxvm::vm
