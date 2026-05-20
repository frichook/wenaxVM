#include "wenaxvm/ir/translator.h"

#include <Zydis/Zydis.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <optional>
#include <stdexcept>
#include <unordered_map>

// Diagnostic counters for the CFG-fptr peepholes -- exposed so the CLI
// can print a summary after lifting. The counters are global because the
// peepholes live in free functions inside an anonymous namespace.
namespace wenaxvm::ir {
std::atomic<std::uint64_t> g_cfg_dispatch_peephole_hits{0};
std::atomic<std::uint64_t> g_cfg_check_peephole_hits{0};
}

namespace wenaxvm::ir {

namespace {

// True iff `r` is one of the 16 architectural GPRs (any width). All other
// register classes -- XMM/YMM/ZMM, FPU ST0..ST7, segment, control/debug,
// MMX, BND, K-mask, RFLAGS/RIP -- cannot be saved into the 16-slot guest
// save area, so the lifter must refuse to push/pop them via cmd_push_reg_ctx.
bool is_gpr_class(ZydisRegister r) {
    switch (ZydisRegisterGetClass(r)) {
        case ZYDIS_REGCLASS_GPR8:
        case ZYDIS_REGCLASS_GPR16:
        case ZYDIS_REGCLASS_GPR32:
        case ZYDIS_REGCLASS_GPR64:
            return true;
        default:
            return false;
    }
}

// Translate Zydis register class to ir_size.
ir_size size_of_reg(ZydisRegister r) {
    switch (ZydisRegisterGetClass(r)) {
        case ZYDIS_REGCLASS_GPR8:  return ir_size::bit_8;
        case ZYDIS_REGCLASS_GPR16: return ir_size::bit_16;
        case ZYDIS_REGCLASS_GPR32: return ir_size::bit_32;
        case ZYDIS_REGCLASS_GPR64: return ir_size::bit_64;
        default:                   return ir_size::bit_64;
    }
}

ir_size size_of_bits(std::uint16_t bits) {
    switch (bits) {
        case 8:  return ir_size::bit_8;
        case 16: return ir_size::bit_16;
        case 32: return ir_size::bit_32;
        case 64: return ir_size::bit_64;
        default: return ir_size::bit_64;
    }
}

// Validate that a MEMORY operand is something the M2 lifter can handle.
// Rejects non-DS segments, unusual scales, and pure-displacement absolute
// addressing.
//   - RSP base: allowed since Phase 3A (shared-stack: VSP == guest RSP).
//   - RIP base: allowed since Phase 3B -- resolved via ZydisCalcAbsoluteAddress
//     to an RVA and materialised through cmd_push_vbase_rva at runtime.
bool ea_supported(const ZydisDecodedOperand& op) {
    if (op.type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    // x64 long mode: SS/DS/ES/CS prefixes have no effect on the address
    // (segment base is always 0). MSVC emits `ss:` on rsp/rbp-relative ops
    // and `ds:` is the default; Zydis surfaces this faithfully. FS/GS still
    // matter (TLS / PEB) and aren't supported yet -- reject those.
    if (op.mem.segment != ZYDIS_REGISTER_NONE &&
        op.mem.segment != ZYDIS_REGISTER_DS   &&
        op.mem.segment != ZYDIS_REGISTER_SS   &&
        op.mem.segment != ZYDIS_REGISTER_ES   &&
        op.mem.segment != ZYDIS_REGISTER_CS) {
        return false;
    }
    if (op.mem.index == ZYDIS_REGISTER_RSP)   return false;  // illegal x64 encoding anyway

    if (op.mem.index != ZYDIS_REGISTER_NONE) {
        std::uint8_t scale = static_cast<std::uint8_t>(op.mem.scale);
        if (scale != 1 && scale != 2 && scale != 4 && scale != 8) return false;
        // VSIB (AVX gather/scatter) encodes XMM/YMM/ZMM as the index register.
        // We materialise EA via cmd_push_reg_ctx which only supports GPRs, so
        // reject anything with a non-GPR index up front.
        if (!is_gpr_class(op.mem.index)) return false;
    }
    if (op.mem.base != ZYDIS_REGISTER_NONE &&
        op.mem.base != ZYDIS_REGISTER_RIP &&
        !is_gpr_class(op.mem.base)) {
        return false;
    }
    // RIP-relative encoding never has a separate index/scale (Zydis surfaces
    // base == RIP with index == NONE), and we reject anything pathological.
    if (op.mem.base == ZYDIS_REGISTER_RIP && op.mem.index != ZYDIS_REGISTER_NONE) {
        return false;
    }
    // At least one register-based component required.
    if (op.mem.base == ZYDIS_REGISTER_NONE && op.mem.index == ZYDIS_REGISTER_NONE) {
        return false;
    }
    return true;
}

// Decomposed effective-address materialisation: emit a sequence
// of IR ops that leave the 64-bit EA on top of VSP. Uses cmd_push_reg_ctx /
// cmd_push_imm / cmd_push_vsp / cmd_push_vbase_rva / cmd_handler_call(add|mul).
//
// `stack_disp` tracks how many bytes VSP has been pushed below the start of
// the current x86 instruction. It's incremented by each net push and
// decremented by each net pop performed during this call (and by the caller
// before/after). When the operand uses RSP as base, the materialised value is
// (VSP + stack_disp_at_first_push) which equals the guest's logical RSP.
//
// `di` provides the enclosing instruction for RIP-relative absolute-address
// computation (ZydisCalcAbsoluteAddress needs runtime_address + length).
//
// The handler_calls here do NOT have an associated cmd_save_flags -- EA
// computation is allowed to clobber native RFLAGS because VFLAGS is the
// source of truth for guest flags.
bool emit_push_ea(block& out,
                  const codec::decoded_instr& di,
                  const ZydisDecodedOperand& op,
                  int& stack_disp) {
    if (!ea_supported(op)) return false;

    // RIP-relative: collapse [rip + disp] to a constant RVA + cmd_push_vbase_rva.
    // ASLR-safe because vbase is loaded with RuntimeImageBase at vm_enter.
    if (op.mem.base == ZYDIS_REGISTER_RIP) {
        std::uint64_t abs_va = 0;
        if (ZYAN_FAILED(ZydisCalcAbsoluteAddress(
                &di.inst, &op, di.runtime_address, &abs_va))) {
            return false;
        }
        // Decoded operands carry an absolute "runtime address" interpreted as
        // an image-relative VA when the input was image_base-anchored. In our
        // pipeline `runtime_address` is set to the function's RVA, so the
        // result of ZydisCalcAbsoluteAddress is already the target RVA.
        out.push_back(std::make_shared<cmd_push_vbase_rva>(abs_va));
        stack_disp += 8;
        return true;
    }

    bool has_base  = op.mem.base  != ZYDIS_REGISTER_NONE;
    bool has_index = op.mem.index != ZYDIS_REGISTER_NONE;
    bool has_disp  = op.mem.disp.has_displacement && op.mem.disp.value != 0;
    int  scale     = has_index ? static_cast<int>(op.mem.scale) : 0;

    // EA is always materialised at 64-bit width (linear address space).
    const ir_size sz = ir_size::bit_64;

    auto push_reg64 = [&](ZydisRegister r) {
        out.push_back(std::make_shared<cmd_push_reg_ctx>(r, sz));
        stack_disp += 8;
    };
    auto push_vsp = [&](std::int32_t adjust) {
        out.push_back(std::make_shared<cmd_push_vsp>(adjust));
        stack_disp += 8;
    };
    auto push_imm64 = [&](std::int64_t v) {
        out.push_back(std::make_shared<cmd_push_imm>(
            static_cast<std::uint64_t>(v), sz));
        stack_disp += 8;
    };
    auto handler_add = [&] {
        out.push_back(std::make_shared<cmd_handler_call>(binop::op_add, sz));
        stack_disp -= 8;
    };
    auto handler_mul = [&] {
        out.push_back(std::make_shared<cmd_handler_call>(binop::op_mul, sz));
        stack_disp -= 8;
    };
    auto push_index_scaled = [&] {
        push_reg64(op.mem.index);
        if (scale > 1) {
            push_imm64(scale);
            handler_mul();
        }
    };

    bool first = true;
    auto combine_with = [&](auto&& push_term) {
        if (first) {
            push_term();
            first = false;
        } else {
            push_term();
            handler_add();
        }
    };

    if (has_base) {
        if (op.mem.base == ZYDIS_REGISTER_RSP) {
            // RSP-relative: materialise guest's logical rsp via VSP + current
            // stack_disp. The base term is always the first push, so the
            // adjust captured here equals stack_disp at the start of this EA.
            int adjust = stack_disp;
            combine_with([&]{ push_vsp(static_cast<std::int32_t>(adjust)); });
        } else {
            combine_with([&]{ push_reg64(op.mem.base); });
        }
    }
    if (has_index) combine_with([&]{ push_index_scaled(); });
    if (has_disp)  combine_with([&]{ push_imm64(op.mem.disp.value); });
    return true;
}

// Push the value of a "source" operand onto VSP.
bool emit_push_src(block& out, const codec::decoded_instr& di,
                   const ZydisDecodedOperand& op, ir_size sz,
                   int& stack_disp) {
    if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        // The 16-slot save-area is GPR-only; XMM/YMM/ZMM/ST/Kn/segment regs
        // can't be persisted there. Reject early so we don't throw deep in
        // the encoder.
        if (!is_gpr_class(op.reg.value)) return false;
        out.push_back(std::make_shared<cmd_push_reg_ctx>(op.reg.value, sz));
        stack_disp += 8;
        return true;
    }
    if (op.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
        // Sign-extended raw immediate value.
        std::uint64_t v = op.imm.is_signed
            ? static_cast<std::uint64_t>(op.imm.value.s)
            : op.imm.value.u;
        out.push_back(std::make_shared<cmd_push_imm>(v, sz));
        stack_disp += 8;
        return true;
    }
    if (op.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        // Decomposed: push EA on top, then cmd_load_mem pops EA + pushes value.
        if (!emit_push_ea(out, di, op, stack_disp)) return false;
        out.push_back(std::make_shared<cmd_load_mem>(sz));
        // load_mem: pop EA (-8), push value (+8) -- net 0.
        return true;
    }
    return false;
}

// Pop the top of VSP into a "destination" operand. For memory dest, the lifter
// must have already deposited the value on VSP; we then push the EA on top
// and cmd_store_mem pops EA + value, writes to [EA].
bool emit_pop_dst(block& out, const codec::decoded_instr& di,
                  const ZydisDecodedOperand& op, ir_size sz,
                  int& stack_disp) {
    if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        if (!is_gpr_class(op.reg.value)) return false;
        out.push_back(std::make_shared<cmd_pop_reg_ctx>(op.reg.value, sz));
        stack_disp -= 8;
        return true;
    }
    if (op.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!emit_push_ea(out, di, op, stack_disp)) return false;
        out.push_back(std::make_shared<cmd_store_mem>(sz));
        stack_disp -= 16;  // pop EA + pop value
        return true;
    }
    return false;
}

// Convenience: emit a cmd_save_flags. The lifter inserts this right after each
// flag-producing IR op so the VFLAGS slot stays in sync with guest x86 flag
// state across handler_calls used for EA arithmetic etc.
void emit_save_flags(block& out) {
    out.push_back(std::make_shared<cmd_save_flags>());
}

bool lift_mov(block& out, const codec::decoded_instr& di) {
    const auto& dst = di.operands[0];
    const auto& src = di.operands[1];
    auto sz = size_of_bits(di.inst.operand_width);
    int stack_disp = 0;
    if (!emit_push_src(out, di, src, sz, stack_disp)) return false;
    if (!emit_pop_dst(out, di, dst, sz, stack_disp))  return false;
    return true;
}

bool lift_binop(block& out, const codec::decoded_instr& di, binop op) {
    const auto& dst = di.operands[0];
    const auto& src = di.operands[1];
    auto sz = size_of_bits(di.inst.operand_width);
    int stack_disp = 0;
    // Push dst (left), push src (right), call handler, save flags, pop result.
    if (!emit_push_src(out, di, dst, sz, stack_disp)) return false;
    if (!emit_push_src(out, di, src, sz, stack_disp)) return false;
    out.push_back(std::make_shared<cmd_handler_call>(op, sz));
    stack_disp -= 8;  // handler_call: pop 2, push 1
    emit_save_flags(out);
    if (!emit_pop_dst(out, di, dst, sz, stack_disp))  return false;
    return true;
}

// LEA reg, [base + index*scale + disp].
// Reuse emit_push_ea: it leaves the 64-bit EA on top of VSP, then we just
// pop into dst at the destination width (truncation matches real LEA semantics
// for 32-bit dst -- top half is zero-extended automatically by cmd_pop_reg_ctx).
// LEA does not touch RFLAGS -- no cmd_save_flags emitted.
bool lift_lea(block& out, const codec::decoded_instr& di) {
    const auto& dst = di.operands[0];
    const auto& memop = di.operands[1];
    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;
    if (!is_gpr_class(dst.reg.value)) return false;
    int stack_disp = 0;
    if (!emit_push_ea(out, di, memop, stack_disp))   return false;
    auto sz = size_of_reg(dst.reg.value);
    // EA was materialised at 64 bits; if dst is narrower the pop will truncate.
    out.push_back(std::make_shared<cmd_pop_reg_ctx>(dst.reg.value, sz));
    return true;
}

bool lift_cmp_test(block& out, const codec::decoded_instr& di, cmp_kind k) {
    const auto& lhs = di.operands[0];
    const auto& rhs = di.operands[1];
    auto sz = size_of_bits(di.inst.operand_width);
    int stack_disp = 0;
    if (!emit_push_src(out, di, lhs, sz, stack_disp)) return false;
    if (!emit_push_src(out, di, rhs, sz, stack_disp)) return false;
    out.push_back(std::make_shared<cmd_cmp_test>(k, sz));
    stack_disp -= 16;  // cmp/test: pop 2, no push
    emit_save_flags(out);
    return true;
}

bool lift_unary(block& out, const codec::decoded_instr& di, unop op) {
    const auto& dst = di.operands[0];
    auto sz = size_of_bits(di.inst.operand_width);
    int stack_disp = 0;
    if (!emit_push_src(out, di, dst, sz, stack_disp)) return false;
    out.push_back(std::make_shared<cmd_unary>(op, sz));
    // unary: pop 1, push 1 -- net 0.
    // NOT and BSWAP do not modify x86 flags; for INC/DEC/NEG we persist VFLAGS.
    if (op != unop::op_not && op != unop::op_bswap) emit_save_flags(out);
    if (!emit_pop_dst(out, di, dst, sz, stack_disp))  return false;
    return true;
}

// 2-op and 3-op imul:
//   imul reg, reg/imm      ->  push dst, push src, mul, save_flags, pop dst
//   imul reg, src, imm     ->  push src, push imm, mul, save_flags, pop dst
bool lift_imul(block& out, const codec::decoded_instr& di) {
    auto sz = size_of_bits(di.inst.operand_width);
    auto count = di.inst.operand_count_visible;
    const auto& dst = di.operands[0];
    int stack_disp = 0;
    if (count == 2) {
        if (!emit_push_src(out, di, dst, sz, stack_disp))           return false;
        if (!emit_push_src(out, di, di.operands[1], sz, stack_disp)) return false;
        out.push_back(std::make_shared<cmd_handler_call>(binop::op_mul, sz));
        stack_disp -= 8;
        emit_save_flags(out);
        if (!emit_pop_dst(out, di, dst, sz, stack_disp))            return false;
        return true;
    }
    if (count == 3) {
        if (!emit_push_src(out, di, di.operands[1], sz, stack_disp)) return false;
        if (!emit_push_src(out, di, di.operands[2], sz, stack_disp)) return false;
        out.push_back(std::make_shared<cmd_handler_call>(binop::op_mul, sz));
        stack_disp -= 8;
        emit_save_flags(out);
        if (!emit_pop_dst(out, di, dst, sz, stack_disp))            return false;
        return true;
    }
    return false;  // 1-op imul -> implicit RDX:RAX, not supported
}

// True if `op` is `[rip+disp]` AND the absolute RVA computed from
// runtime_address+inst-length+disp equals the supplied non-zero `match_rva`.
// Used by lift_call/lift_jmp to short-circuit MSVC CFG dispatcher calls.
bool rip_rel_matches_rva(const codec::decoded_instr& di,
                         const ZydisDecodedOperand&  op,
                         std::uint32_t               match_rva) {
    if (match_rva == 0) return false;
    if (op.type != ZYDIS_OPERAND_TYPE_MEMORY) return false;
    if (op.mem.base != ZYDIS_REGISTER_RIP)    return false;
    if (op.mem.index != ZYDIS_REGISTER_NONE)  return false;
    std::uint64_t abs_va = 0;
    if (ZYAN_FAILED(ZydisCalcAbsoluteAddress(
            &di.inst, &op, di.runtime_address, &abs_va))) {
        return false;
    }
    return abs_va == static_cast<std::uint64_t>(match_rva);
}

// Resolve an in-function branch target rva to an IR label id. External targets
// (MSVC tail-call: `jmp foreign_func`) fall through to cmd_exit_to_rva, which
// emits the vm_exit body + native `jmp rel32`. The CLI patches disp32 once the
// blob's section RVA is known.
bool lift_jmp(block& out,
              const codec::decoded_instr& di,
              std::unordered_map<std::uint64_t, std::uint32_t>& label_for_rva,
              std::uint32_t& /*next_id*/,
              const translator_config& cfg) {
    const auto& op = di.operands[0];

    // Direct rel32 jmp -- intra-function label or external tail-call RVA.
    if (op.type == ZYDIS_OPERAND_TYPE_IMMEDIATE && op.imm.is_relative) {
        std::uint64_t tgt = 0;
        if (ZYAN_FAILED(ZydisCalcAbsoluteAddress(&di.inst, &op, di.runtime_address, &tgt))) {
            return false;
        }
        auto it = label_for_rva.find(tgt);
        if (it != label_for_rva.end()) {
            out.push_back(std::make_shared<cmd_jmp>(it->second));
            return true;
        }
        out.push_back(std::make_shared<cmd_exit_to_rva>(tgt, /*is_call*/ false));
        return true;
    }

    // ---- CFG-dispatcher tail-call peephole ------------------------------
    // `jmp qword ptr [__guard_dispatch_icall_fptr]` is the tail-call analog
    // of the CFG-protected call: the dispatcher's body is `jmp rax`, so the
    // whole instruction is semantically `jmp rax`. Lift it as such -- the
    // guest's RAX already holds the real target.
    if (rip_rel_matches_rva(di, op, cfg.cfg_dispatch_fptr_rva)) {
        g_cfg_dispatch_peephole_hits.fetch_add(1, std::memory_order_relaxed);
        out.push_back(std::make_shared<cmd_push_reg_ctx>(
            ZYDIS_REGISTER_RAX, ir_size::bit_64));
        out.push_back(std::make_shared<cmd_exit_to_top>(/*is_call*/ false));
        return true;
    }
    // `jmp qword ptr [__guard_check_icall_fptr]` is extremely rare (no MSVC
    // idiom emits a tail-jmp through the check pointer), but handle it
    // anyway: the check function is a `ret`, so the tail-jmp is observably
    // equivalent to a function `ret` -- emit vm_exit.
    if (rip_rel_matches_rva(di, op, cfg.cfg_check_fptr_rva)) {
        g_cfg_check_peephole_hits.fetch_add(1, std::memory_order_relaxed);
        out.push_back(std::make_shared<cmd_vm_exit>());
        return true;
    }

    // Indirect jmp through register or memory (e.g. `jmp rax`,
    // `jmp qword ptr [rip+disp]` for jump tables / IAT thunks):
    // materialise target on VSP, then exit VM with `jmp rax` semantics.
    if (op.type == ZYDIS_OPERAND_TYPE_REGISTER ||
        op.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        int stack_disp = 0;
        if (!emit_push_src(out, di, op, ir_size::bit_64, stack_disp)) return false;
        out.push_back(std::make_shared<cmd_exit_to_top>(/*is_call*/ false));
        return true;
    }
    return false;
}

// Map a Jcc mnemonic to its inverted counterpart. Used to lower an external
// `Jcc target` into a skip-around pattern:
//     J!cc skip
//     cmd_exit_to_rva(target, false)
//   skip:
// JCXZ/JECXZ/JRCXZ don't have a single-mnemonic inverse (would need test+jnz),
// so we reject them.
ZydisMnemonic invert_jcc(ZydisMnemonic m) {
    switch (m) {
        case ZYDIS_MNEMONIC_JB:   return ZYDIS_MNEMONIC_JNB;
        case ZYDIS_MNEMONIC_JNB:  return ZYDIS_MNEMONIC_JB;
        case ZYDIS_MNEMONIC_JBE:  return ZYDIS_MNEMONIC_JNBE;
        case ZYDIS_MNEMONIC_JNBE: return ZYDIS_MNEMONIC_JBE;
        case ZYDIS_MNEMONIC_JL:   return ZYDIS_MNEMONIC_JNL;
        case ZYDIS_MNEMONIC_JNL:  return ZYDIS_MNEMONIC_JL;
        case ZYDIS_MNEMONIC_JLE:  return ZYDIS_MNEMONIC_JNLE;
        case ZYDIS_MNEMONIC_JNLE: return ZYDIS_MNEMONIC_JLE;
        case ZYDIS_MNEMONIC_JZ:   return ZYDIS_MNEMONIC_JNZ;
        case ZYDIS_MNEMONIC_JNZ:  return ZYDIS_MNEMONIC_JZ;
        case ZYDIS_MNEMONIC_JS:   return ZYDIS_MNEMONIC_JNS;
        case ZYDIS_MNEMONIC_JNS:  return ZYDIS_MNEMONIC_JS;
        case ZYDIS_MNEMONIC_JO:   return ZYDIS_MNEMONIC_JNO;
        case ZYDIS_MNEMONIC_JNO:  return ZYDIS_MNEMONIC_JO;
        case ZYDIS_MNEMONIC_JP:   return ZYDIS_MNEMONIC_JNP;
        case ZYDIS_MNEMONIC_JNP:  return ZYDIS_MNEMONIC_JP;
        default:                  return ZYDIS_MNEMONIC_INVALID;
    }
}

bool lift_jcc(block& out,
              const codec::decoded_instr& di,
              std::unordered_map<std::uint64_t, std::uint32_t>& label_for_rva,
              std::uint32_t& next_id) {
    const auto& op = di.operands[0];
    if (op.type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;
    if (!op.imm.is_relative) return false;
    std::uint64_t tgt = 0;
    if (ZYAN_FAILED(ZydisCalcAbsoluteAddress(&di.inst, &op, di.runtime_address, &tgt))) {
        return false;
    }
    auto it = label_for_rva.find(tgt);
    if (it != label_for_rva.end()) {
        out.push_back(std::make_shared<cmd_jcc>(di.inst.mnemonic, it->second));
        return true;
    }
    // External target: emit `J!cc skip ; cmd_exit_to_rva(target) ; skip:`.
    ZydisMnemonic inv = invert_jcc(di.inst.mnemonic);
    if (inv == ZYDIS_MNEMONIC_INVALID) return false;
    std::uint32_t skip_id = next_id++;
    out.push_back(std::make_shared<cmd_jcc>(inv, skip_id));
    out.push_back(std::make_shared<cmd_exit_to_rva>(tgt, /*is_call*/ false));
    out.push_back(std::make_shared<cmd_label>(skip_id));
    return true;
}

// CALL rel32: external call only (in-function calls are loops disguised as
// calls and we don't support them yet). Emit:
//   cmd_exit_to_rva(rva, is_call=true)  ; native `call rel32` from blob
//   cmd_vm_enter                        ; re-establish VM context on return
// After the native callee returns, native rsp is at G again, and the new
// vm_enter re-saves all GPRs (including the rax return value) so lifting
// continues against fresh guest state.
bool lift_call(block& out, const codec::decoded_instr& di,
               const translator_config& cfg) {
    const auto& op = di.operands[0];

    // Direct rel32 call.
    if (op.type == ZYDIS_OPERAND_TYPE_IMMEDIATE && op.imm.is_relative) {
        std::uint64_t tgt = 0;
        if (ZYAN_FAILED(ZydisCalcAbsoluteAddress(&di.inst, &op, di.runtime_address, &tgt))) {
            return false;
        }
        out.push_back(std::make_shared<cmd_exit_to_rva>(tgt, /*is_call*/ true));
        out.push_back(std::make_shared<cmd_vm_enter>());
        return true;
    }

    // ---- CFG-dispatcher peephole ----------------------------------------
    // `call qword ptr [__guard_dispatch_icall_fptr]` ≡ `call rax` (the
    // dispatcher's body is `jmp rax`; guest RAX holds the real callee per
    // the MSVC CFG idiom: `mov rax, target ; call [dispatch_fptr]`).
    // Without this peephole the generic indirect-mem path would load RAX
    // with the dispatcher's own address, restore all other guest GPRs, and
    // `call rax`. The dispatcher then `jmp rax`-loops to itself forever.
    if (rip_rel_matches_rva(di, op, cfg.cfg_dispatch_fptr_rva)) {
        g_cfg_dispatch_peephole_hits.fetch_add(1, std::memory_order_relaxed);
        out.push_back(std::make_shared<cmd_push_reg_ctx>(
            ZYDIS_REGISTER_RAX, ir_size::bit_64));
        out.push_back(std::make_shared<cmd_exit_to_top>(/*is_call*/ true));
        out.push_back(std::make_shared<cmd_vm_enter>());
        return true;
    }
    // `call qword ptr [__guard_check_icall_fptr]` is the per-call validator.
    // With CFG disabled the slot points at `_guard_check_icall_nop` (a bare
    // `ret`) and the contract is "preserve all guest regs, including RAX".
    // Lifting it through exit_to_top would clobber guest RAX with the nop's
    // address, breaking the very next `call [dispatch_fptr]`. Drop it.
    if (rip_rel_matches_rva(di, op, cfg.cfg_check_fptr_rva)) {
        g_cfg_check_peephole_hits.fetch_add(1, std::memory_order_relaxed);
        return true;  // emit no IR -- semantically a no-op
    }

    // Indirect call through register or memory (`call rax`, `call qword
    // ptr [rip+disp]` for IAT thunks, `call [rax+8]` for vtables, etc.):
    // push target onto VSP, exit VM with `call rax` semantics, then re-enter.
    if (op.type == ZYDIS_OPERAND_TYPE_REGISTER ||
        op.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        int stack_disp = 0;
        if (!emit_push_src(out, di, op, ir_size::bit_64, stack_disp)) return false;
        out.push_back(std::make_shared<cmd_exit_to_top>(/*is_call*/ true));
        out.push_back(std::make_shared<cmd_vm_enter>());
        return true;
    }
    return false;
}

// cmovcc dst, src  -- 16/32/64. If cond then dst = src; flags untouched.
bool lift_cmov(block& out, const codec::decoded_instr& di) {
    const auto& dst = di.operands[0];
    const auto& src = di.operands[1];
    auto sz = size_of_bits(di.inst.operand_width);
    if (sz == ir_size::bit_8) return false;
    int stack_disp = 0;
    if (!emit_push_src(out, di, dst, sz, stack_disp)) return false;
    if (!emit_push_src(out, di, src, sz, stack_disp)) return false;
    out.push_back(std::make_shared<cmd_cmov>(di.inst.mnemonic, sz));
    stack_disp -= 8;  // cmov: pop 2, push 1
    if (!emit_pop_dst(out, di, dst, sz, stack_disp))  return false;
    return true;
}

// setcc r/m8 -- dst is 8-bit, sets it to 1 if cond else 0.
bool lift_setcc(block& out, const codec::decoded_instr& di) {
    const auto& dst = di.operands[0];
    out.push_back(std::make_shared<cmd_setcc>(di.inst.mnemonic));
    int stack_disp = 8;  // setcc pushes 1 zero-extended byte
    return emit_pop_dst(out, di, dst, ir_size::bit_8, stack_disp);
}

// movzx / movsx / movsxd : extend src (8/16/32) to dst (16/32/64).
// dst is always a register on x86 (no memory-destination form). src may be
// register or memory; for the memory case emit_push_src already routes through
// cmd_load_mem which zero-extends to the 64-bit slot -- safe for both MOVZX
// (we then no-op extend) and MOVSX (we then re-extend from low src_sz bits).
bool lift_movx(block& out, const codec::decoded_instr& di, movx_kind k) {
    const auto& dst = di.operands[0];
    const auto& src = di.operands[1];
    if (dst.type != ZYDIS_OPERAND_TYPE_REGISTER) return false;

    ir_size dst_sz = size_of_reg(dst.reg.value);
    ir_size src_sz;
    if (src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        src_sz = size_of_reg(src.reg.value);
    } else if (src.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        src_sz = size_of_bits(src.size);
    } else {
        return false;
    }

    int stack_disp = 0;
    if (!emit_push_src(out, di, src, src_sz, stack_disp)) return false;
    out.push_back(std::make_shared<cmd_movx>(k, src_sz, dst_sz));
    // movx: pop 1, push 1 -- net 0.
    if (!emit_pop_dst(out, di, dst, dst_sz, stack_disp)) return false;
    return true;
}

// shl/shr/sar with an immediate count operand.
bool lift_shift(block& out, const codec::decoded_instr& di, shift_kind k) {
    const auto& dst = di.operands[0];
    const auto& src = di.operands[1];
    if (src.type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return false;  // CL form in M5
    auto sz = size_of_bits(di.inst.operand_width);
    std::uint8_t count = static_cast<std::uint8_t>(src.imm.value.u & 0x3f);
    if (count == 0) return true;  // SHL/SHR/SAR by 0 is a true no-op
    int stack_disp = 0;
    if (!emit_push_src(out, di, dst, sz, stack_disp)) return false;
    out.push_back(std::make_shared<cmd_shift>(k, sz, count));
    // shift: pop 1, push 1 -- net 0.
    emit_save_flags(out);
    if (!emit_pop_dst(out, di, dst, sz, stack_disp)) return false;
    return true;
}

// Native PUSH/POP. Shared-stack model: VSP IS guest RSP, so cmd_push_reg_ctx /
// cmd_pop_reg_ctx already implement guest push/pop semantics (decrement VSP by
// 8 and write/read [VSP]). For PUSH imm, the immediate is sign-extended to 64
// bits (matching x64 PUSH semantics). For PUSH mem, we materialise the 64-bit
// value via emit_push_src's MEMORY path.
//
// We restrict to 64-bit operand width: 16-bit push (66h prefix) would only
// decrement VSP by 2 and is not used in MSVC-generated function prologues.
// CDQ / CDQE / CWDE / CWD / CBW : implicit-operand sign-extension. Lower them
// to existing IR ops (push reg, movx/shift, pop reg) -- no new IR needed.
//
//   CBW  : AX  = sign_extend(AL)    -- src 8,  dst 16, dst-reg AX
//   CWDE : EAX = sign_extend(AX)    -- src 16, dst 32, dst-reg EAX
//   CDQE : RAX = sign_extend(EAX)   -- src 32, dst 64, dst-reg RAX
//   CWD  : DX  = sign_extend(AX>>15) -- src 16, sar 15, dst-reg DX
//   CDQ  : EDX = sign_extend(EAX>>31)-- src 32, sar 31, dst-reg EDX
//   CQO  : RDX = sign_extend(RAX>>63)-- src 64, sar 63, dst-reg RDX
bool lift_sign_extend_self(block& out, const codec::decoded_instr& di) {
    ZydisRegister src_reg, dst_reg;
    ir_size       src_sz,  dst_sz;
    bool          uses_movx;   // true: movx(sign, src->dst); false: sar by sz-1
    std::uint8_t  sar_count = 0;
    switch (di.inst.mnemonic) {
        case ZYDIS_MNEMONIC_CBW:
            src_reg = ZYDIS_REGISTER_AL;  dst_reg = ZYDIS_REGISTER_AX;
            src_sz  = ir_size::bit_8;     dst_sz  = ir_size::bit_16;
            uses_movx = true; break;
        case ZYDIS_MNEMONIC_CWDE:
            src_reg = ZYDIS_REGISTER_AX;  dst_reg = ZYDIS_REGISTER_EAX;
            src_sz  = ir_size::bit_16;    dst_sz  = ir_size::bit_32;
            uses_movx = true; break;
        case ZYDIS_MNEMONIC_CDQE:
            src_reg = ZYDIS_REGISTER_EAX; dst_reg = ZYDIS_REGISTER_RAX;
            src_sz  = ir_size::bit_32;    dst_sz  = ir_size::bit_64;
            uses_movx = true; break;
        case ZYDIS_MNEMONIC_CWD:
            src_reg = ZYDIS_REGISTER_AX;  dst_reg = ZYDIS_REGISTER_DX;
            src_sz  = ir_size::bit_16;    dst_sz  = ir_size::bit_16;
            uses_movx = false; sar_count = 15; break;
        case ZYDIS_MNEMONIC_CDQ:
            src_reg = ZYDIS_REGISTER_EAX; dst_reg = ZYDIS_REGISTER_EDX;
            src_sz  = ir_size::bit_32;    dst_sz  = ir_size::bit_32;
            uses_movx = false; sar_count = 31; break;
        case ZYDIS_MNEMONIC_CQO:
            src_reg = ZYDIS_REGISTER_RAX; dst_reg = ZYDIS_REGISTER_RDX;
            src_sz  = ir_size::bit_64;    dst_sz  = ir_size::bit_64;
            uses_movx = false; sar_count = 63; break;
        default:
            return false;
    }
    // Push the source value, transform, pop into the destination.
    out.push_back(std::make_shared<cmd_push_reg_ctx>(src_reg, src_sz));
    if (uses_movx) {
        out.push_back(std::make_shared<cmd_movx>(movx_kind::sign, src_sz, dst_sz));
    } else {
        // SAR by (sz-1) replicates the sign bit across all bits of the slot.
        out.push_back(std::make_shared<cmd_shift>(
            shift_kind::sar, src_sz, sar_count));
        // SAR sets flags; emit save_flags to keep VFLAGS in sync.
        emit_save_flags(out);
    }
    out.push_back(std::make_shared<cmd_pop_reg_ctx>(dst_reg, dst_sz));
    return true;
}

// SIMD pass-through. Handles mnemonics whose only "VM-visible" side effect is
// memory access -- the vector register state lives outside the VM save area,
// so we just emit the native instruction inside the blob and (when needed)
// route address computation through emit_push_ea.
//
// Accepted shapes:
//   reg, reg          : `mnem xmmA, xmmB`     -- emit verbatim, no VSP traffic
//   reg, mem          : `mnem xmmA, [ea]`     -- push EA via emit_push_ea
//   mem, reg          : `mnem [ea], xmmA`     -- push EA via emit_push_ea
//
// Width-determining mnemonics (movdqu/movaps/movups/movdqa/xorps/etc.) carry
// 128-bit operands; movss is 32-bit; movsd_xmm is 64-bit; vmov* may be 128
// or 256. We trust di.inst.operand_width to give the correct memory-width.
//
// VEX-encoded forms (vmovdqu/vmovups/vmovupd/vmovdqa/vpxor/...) lifted by the
// same helper -- Zydis encoder picks the right prefix from the mnemonic.
bool lift_xmm_mov(block& out, const codec::decoded_instr& di) {
    if (di.inst.operand_count_visible != 2) return false;
    const auto& dst = di.operands[0];
    const auto& src = di.operands[1];
    // For reg/reg pass-through we accept XMM/YMM/ZMM in either operand.
    auto is_vec = [](ZydisRegister r) {
        auto cls = ZydisRegisterGetClass(r);
        return cls == ZYDIS_REGCLASS_XMM ||
               cls == ZYDIS_REGCLASS_YMM ||
               cls == ZYDIS_REGCLASS_ZMM;
    };

    if (dst.type == ZYDIS_OPERAND_TYPE_REGISTER &&
        src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        if (!is_vec(dst.reg.value) || !is_vec(src.reg.value)) return false;
        out.push_back(std::make_shared<cmd_xmm_op>(
            di.inst.mnemonic, xmm_op_kind::reg_reg,
            dst.reg.value, src.reg.value,
            di.inst.operand_width));
        return true;
    }

    if (dst.type == ZYDIS_OPERAND_TYPE_REGISTER &&
        src.type == ZYDIS_OPERAND_TYPE_MEMORY) {
        if (!is_vec(dst.reg.value)) return false;
        int stack_disp = 0;
        if (!emit_push_ea(out, di, src, stack_disp)) return false;
        out.push_back(std::make_shared<cmd_xmm_op>(
            di.inst.mnemonic, xmm_op_kind::reg_mem,
            dst.reg.value, ZYDIS_REGISTER_NONE,
            src.size /* memory operand width in bits */));
        return true;
    }

    if (dst.type == ZYDIS_OPERAND_TYPE_MEMORY &&
        src.type == ZYDIS_OPERAND_TYPE_REGISTER) {
        if (!is_vec(src.reg.value)) return false;
        int stack_disp = 0;
        if (!emit_push_ea(out, di, dst, stack_disp)) return false;
        out.push_back(std::make_shared<cmd_xmm_op>(
            di.inst.mnemonic, xmm_op_kind::mem_reg,
            ZYDIS_REGISTER_NONE, src.reg.value,
            dst.size));
        return true;
    }
    return false;
}

bool lift_push(block& out, const codec::decoded_instr& di) {
    auto sz = size_of_bits(di.inst.operand_width);
    if (sz != ir_size::bit_64) return false;
    const auto& src = di.operands[0];
    // Reading [rsp+N] mid-PUSH would require capturing stack_disp -- but real
    // code rarely does `push [rsp+8]`. Reject memory sources with RSP base for
    // safety; reg/imm/non-RSP-mem paths are fine.
    if (src.type == ZYDIS_OPERAND_TYPE_MEMORY &&
        src.mem.base == ZYDIS_REGISTER_RSP) return false;
    int stack_disp = 0;
    return emit_push_src(out, di, src, sz, stack_disp);
}

bool lift_pop(block& out, const codec::decoded_instr& di) {
    auto sz = size_of_bits(di.inst.operand_width);
    if (sz != ir_size::bit_64) return false;
    const auto& dst = di.operands[0];
    if (dst.type == ZYDIS_OPERAND_TYPE_MEMORY &&
        dst.mem.base == ZYDIS_REGISTER_RSP) return false;
    int stack_disp = 0;
    return emit_pop_dst(out, di, dst, sz, stack_disp);
}

// True iff the instruction reads or writes any memory operand using
// [rip + disp]. Pass-through re-emits the instruction at a NEW RVA inside the
// .wenax0 section, so a stale disp32 would point at garbage. We reject these
// (option A from the design); option B (re-encode with a patched disp32 via
// post_link_fixup machinery) is future work.
bool has_rip_relative_operand(const codec::decoded_instr& di) {
    for (std::uint8_t i = 0; i < di.inst.operand_count; ++i) {
        const auto& op = di.operands[i];
        if (op.type == ZYDIS_OPERAND_TYPE_MEMORY &&
            op.mem.base == ZYDIS_REGISTER_RIP) {
            return true;
        }
    }
    return false;
}

// Categories of x86 instructions we are willing to wrap in a vm_exit ->
// native -> vm_enter sandwich. We keep this whitelisted on purpose: control
// flow (call/jmp/ret/jcc/syscall/iret) has VM-specific lifting; stack/IO/
// system instructions would either confuse the shared-stack model or fault
// at ring 3.
//
// "true" means: pass-through is acceptable assuming no RIP-relative operand
// (and that's checked separately by has_rip_relative_operand).
bool is_pass_through_safe(const codec::decoded_instr& di) {
    // First reject anything clearly unsafe regardless of category.
    switch (di.inst.mnemonic) {
        // Control flow -- has dedicated lifting paths above.
        case ZYDIS_MNEMONIC_CALL: case ZYDIS_MNEMONIC_JMP: case ZYDIS_MNEMONIC_RET:
        case ZYDIS_MNEMONIC_IRET: case ZYDIS_MNEMONIC_IRETD: case ZYDIS_MNEMONIC_IRETQ:
        // System / privileged.
        case ZYDIS_MNEMONIC_SYSCALL: case ZYDIS_MNEMONIC_SYSENTER:
        case ZYDIS_MNEMONIC_SYSRET:  case ZYDIS_MNEMONIC_SYSEXIT:
        case ZYDIS_MNEMONIC_HLT:
        case ZYDIS_MNEMONIC_IN:  case ZYDIS_MNEMONIC_OUT:
        case ZYDIS_MNEMONIC_INSB: case ZYDIS_MNEMONIC_INSW: case ZYDIS_MNEMONIC_INSD:
        case ZYDIS_MNEMONIC_OUTSB: case ZYDIS_MNEMONIC_OUTSW: case ZYDIS_MNEMONIC_OUTSD:
        case ZYDIS_MNEMONIC_LGDT: case ZYDIS_MNEMONIC_LIDT: case ZYDIS_MNEMONIC_LLDT:
        case ZYDIS_MNEMONIC_SGDT: case ZYDIS_MNEMONIC_SIDT: case ZYDIS_MNEMONIC_SLDT:
        case ZYDIS_MNEMONIC_LTR:  case ZYDIS_MNEMONIC_STR:
        case ZYDIS_MNEMONIC_CLI:  case ZYDIS_MNEMONIC_STI:
        case ZYDIS_MNEMONIC_INVLPG: case ZYDIS_MNEMONIC_INVD: case ZYDIS_MNEMONIC_WBINVD:
        case ZYDIS_MNEMONIC_RDMSR: case ZYDIS_MNEMONIC_WRMSR:
        case ZYDIS_MNEMONIC_CPUID:
            return false;
        // Other Jcc variants -- enumerated in is_jcc().
        case ZYDIS_MNEMONIC_JB: case ZYDIS_MNEMONIC_JBE: case ZYDIS_MNEMONIC_JCXZ:
        case ZYDIS_MNEMONIC_JECXZ: case ZYDIS_MNEMONIC_JKNZD: case ZYDIS_MNEMONIC_JKZD:
        case ZYDIS_MNEMONIC_JL: case ZYDIS_MNEMONIC_JLE: case ZYDIS_MNEMONIC_JNB:
        case ZYDIS_MNEMONIC_JNBE: case ZYDIS_MNEMONIC_JNL: case ZYDIS_MNEMONIC_JNLE:
        case ZYDIS_MNEMONIC_JNO: case ZYDIS_MNEMONIC_JNP: case ZYDIS_MNEMONIC_JNS:
        case ZYDIS_MNEMONIC_JNZ: case ZYDIS_MNEMONIC_JO:  case ZYDIS_MNEMONIC_JP:
        case ZYDIS_MNEMONIC_JRCXZ: case ZYDIS_MNEMONIC_JS: case ZYDIS_MNEMONIC_JZ:
        case ZYDIS_MNEMONIC_LOOP: case ZYDIS_MNEMONIC_LOOPE: case ZYDIS_MNEMONIC_LOOPNE:
            return false;
        // PUSH / POP are stack-aware: VSP is shared with native rsp but the
        // VM has dedicated lifting; pass-through could leave VSP out of sync.
        case ZYDIS_MNEMONIC_PUSH: case ZYDIS_MNEMONIC_POP:
        case ZYDIS_MNEMONIC_PUSHF: case ZYDIS_MNEMONIC_PUSHFD: case ZYDIS_MNEMONIC_PUSHFQ:
        case ZYDIS_MNEMONIC_POPF:  case ZYDIS_MNEMONIC_POPFD:  case ZYDIS_MNEMONIC_POPFQ:
        case ZYDIS_MNEMONIC_ENTER: case ZYDIS_MNEMONIC_LEAVE:
            return false;
        // INT software interrupt: only allow int3 / int 0x29 (fastfail).
        case ZYDIS_MNEMONIC_INT3:
            return true;
        case ZYDIS_MNEMONIC_INT: {
            // int 0x29 -> __fastfail. Anything else is unsafe.
            if (di.inst.operand_count_visible >= 1 &&
                di.operands[0].type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
                std::uint64_t v = di.operands[0].imm.value.u;
                if (v == 0x29) return true;
            }
            return false;
        }
        case ZYDIS_MNEMONIC_INT1: case ZYDIS_MNEMONIC_INTO:
            return false;
        default:
            break;
    }

    // Whitelisted families by ISA-extension. Anything that touches only
    // GPRs / vector regs / flags / memory and isn't on the deny-list above
    // is acceptable for pass-through.
    switch (di.inst.meta.isa_ext) {
        // SSE / SSE2 / SSE3 / SSSE3 / SSE4.x  (movd/movq/movsd/movdqu/...).
        case ZYDIS_ISA_EXT_SSE:
        case ZYDIS_ISA_EXT_SSE2:
        case ZYDIS_ISA_EXT_SSE3:
        case ZYDIS_ISA_EXT_SSSE3:
        case ZYDIS_ISA_EXT_SSE4:
        case ZYDIS_ISA_EXT_SSE4A:
        // AVX / AVX2 / FMA / F16C / AVX-VNNI etc.
        case ZYDIS_ISA_EXT_AVX:
        case ZYDIS_ISA_EXT_AVX2:
        case ZYDIS_ISA_EXT_FMA:
        case ZYDIS_ISA_EXT_FMA4:
        case ZYDIS_ISA_EXT_F16C:
        // Bit manipulation.
        case ZYDIS_ISA_EXT_BMI1:
        case ZYDIS_ISA_EXT_BMI2:
        // CRC32 and related.
        case ZYDIS_ISA_EXT_PCLMULQDQ:
        case ZYDIS_ISA_EXT_AES:
        case ZYDIS_ISA_EXT_SHA:
        case ZYDIS_ISA_EXT_RDRAND:
        case ZYDIS_ISA_EXT_RDSEED:
        // Cache-line / memory ordering -- safe pass-through.
        case ZYDIS_ISA_EXT_CLFLUSHOPT:
        case ZYDIS_ISA_EXT_CLWB:
        case ZYDIS_ISA_EXT_PREFETCHWT1:
            return true;
        default:
            break;
    }

    // Specific base-ISA mnemonics that today fall through to "rejected" but
    // are perfectly safe to wrap in vm_exit/vm_enter:
    //   - DIV / IDIV  (RDX:RAX / EDX:EAX dividend, implicit operand)
    //   - 1-operand IMUL / MUL (implicit RDX:RAX result)
    //   - BT / BTS / BTR / BTC (bit-test family)
    //   - rep-prefixed string ops (movs/stos/lods/cmps/scas)
    //   - atomic xchg / cmpxchg / cmpxchg8b / cmpxchg16b / lock-prefixed RMW
    //   - movbe, popcnt, lzcnt, tzcnt
    //   - bsf / bsr / bswap
    //   - sahf / lahf
    //   - adc / sbb / shld / shrd
    switch (di.inst.mnemonic) {
        case ZYDIS_MNEMONIC_DIV:
        case ZYDIS_MNEMONIC_IDIV:
        case ZYDIS_MNEMONIC_MUL:
        // IMUL handled by lift_imul for 2/3-operand; pass-through covers 1-op.
        case ZYDIS_MNEMONIC_BT:
        case ZYDIS_MNEMONIC_BTS:
        case ZYDIS_MNEMONIC_BTR:
        case ZYDIS_MNEMONIC_BTC:
        case ZYDIS_MNEMONIC_BSF:
        case ZYDIS_MNEMONIC_BSR:
        // BSWAP is now lifted natively (see ZYDIS_MNEMONIC_BSWAP below).
        case ZYDIS_MNEMONIC_MOVBE:
        case ZYDIS_MNEMONIC_POPCNT:
        case ZYDIS_MNEMONIC_LZCNT:
        case ZYDIS_MNEMONIC_TZCNT:
        case ZYDIS_MNEMONIC_SAHF:
        case ZYDIS_MNEMONIC_LAHF:
        case ZYDIS_MNEMONIC_ADC:
        case ZYDIS_MNEMONIC_SBB:
        case ZYDIS_MNEMONIC_SHLD:
        case ZYDIS_MNEMONIC_SHRD:
        case ZYDIS_MNEMONIC_XCHG:
        case ZYDIS_MNEMONIC_XADD:
        case ZYDIS_MNEMONIC_CMPXCHG:
        case ZYDIS_MNEMONIC_CMPXCHG8B:
        case ZYDIS_MNEMONIC_CMPXCHG16B:
        // rep / repe / repne string ops.
        case ZYDIS_MNEMONIC_MOVSB: case ZYDIS_MNEMONIC_MOVSW:
        case ZYDIS_MNEMONIC_MOVSD: case ZYDIS_MNEMONIC_MOVSQ:
        case ZYDIS_MNEMONIC_STOSB: case ZYDIS_MNEMONIC_STOSW:
        case ZYDIS_MNEMONIC_STOSD: case ZYDIS_MNEMONIC_STOSQ:
        case ZYDIS_MNEMONIC_LODSB: case ZYDIS_MNEMONIC_LODSW:
        case ZYDIS_MNEMONIC_LODSD: case ZYDIS_MNEMONIC_LODSQ:
        case ZYDIS_MNEMONIC_CMPSB: case ZYDIS_MNEMONIC_CMPSW:
        case ZYDIS_MNEMONIC_CMPSD: case ZYDIS_MNEMONIC_CMPSQ:
        case ZYDIS_MNEMONIC_SCASB: case ZYDIS_MNEMONIC_SCASW:
        case ZYDIS_MNEMONIC_SCASD: case ZYDIS_MNEMONIC_SCASQ:
        // Misc safe ALU/flag ops not yet lifted.
        case ZYDIS_MNEMONIC_CLC: case ZYDIS_MNEMONIC_STC: case ZYDIS_MNEMONIC_CMC:
        case ZYDIS_MNEMONIC_CLD: case ZYDIS_MNEMONIC_STD:
        case ZYDIS_MNEMONIC_PREFETCHNTA: case ZYDIS_MNEMONIC_PREFETCH:
            return true;
        default:
            return false;
    }
}

// Attempt raw-bytes pass-through: rewrite the instruction as
// vm_exit_body -> raw original bytes -> pushfq -> vm_enter -> stash rflags.
// Returns false (and emits nothing) if pass-through isn't safe for this insn.
bool lift_pass_through(block& /*out*/, const codec::decoded_instr& /*di*/) {
    // Pass-through (cmd_x86_exec sandwich) is currently disabled: the smoke
    // test of viral.vm.exe regressed -- the vm_exit -> raw bytes -> vm_enter
    // path is not yet correct end-to-end (suspected RDI/RCX/RAX corruption
    // around `rep stosd` sites). Re-enable only after the sandwich has been
    // root-caused and a per-instruction smoke-test passes.
    return false;
}

bool lift_one(block& out, const codec::decoded_instr& di,
              const translator_config& cfg) {
    // CMOVcc family is large -- handle by mnemonic prefix check.
    switch (di.inst.mnemonic) {
        case ZYDIS_MNEMONIC_CMOVB:  case ZYDIS_MNEMONIC_CMOVBE:
        case ZYDIS_MNEMONIC_CMOVL:  case ZYDIS_MNEMONIC_CMOVLE:
        case ZYDIS_MNEMONIC_CMOVNB: case ZYDIS_MNEMONIC_CMOVNBE:
        case ZYDIS_MNEMONIC_CMOVNL: case ZYDIS_MNEMONIC_CMOVNLE:
        case ZYDIS_MNEMONIC_CMOVNO: case ZYDIS_MNEMONIC_CMOVNP:
        case ZYDIS_MNEMONIC_CMOVNS: case ZYDIS_MNEMONIC_CMOVNZ:
        case ZYDIS_MNEMONIC_CMOVO:  case ZYDIS_MNEMONIC_CMOVP:
        case ZYDIS_MNEMONIC_CMOVS:  case ZYDIS_MNEMONIC_CMOVZ:
            return lift_cmov(out, di);
        case ZYDIS_MNEMONIC_SETB:   case ZYDIS_MNEMONIC_SETBE:
        case ZYDIS_MNEMONIC_SETL:   case ZYDIS_MNEMONIC_SETLE:
        case ZYDIS_MNEMONIC_SETNB:  case ZYDIS_MNEMONIC_SETNBE:
        case ZYDIS_MNEMONIC_SETNL:  case ZYDIS_MNEMONIC_SETNLE:
        case ZYDIS_MNEMONIC_SETNO:  case ZYDIS_MNEMONIC_SETNP:
        case ZYDIS_MNEMONIC_SETNS:  case ZYDIS_MNEMONIC_SETNZ:
        case ZYDIS_MNEMONIC_SETO:   case ZYDIS_MNEMONIC_SETP:
        case ZYDIS_MNEMONIC_SETS:   case ZYDIS_MNEMONIC_SETZ:
            return lift_setcc(out, di);
        default:
            break;
    }

    switch (di.inst.mnemonic) {
        case ZYDIS_MNEMONIC_MOV:  return lift_mov(out, di);
        case ZYDIS_MNEMONIC_ADD:  return lift_binop(out, di, binop::op_add);
        case ZYDIS_MNEMONIC_SUB:  return lift_binop(out, di, binop::op_sub);
        case ZYDIS_MNEMONIC_XOR:  return lift_binop(out, di, binop::op_xor);
        case ZYDIS_MNEMONIC_AND:  return lift_binop(out, di, binop::op_and);
        case ZYDIS_MNEMONIC_OR:   return lift_binop(out, di, binop::op_or);
        case ZYDIS_MNEMONIC_LEA:  return lift_lea(out, di);
        case ZYDIS_MNEMONIC_CMP:  return lift_cmp_test(out, di, cmp_kind::op_cmp);
        case ZYDIS_MNEMONIC_TEST: return lift_cmp_test(out, di, cmp_kind::op_test);
        case ZYDIS_MNEMONIC_INC:  return lift_unary(out, di, unop::op_inc);
        case ZYDIS_MNEMONIC_DEC:  return lift_unary(out, di, unop::op_dec);
        case ZYDIS_MNEMONIC_NEG:  return lift_unary(out, di, unop::op_neg);
        case ZYDIS_MNEMONIC_NOT:  return lift_unary(out, di, unop::op_not);
        case ZYDIS_MNEMONIC_BSWAP: {
            // BSWAP is only architecturally defined for r32/r64. Reject the
            // r16 form (its behaviour is undefined per the Intel SDM, and the
            // emit path narrows to r32 anyway).
            auto bw = di.inst.operand_width;
            if (bw != 32 && bw != 64) return false;
            return lift_unary(out, di, unop::op_bswap);
        }
        case ZYDIS_MNEMONIC_IMUL:   return lift_imul(out, di);
        case ZYDIS_MNEMONIC_MOVZX:  return lift_movx(out, di, movx_kind::zero);
        case ZYDIS_MNEMONIC_MOVSX:  return lift_movx(out, di, movx_kind::sign);
        case ZYDIS_MNEMONIC_MOVSXD: return lift_movx(out, di, movx_kind::sign);
        case ZYDIS_MNEMONIC_SHL:    return lift_shift(out, di, shift_kind::shl);
        case ZYDIS_MNEMONIC_SHR:    return lift_shift(out, di, shift_kind::shr);
        case ZYDIS_MNEMONIC_SAR:    return lift_shift(out, di, shift_kind::sar);
        case ZYDIS_MNEMONIC_ROL:    return lift_shift(out, di, shift_kind::rol);
        case ZYDIS_MNEMONIC_ROR:    return lift_shift(out, di, shift_kind::ror);
        case ZYDIS_MNEMONIC_PUSH:   return lift_push(out, di);
        case ZYDIS_MNEMONIC_POP:    return lift_pop(out, di);
        case ZYDIS_MNEMONIC_CALL:   return lift_call(out, di, cfg);
        case ZYDIS_MNEMONIC_CBW:    case ZYDIS_MNEMONIC_CWDE:
        case ZYDIS_MNEMONIC_CDQE:   case ZYDIS_MNEMONIC_CWD:
        case ZYDIS_MNEMONIC_CDQ:    case ZYDIS_MNEMONIC_CQO:
            return lift_sign_extend_self(out, di);
        case ZYDIS_MNEMONIC_NOP:    return true;  // no IR emit; pure padding
        case ZYDIS_MNEMONIC_VZEROUPPER:
        case ZYDIS_MNEMONIC_VZEROALL:
            // We don't model YMM/ZMM upper halves; the lower XMM lanes are
            // unaffected, so for our shared-stack VM these are observably NOPs.
            return true;
        // ---- SIMD pass-through ---------------------------------------------
        // Vector mov-class (reg/reg, reg/mem, mem/reg).
        case ZYDIS_MNEMONIC_MOVAPS:
        case ZYDIS_MNEMONIC_MOVAPD:
        case ZYDIS_MNEMONIC_MOVUPS:
        case ZYDIS_MNEMONIC_MOVUPD:
        case ZYDIS_MNEMONIC_MOVDQA:
        case ZYDIS_MNEMONIC_MOVDQU:
        case ZYDIS_MNEMONIC_MOVSS:
        case ZYDIS_MNEMONIC_MOVSD:
        case ZYDIS_MNEMONIC_MOVD:
        case ZYDIS_MNEMONIC_MOVQ:
        case ZYDIS_MNEMONIC_VMOVAPS:
        case ZYDIS_MNEMONIC_VMOVAPD:
        case ZYDIS_MNEMONIC_VMOVUPS:
        case ZYDIS_MNEMONIC_VMOVUPD:
        case ZYDIS_MNEMONIC_VMOVDQA:
        case ZYDIS_MNEMONIC_VMOVDQU:
        case ZYDIS_MNEMONIC_VMOVSS:
        case ZYDIS_MNEMONIC_VMOVSD:
        case ZYDIS_MNEMONIC_VMOVD:
        case ZYDIS_MNEMONIC_VMOVQ:
        // Reg/reg & reg/mem ALU-style ops that touch only vector regs.
        case ZYDIS_MNEMONIC_XORPS:
        case ZYDIS_MNEMONIC_XORPD:
        case ZYDIS_MNEMONIC_ANDPS:
        case ZYDIS_MNEMONIC_ANDPD:
        case ZYDIS_MNEMONIC_ORPS:
        case ZYDIS_MNEMONIC_ORPD:
        case ZYDIS_MNEMONIC_PXOR:
        case ZYDIS_MNEMONIC_PAND:
        case ZYDIS_MNEMONIC_POR:
        case ZYDIS_MNEMONIC_VXORPS:
        case ZYDIS_MNEMONIC_VXORPD:
        case ZYDIS_MNEMONIC_VPXOR: {
            // Try the lightweight xmm_op path first (no vm_exit/enter cycle).
            block tmp;
            if (lift_xmm_mov(tmp, di)) {
                out.insert(out.end(), tmp.begin(), tmp.end());
                return true;
            }
            // Fallback: raw pass-through (vm_exit -> native bytes -> vm_enter).
            return lift_pass_through(out, di);
        }
        case ZYDIS_MNEMONIC_RET:
            out.push_back(std::make_shared<cmd_vm_exit>());
            return true;
        default:
            // Fall through to raw-bytes pass-through for whitelisted
            // SSE/AVX/atomic/bit-test/string/div/etc. instructions.
            return lift_pass_through(out, di);
    }
}

// Categorise a Zydis mnemonic as a conditional branch (Jcc family).
bool is_jcc(ZydisMnemonic m) {
    switch (m) {
        case ZYDIS_MNEMONIC_JB: case ZYDIS_MNEMONIC_JBE: case ZYDIS_MNEMONIC_JCXZ:
        case ZYDIS_MNEMONIC_JECXZ: case ZYDIS_MNEMONIC_JKNZD: case ZYDIS_MNEMONIC_JKZD:
        case ZYDIS_MNEMONIC_JL: case ZYDIS_MNEMONIC_JLE: case ZYDIS_MNEMONIC_JNB:
        case ZYDIS_MNEMONIC_JNBE: case ZYDIS_MNEMONIC_JNL: case ZYDIS_MNEMONIC_JNLE:
        case ZYDIS_MNEMONIC_JNO: case ZYDIS_MNEMONIC_JNP: case ZYDIS_MNEMONIC_JNS:
        case ZYDIS_MNEMONIC_JNZ: case ZYDIS_MNEMONIC_JO:  case ZYDIS_MNEMONIC_JP:
        case ZYDIS_MNEMONIC_JRCXZ: case ZYDIS_MNEMONIC_JS: case ZYDIS_MNEMONIC_JZ:
            return true;
        default:
            return false;
    }
}

}  // namespace

block translator::lift(const disasm::basic_block& bb) {
    fail_ = {};
    block out;
    out.push_back(std::make_shared<cmd_vm_enter>());

    for (const auto& di : bb.instructions) {
        block tmp;
        if (!lift_one(tmp, di, cfg_)) {
            fail_ = {di.inst.mnemonic, di.runtime_address,
                     fail_info::unsupported_insn};
            return {};
        }
        out.insert(out.end(), tmp.begin(), tmp.end());
    }

    // If the block didn't end with a ret/jmp we still want a sane exit.
    if (out.empty() || out.back()->kind() != command_kind::vm_exit) {
        out.push_back(std::make_shared<cmd_vm_exit>());
    }
    return out;
}

block translator::lift(const disasm::function_view& fv) {
    fail_ = {};
    if (!fv.complete) {
        fail_ = {ZYDIS_MNEMONIC_INVALID, fv.start_rva, fail_info::undecodable};
        return {};
    }
    if (fv.instructions.empty()) {
        fail_ = {ZYDIS_MNEMONIC_INVALID, fv.start_rva, fail_info::empty_function};
        return {};
    }

    block out;
    out.push_back(std::make_shared<cmd_vm_enter>());

    // Pre-allocate IR label ids for every branch target rva.
    std::unordered_map<std::uint64_t, std::uint32_t> label_for_rva;
    std::uint32_t next_id = 0;
    for (auto rva : fv.branch_targets) {
        label_for_rva.emplace(rva, next_id++);
    }

    for (const auto& di : fv.instructions) {
        // Drop in an IR label if this instruction is targeted by a branch.
        auto it = label_for_rva.find(di.runtime_address);
        if (it != label_for_rva.end()) {
            out.push_back(std::make_shared<cmd_label>(it->second));
        }

        block tmp;
        bool ok;
        // Branches: lifted into cmd_jmp / cmd_jcc against the label map.
        if (di.inst.mnemonic == ZYDIS_MNEMONIC_JMP) {
            ok = lift_jmp(tmp, di, label_for_rva, next_id, cfg_);
        } else if (is_jcc(di.inst.mnemonic)) {
            ok = lift_jcc(tmp, di, label_for_rva, next_id);
        } else {
            ok = lift_one(tmp, di, cfg_);
        }
        if (!ok) {
            fail_ = {di.inst.mnemonic, di.runtime_address,
                     fail_info::unsupported_insn};
            return {};
        }
        out.insert(out.end(), tmp.begin(), tmp.end());
    }

    if (out.empty() || out.back()->kind() != command_kind::vm_exit) {
        out.push_back(std::make_shared<cmd_vm_exit>());
    }
    return out;
}

}  // namespace wenaxvm::ir
