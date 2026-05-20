#include "wenaxvm/vm/machine.h"

#include <Zydis/Zydis.h>
#include <stdexcept>
#include <string>

namespace wenaxvm::vm {

using codec::encode_builder;
using codec::reg;
using codec::imm;
using codec::mem;

namespace {

// Map ir_size -> mem operand bit-width.
std::uint16_t mem_bits(ir::ir_size sz) {
    return static_cast<std::uint16_t>(static_cast<std::uint8_t>(sz) * 8);
}

// imm operand byte-count derived from ir_size.
std::uint8_t imm_bytes(ir::ir_size sz) {
    return static_cast<std::uint8_t>(sz);
}

// Order of GPRs inside the save area. RSP slot (index 4) is unused.
constexpr ZydisRegister kGpr64Order[16] = {
    ZYDIS_REGISTER_RAX, ZYDIS_REGISTER_RCX, ZYDIS_REGISTER_RDX, ZYDIS_REGISTER_RBX,
    ZYDIS_REGISTER_RSP, ZYDIS_REGISTER_RBP, ZYDIS_REGISTER_RSI, ZYDIS_REGISTER_RDI,
    ZYDIS_REGISTER_R8,  ZYDIS_REGISTER_R9,  ZYDIS_REGISTER_R10, ZYDIS_REGISTER_R11,
    ZYDIS_REGISTER_R12, ZYDIS_REGISTER_R13, ZYDIS_REGISTER_R14, ZYDIS_REGISTER_R15,
};

// Offset (relative to native rsp after `sub rsp, total_alloc`) of save-area slot.
constexpr std::int64_t save_off(std::size_t slot) {
    return static_cast<std::int64_t>(machine::save_area_off + slot * 8);
}

// VFLAGS lives one qword below the save area, i.e. at [VREGS - 8].
constexpr std::int64_t vflags_off() {
    return -8;
}

}  // namespace

machine::machine(register_manager regs) : regs_(regs) {}

// -----------------------------------------------------------------------------
// Flag stash helpers (avoid pushfq/rsp manipulation).
//
// Save:   seto al ; lahf       -> rax low 16 bits = [ah=flags | al=OF]
//         mov word [VREGS+vflags_off], ax
// Load:   movzx eax, word [VREGS+vflags_off]
//         add al, 0x7f         -> sets OF iff old al was 1
//         sahf                 -> restores ZF/SF/AF/PF/CF from AH
// -----------------------------------------------------------------------------
void machine::emit_save_flags(encode_builder& eb) {
    auto vregs = regs_.gpr(role::vregs);
    // VTEMPF must be RAX for lahf/sahf semantics.
    eb.make(ZYDIS_MNEMONIC_SETO, reg(ZYDIS_REGISTER_AL));
    eb.make(ZYDIS_MNEMONIC_LAHF);
    eb.make(ZYDIS_MNEMONIC_MOV,
            mem(vregs, vflags_off(), 16),
            reg(ZYDIS_REGISTER_AX));
}

void machine::emit_load_flags(encode_builder& eb) {
    auto vregs = regs_.gpr(role::vregs);
    eb.make(ZYDIS_MNEMONIC_MOVZX,
            reg(ZYDIS_REGISTER_EAX),
            mem(vregs, vflags_off(), 16));
    // Reconstruct OF: add al, 0x7f.  Was 0 -> result 0x7f, OF=0.
    //                                Was 1 -> result 0x80, OF=1.
    eb.make(ZYDIS_MNEMONIC_ADD, reg(ZYDIS_REGISTER_AL), imm(0x7f, 1));
    eb.make(ZYDIS_MNEMONIC_SAHF);
}

codec::label_id machine::resolve_label(encode_builder& eb, std::uint32_t ir_id) {
    if (ir_id >= ir_label_map_.size()) {
        ir_label_map_.resize(ir_id + 1, codec::label_id{static_cast<std::size_t>(-1)});
    }
    auto& slot = ir_label_map_[ir_id];
    if (slot.value == static_cast<std::size_t>(-1)) {
        slot = eb.make_label();
    }
    return slot;
}

// -----------------------------------------------------------------------------
// Emit `lea vbase, [rip + 0]` and record a post-link fixup so the disp32 is
// patched to make vbase = RuntimeImageBase (= the actual VA at which the PE
// loaded, after ASLR rebasing).
//
// At runtime, the LEA computes:    vbase = RIP_after_lea + sign_ext(disp32)
//                                        = (RuntimeImageBase + lea_rva + 7)  + disp
// We want vbase = RuntimeImageBase, so:
//                              disp = -(lea_rva + 7)
//                                   = -(section_va + lea_offset_in_section + 7)
// where (lea_offset_in_section + 7) == section-relative offset of the byte
// just after the LEA, i.e. `next_pc`. The CLI patches the slot with
//                       disp32 = -(section_va + next_pc) = 0 - (section_va + next_pc)
// which matches the generic formula `target_rva - (section_va + next_pc)`
// for `target_rva = 0`.
void machine::emit_load_image_base(encode_builder& eb) {
    auto vbase = regs_.gpr(role::vbase);
    eb.make(ZYDIS_MNEMONIC_LEA,
            reg(vbase),
            mem(ZYDIS_REGISTER_RIP, 0, 64));
    std::size_t next_pc = eb.cursor();
    pending_fixups_.push_back(post_link_fixup{
        /*patch_at*/ next_pc - 4,
        /*next_pc */ next_pc,
        /*target  */ 0,
    });
}

// -----------------------------------------------------------------------------
// VEnter (shared-stack model: VSP rides on the native RSP region)
// -----------------------------------------------------------------------------
// Native rsp at entry is G == guest's RSP at original function entry.
// After `sub rsp, total_alloc` (1160 bytes, 8-mod-16):
//   [rsp + 0)              - VFLAGS slot
//   [rsp + 8 .. + 136)     - save area (slot i at +8+i*8; slot 4 holds G)
//   [rsp + 136 .. + 1160)  - VM operand-stack headroom
//   VREGS = rsp + 8 ; VSP = rsp + 1160 (= G).
// Slot 4 (RSP) stores G so vm_exit can switch native rsp via a single MOV.
//
// We also load VBASE = RuntimeImageBase here so subsequent IR ops can
// materialise absolute VAs from RVAs (RIP-relative operand support).
void machine::emit_vm_enter(encode_builder& eb) {
    auto vsp   = regs_.gpr(role::vsp);
    auto vregs = regs_.gpr(role::vregs);

    // Allocate frame; rsp now 16-byte aligned.
    eb.make(ZYDIS_MNEMONIC_SUB,
            reg(ZYDIS_REGISTER_RSP),
            imm(machine::total_alloc));

    // Save 15 guest GPRs into [rsp + save_off(i)]. Slot 4 (RSP) handled below.
    // Saving happens BEFORE loading vbase so the physical reg currently
    // playing the vbase role is preserved in its save-area slot before we
    // clobber it with the image-base LEA.
    for (std::size_t i = 0; i < 16; ++i) {
        if (i == 4) continue;
        eb.make(ZYDIS_MNEMONIC_MOV,
                mem(ZYDIS_REGISTER_RSP, save_off(i), 64),
                reg(kGpr64Order[i]));
    }

    // VREGS = rsp + save_area_off (slot 0 base).
    eb.make(ZYDIS_MNEMONIC_LEA,
            reg(vregs),
            mem(ZYDIS_REGISTER_RSP, static_cast<std::int64_t>(machine::save_area_off), 64));

    // VSP = rsp + total_alloc == G (guest RSP at function entry).
    eb.make(ZYDIS_MNEMONIC_LEA,
            reg(vsp),
            mem(ZYDIS_REGISTER_RSP, static_cast<std::int64_t>(machine::total_alloc), 64));

    // Store G into slot 4 (RSP). vm_exit later writes current_VSP here, then
    // restores native rsp from this slot to land at guest's exit-time rsp.
    eb.make(ZYDIS_MNEMONIC_MOV,
            mem(vregs, 4 * 8, 64),
            reg(vsp));

    // Load VBASE = RuntimeImageBase via RIP-relative LEA + post-link fixup.
    emit_load_image_base(eb);
}

// -----------------------------------------------------------------------------
// VExit body (shared-stack tear-down) -- the work shared by `ret`-style
// vm_exit and external `jmp/call rel32`-style vm_exit_to_rva. Does NOT emit
// the final transfer-of-control instruction; the caller appends `ret` or the
// raw native rel32.
//
// On entry:
//   - VSP holds the guest's current logical RSP (for plain vm_exit) or the
//     pre-call/jmp guest RSP (for vm_exit_to_rva: the lifter has already
//     pushed the native return address etc. as needed via cmd_push_imm).
//   - VREGS points at slot 0 of the save area.
//   - Native rsp is at G - total_alloc (the allocated VM frame).
//
// Procedure:
//   1) Overwrite slot 4 (RSP) with current VSP -- this is the value native rsp
//      must hold after restore so any final `ret` pops from [rsp] = G.
//   2) Copy VREGS into VTEMP0 so the save-area pointer survives even after we
//      restore the physical reg that VREGS is mapped to.
//   3) For i in 0..15, skipping the slot that maps to VTEMP0's physical reg:
//        - i == 4  -> `mov rsp, [VTEMP0 + 32]` (switches native rsp to guest's
//                     exit-time rsp; subsequent ops use VTEMP0, not rsp, for
//                     addressing so they remain valid).
//        - else    -> `mov R_i, [VTEMP0 + i*8]` (restores guest GPR).
//   4) Restore VTEMP0_phys last: `mov VTEMP0, [VTEMP0 + vtemp0_slot*8]`. The
//      read uses the old base, the write installs the original guest value.
void machine::emit_vm_exit_body(encode_builder& eb) {
    auto vsp_phys    = regs_.gpr(role::vsp);
    auto vregs_phys  = regs_.gpr(role::vregs);
    auto vtemp0_phys = regs_.gpr(role::vtemp0);

    const std::size_t vtemp0_slot =
        register_manager::guest_slot_index(vtemp0_phys);

    // 1) Stash current_VSP into slot 4 (so the i==4 load below sets native rsp).
    eb.make(ZYDIS_MNEMONIC_MOV,
            mem(vregs_phys, 4 * 8, 64),
            reg(vsp_phys));

    // 2) VTEMP0 = VREGS (save-area pointer). Survives clobber of vregs_phys.
    eb.make(ZYDIS_MNEMONIC_MOV, reg(vtemp0_phys), reg(vregs_phys));

    // 3) Restore each GPR through VTEMP0, deferring VTEMP0's own slot.
    for (std::size_t i = 0; i < 16; ++i) {
        if (i == vtemp0_slot) continue;
        if (i == 4) {
            // Switch native rsp to current_VSP (= guest's exit-time rsp).
            eb.make(ZYDIS_MNEMONIC_MOV,
                    reg(ZYDIS_REGISTER_RSP),
                    mem(vtemp0_phys, static_cast<std::int64_t>(i * 8), 64));
        } else {
            eb.make(ZYDIS_MNEMONIC_MOV,
                    reg(kGpr64Order[i]),
                    mem(vtemp0_phys, static_cast<std::int64_t>(i * 8), 64));
        }
    }

    // 4) Restore VTEMP0_phys last (read-then-write of the same physical reg).
    eb.make(ZYDIS_MNEMONIC_MOV,
            reg(vtemp0_phys),
            mem(vtemp0_phys, static_cast<std::int64_t>(vtemp0_slot * 8), 64));
}

// vm_exit: body + native `ret`. Pops return address from [rsp] which sits at
// the guest's exit-time rsp.
void machine::emit_vm_exit(encode_builder& eb) {
    emit_vm_exit_body(eb);
    eb.make(ZYDIS_MNEMONIC_RET);
}

// -----------------------------------------------------------------------------
// exit_to_rva : emit_vm_exit_body + raw native `jmp rel32` (E9) or `call rel32`
// (E8) into an external RVA. The disp32 is left as 4 zero bytes and a
// post_link_fixup is recorded so the CLI can patch it once the section RVA
// is finalised.
//
// disp32 formula (matches lea-image-base case):
//     disp32 = target_rva - (section_va + next_pc)
//
// For is_call=true, the caller (lifter) must follow this command with a
// cmd_vm_enter so that, when the native callee returns, VBASE/VSP/VREGS are
// re-established and lifting can continue with the post-call register state
// (rax now holds the return value etc.).
// -----------------------------------------------------------------------------
void machine::emit_exit_to_rva(encode_builder& eb, const ir::cmd_exit_to_rva& c) {
    emit_vm_exit_body(eb);

    // Raw encoding of `jmp rel32` (E9 disp32) or `call rel32` (E8 disp32).
    std::uint8_t opcode = c.is_call() ? 0xE8 : 0xE9;
    std::uint8_t payload[5] = {opcode, 0, 0, 0, 0};
    eb.append_bytes(payload, sizeof(payload));

    std::size_t next_pc  = eb.cursor();
    std::size_t patch_at = next_pc - 4;
    pending_fixups_.push_back(post_link_fixup{
        /*patch_at*/ patch_at,
        /*next_pc */ next_pc,
        /*target  */ c.rva(),
    });
}

// -----------------------------------------------------------------------------
// exit_to_top : indirect call/jmp where the target sits on top of VSP.
//
// The naive lowering (stash target into RAX save slot, vm_exit_body, `jmp rax`)
// is unsafe: it permanently clobbers guest RAX with the target. That breaks the
// MSVC CFG idiom `mov rax, real_target ; call [__guard_dispatch_icall_fptr]`
// where the dispatcher thunk is literally `jmp rax` -- if we wrote the fptr
// into RAX_slot then on return into the dispatcher RAX would still be the
// dispatcher's own address and we'd `jmp rax` to ourselves forever.
//
// Instead we use the "push target on guest stack + native ret" trick,
// which never touches any guest GPR save slot:
//
//   jmp form:
//     1) VTEMP0 = [VSP] ; VSP += 8           ; pop target
//     2) VSP -= 8 ; [VSP] = VTEMP0           ; re-push so it sits at top
//                                              once native rsp gets switched
//     3) emit_vm_exit_body                   ; restores ALL guest GPRs
//                                              (incl. VTEMP0_phys) and sets
//                                              native rsp := current VSP
//     4) ret                                 ; pops target into RIP
//
//   call form:
//     1) lea VTEMP1, [rip + after_ret_lbl]   ; absolute return address (in-blob);
//                                              the encoder patches disp32 to
//                                              point at the byte after our ret
//     2) VTEMP0 = [VSP] ; VSP += 8           ; pop target
//     3) VSP -= 8 ; [VSP] = VTEMP1           ; push return address
//     4) VSP -= 8 ; [VSP] = VTEMP0           ; push target (top of stack)
//     5) emit_vm_exit_body                   ; restores guest state, native
//                                              rsp = current VSP, top = target,
//                                              under-it = return address
//     6) ret                                 ; pops target into RIP, callee
//                                              later RETs into our after_ret_lbl
//     7) [after_ret_lbl] : the lifter has emitted a cmd_vm_enter immediately
//                          after this command, so execution resumes inside the
//                          VM with all guest state re-established.
//
// VTEMP0 / VTEMP1 are scratch in the IR ABI and emit_vm_exit_body restores them
// to their guest values from the save area at body-end. The "push target" /
// "push return" writes happen BEFORE vm_exit_body, but they target VSP-relative
// memory (the guest stack), not VTEMP slots, so they survive the body verbatim.
//
// Note on body interaction: vm_exit_body's first instruction is
// `mov [VREGS + 32], VSP` (slot 4 = RSP). That snapshots whatever VSP is at
// THAT moment -- which is exactly the post-push VSP we need native rsp to land
// on. So the order "set up guest stack, THEN run body" is correct.
// -----------------------------------------------------------------------------
void machine::emit_exit_to_top(encode_builder& eb, const ir::cmd_exit_to_top& c) {
    auto vsp = regs_.gpr(role::vsp);
    auto t0  = regs_.gpr(role::vtemp0);
    auto t1  = regs_.gpr(role::vtemp1);

    codec::label_id after_ret{};
    if (c.is_call()) {
        // Materialise the return address as an absolute VA via RIP-relative
        // LEA. The encoder records a within-blob fixup; finalize() patches
        // disp32 = (after_ret_offset - next_pc_after_lea).
        after_ret = eb.make_label();
        eb.make(ZYDIS_MNEMONIC_LEA, reg(t1), codec::lbl(after_ret, /*rip_lea*/ true));
    }

    // Pop target into VTEMP0. Use LEA for the VSP bump (RFLAGS-safe).
    eb.make(ZYDIS_MNEMONIC_MOV, reg(t0), mem(vsp, 0, 64));
    eb.make(ZYDIS_MNEMONIC_LEA, reg(vsp), mem(vsp, 8, 64));

    if (c.is_call()) {
        // Push return address (sits below target after final ret pops target).
        eb.make(ZYDIS_MNEMONIC_LEA, reg(vsp), mem(vsp, -8, 64));
        eb.make(ZYDIS_MNEMONIC_MOV, mem(vsp, 0, 64), reg(t1));
    }

    // Push target on the guest stack so it ends up at [new native rsp]
    // immediately after vm_exit_body switches rsp := VSP.
    eb.make(ZYDIS_MNEMONIC_LEA, reg(vsp), mem(vsp, -8, 64));
    eb.make(ZYDIS_MNEMONIC_MOV, mem(vsp, 0, 64), reg(t0));

    // Restore guest GPRs (incl. VTEMP0/VTEMP1) and switch native rsp to VSP.
    emit_vm_exit_body(eb);

    // Native ret: pops target into RIP. For the call case the callee will
    // eventually RET to after_ret which lies just past this `ret` byte.
    eb.make(ZYDIS_MNEMONIC_RET);

    if (c.is_call()) {
        // Bind after_ret_lbl at the current cursor -- the encoder fixup turns
        // the LEA above into "address of this byte".
        eb.label(after_ret);
    }
}

// -----------------------------------------------------------------------------
// xmm_op : SIMD pass-through.
//   reg_reg : emit `mnem dst_xmm, src_xmm` as-is. Vector regs aren't part of
//             our save area, so this preserves all observable state.
//   reg_mem : top of VSP holds the EA; pop into VTEMP0 then
//             `mnem dst_xmm, [VTEMP0]` at the declared memory width.
//   mem_reg : symmetric.
// -----------------------------------------------------------------------------
void machine::emit_xmm_op(encode_builder& eb, const ir::cmd_xmm_op& c) {
    if (c.kind() == ir::xmm_op_kind::reg_reg) {
        eb.make(c.mnem(), reg(c.dst()), reg(c.src()));
        return;
    }
    auto vsp = regs_.gpr(role::vsp);
    auto t0  = regs_.gpr(role::vtemp0);
    // Pop the EA into VTEMP0.
    eb.make(ZYDIS_MNEMONIC_MOV, reg(t0), mem(vsp, 0, 64));
    eb.make(ZYDIS_MNEMONIC_LEA, reg(vsp), mem(vsp, 8, 64));

    if (c.kind() == ir::xmm_op_kind::reg_mem) {
        eb.make(c.mnem(), reg(c.dst()), mem(t0, 0, c.size_bits()));
    } else {  // mem_reg
        eb.make(c.mnem(), mem(t0, 0, c.size_bits()), reg(c.src()));
    }
}

// -----------------------------------------------------------------------------
// push.imm  -- always pushes a 64-bit slot, value zero-extended.
// -----------------------------------------------------------------------------
void machine::emit_push_imm(encode_builder& eb, const ir::cmd_push_imm& c) {
    auto sz   = c.size();
    auto vsp  = regs_.gpr(role::vsp);
    auto t0   = regs_.gpr(role::vtemp0);
    std::uint64_t v = c.value();

    // Load imm into VTEMP0 sized as a zero-extended 64-bit value.
    // For 32-bit form a sized mov already zero-extends the upper 32. For
    // 8/16 a sub-32 mov leaves upper bits untouched, so go through the
    // 32-bit form with a pre-masked immediate.
    if (sz == ir::ir_size::bit_64) {
        eb.make(ZYDIS_MNEMONIC_MOV, reg(t0), imm(v, 8));
    } else if (sz == ir::ir_size::bit_32) {
        auto t0_32 = register_manager::narrow(t0, ir::ir_size::bit_32);
        eb.make(ZYDIS_MNEMONIC_MOV, reg(t0_32), imm(v & 0xffffffffu, 4));
    } else {
        // bit_8 / bit_16: write via 32-bit form with masked imm so the upper
        // bits of VTEMP0 are guaranteed zero.
        auto t0_32 = register_manager::narrow(t0, ir::ir_size::bit_32);
        std::uint64_t masked = (sz == ir::ir_size::bit_8) ? (v & 0xffu) : (v & 0xffffu);
        eb.make(ZYDIS_MNEMONIC_MOV, reg(t0_32), imm(masked, 4));
    }

    // sub VSP, 8 ; mov [VSP], VTEMP0_64
    eb.make(ZYDIS_MNEMONIC_SUB, reg(vsp), imm(8));
    eb.make(ZYDIS_MNEMONIC_MOV, mem(vsp, 0, 64), reg(t0));
}

// -----------------------------------------------------------------------------
// push.ctx -- read guest GPR slot, push 64-bit slot (zero-extended).
// -----------------------------------------------------------------------------
void machine::emit_push_reg_ctx(encode_builder& eb, const ir::cmd_push_reg_ctx& c) {
    auto sz    = c.size();
    auto vsp   = regs_.gpr(role::vsp);
    auto vregs = regs_.gpr(role::vregs);
    auto t0    = regs_.gpr(role::vtemp0);
    auto t0n   = register_manager::narrow(t0, sz);

    std::size_t slot = register_manager::guest_slot_index(c.reg());
    if (slot == static_cast<std::size_t>(-1)) {
        const char* nm = ZydisRegisterGetString(c.reg());
        throw std::runtime_error(std::string("emit_push_reg_ctx: non-GPR register: ") + (nm ? nm : "?"));
    }

    // Shared-stack model: VSP_phys IS guest RSP. The rsp save-area slot (4)
    // is only valid at vm_enter time and gets overwritten by vm_exit_body;
    // during steady-state IR execution every native PUSH/POP shifts VSP_phys.
    // So a read of "guest RSP" must come from VSP_phys, not from slot 4 --
    // otherwise idioms like `mov rbp, rsp` (push rsp; pop rbp) and
    // `sub rsp, imm` (push rsp; push imm; sub; pop rsp) read a stale value.
    if (slot == 4 && sz == ir::ir_size::bit_64) {
        // push.ctx64 rsp: save old VSP, decrement VSP, store old VSP at [VSP].
        eb.make(ZYDIS_MNEMONIC_MOV, reg(t0), reg(vsp));
        eb.make(ZYDIS_MNEMONIC_SUB, reg(vsp), imm(8));
        eb.make(ZYDIS_MNEMONIC_MOV, mem(vsp, 0, 64), reg(t0));
        return;
    }

    std::int64_t off = static_cast<std::int64_t>(slot * 8);

    if (sz == ir::ir_size::bit_64) {
        eb.make(ZYDIS_MNEMONIC_MOV, reg(t0), mem(vregs, off, 64));
    } else if (sz == ir::ir_size::bit_32) {
        // 32-bit mov to GPR32 zero-extends to 64.
        eb.make(ZYDIS_MNEMONIC_MOV, reg(t0n), mem(vregs, off, 32));
    } else {
        // bit_8 / bit_16: explicitly zero-extend via MOVZX into the 32-bit
        // form so VTEMP0_64 has all upper bits cleared.
        auto t0_32 = register_manager::narrow(t0, ir::ir_size::bit_32);
        eb.make(ZYDIS_MNEMONIC_MOVZX, reg(t0_32), mem(vregs, off, mem_bits(sz)));
    }

    eb.make(ZYDIS_MNEMONIC_SUB, reg(vsp), imm(8));
    eb.make(ZYDIS_MNEMONIC_MOV, mem(vsp, 0, 64), reg(t0));
}

// -----------------------------------------------------------------------------
// pop.ctx -- pop 64-bit slot, write low `sz` bits to guest GPR slot.
// For 32-bit ops we zero-extend (matching x64 semantics) by `mov t0n, t0n`.
// -----------------------------------------------------------------------------
void machine::emit_pop_reg_ctx(encode_builder& eb, const ir::cmd_pop_reg_ctx& c) {
    auto sz    = c.size();
    auto vsp   = regs_.gpr(role::vsp);
    auto vregs = regs_.gpr(role::vregs);
    auto t0    = regs_.gpr(role::vtemp0);
    auto t0n   = register_manager::narrow(t0, sz);

    std::size_t slot = register_manager::guest_slot_index(c.reg());
    if (slot == static_cast<std::size_t>(-1)) {
        const char* nm = ZydisRegisterGetString(c.reg());
        throw std::runtime_error(std::string("emit_pop_reg_ctx: non-GPR register: ") + (nm ? nm : "?"));
    }

    // Shared-stack model: VSP_phys IS guest RSP. A pop into RSP must update
    // VSP_phys directly so the new logical RSP is reflected in every
    // subsequent IR op. The natural `pop` semantics (read top, +=8) is
    // replaced by a single `mov VSP, [VSP]` since the popped value IS the
    // new VSP -- the +8 increment is subsumed by the new value.
    if (slot == 4 && sz == ir::ir_size::bit_64) {
        eb.make(ZYDIS_MNEMONIC_MOV, reg(vsp), mem(vsp, 0, 64));
        return;
    }

    std::int64_t off = static_cast<std::int64_t>(slot * 8);

    // Load 64 bits from VSP into VTEMP0.
    eb.make(ZYDIS_MNEMONIC_MOV, reg(t0), mem(vsp, 0, 64));
    eb.make(ZYDIS_MNEMONIC_ADD, reg(vsp), imm(8));

    // Write `sz` bits of VTEMP0 into the slot. For 32-bit we additionally
    // zero-extend (matching x64 register-write semantics); for 8/16 we
    // preserve the upper bits of the slot exactly like a real guest write.
    if (sz == ir::ir_size::bit_64) {
        eb.make(ZYDIS_MNEMONIC_MOV, mem(vregs, off, 64), reg(t0));
    } else if (sz == ir::ir_size::bit_32) {
        // Zero-extend the destination by writing 32 bits and clearing the
        // top half of the 8-byte slot explicitly.
        eb.make(ZYDIS_MNEMONIC_MOV, mem(vregs, off, 32), reg(t0n));
        eb.make(ZYDIS_MNEMONIC_MOV, mem(vregs, off + 4, 32), imm(0, 4));
    } else {
        // bit_8 / bit_16: sub-32 mov writes only the low bytes, upper bytes
        // of the slot are preserved -- which matches real x64 semantics.
        eb.make(ZYDIS_MNEMONIC_MOV, mem(vregs, off, mem_bits(sz)), reg(t0n));
    }
}

// -----------------------------------------------------------------------------
// handler_call -- pop right (VTEMP1), pop left (VTEMP0),
// VTEMP0 = VTEMP0 op VTEMP1, push VTEMP0.
// -----------------------------------------------------------------------------
void machine::emit_handler_call(encode_builder& eb, const ir::cmd_handler_call& c) {
    auto sz   = c.size();
    auto vsp  = regs_.gpr(role::vsp);
    auto t0   = regs_.gpr(role::vtemp0);
    auto t1   = regs_.gpr(role::vtemp1);
    auto t0n  = register_manager::narrow(t0, sz);
    auto t1n  = register_manager::narrow(t1, sz);

    // Pop right -> VTEMP1
    eb.make(ZYDIS_MNEMONIC_MOV, reg(t1), mem(vsp, 0, 64));
    eb.make(ZYDIS_MNEMONIC_ADD, reg(vsp), imm(8));
    // Pop left -> VTEMP0
    eb.make(ZYDIS_MNEMONIC_MOV, reg(t0), mem(vsp, 0, 64));
    eb.make(ZYDIS_MNEMONIC_ADD, reg(vsp), imm(8));

    ZydisMnemonic m = ZYDIS_MNEMONIC_INVALID;
    switch (c.op()) {
        case ir::binop::op_add: m = ZYDIS_MNEMONIC_ADD;  break;
        case ir::binop::op_sub: m = ZYDIS_MNEMONIC_SUB;  break;
        case ir::binop::op_xor: m = ZYDIS_MNEMONIC_XOR;  break;
        case ir::binop::op_and: m = ZYDIS_MNEMONIC_AND;  break;
        case ir::binop::op_or:  m = ZYDIS_MNEMONIC_OR;   break;
        case ir::binop::op_mul: m = ZYDIS_MNEMONIC_IMUL; break;
    }
    if (m == ZYDIS_MNEMONIC_INVALID) {
        throw std::runtime_error("emit_handler_call: unknown binop");
    }

    // VTEMP0 = VTEMP0 op VTEMP1 (sized -- zero-extends t0_64 for 32-bit).
    // imul on 8-bit isn't a 2-op form; M2 caller restricts to 16/32/64.
    if (sz == ir::ir_size::bit_64) {
        eb.make(m, reg(t0), reg(t1));
    } else {
        eb.make(m, reg(t0n), reg(t1n));
    }

    // Push result using LEA+MOV so native RFLAGS produced by the arith op
    // above survives until a subsequent cmd_save_flags reads it.
    eb.make(ZYDIS_MNEMONIC_LEA, reg(vsp), mem(vsp, -8, 64));
    eb.make(ZYDIS_MNEMONIC_MOV, mem(vsp, 0, 64), reg(t0));
}

// -----------------------------------------------------------------------------
// cmp / test : pop two, run op, set flags only (no push).
// -----------------------------------------------------------------------------
void machine::emit_cmp_test(encode_builder& eb, const ir::cmd_cmp_test& c) {
    auto sz   = c.size();
    auto vsp  = regs_.gpr(role::vsp);
    auto t0   = regs_.gpr(role::vtemp0);
    auto t1   = regs_.gpr(role::vtemp1);
    auto t0n  = register_manager::narrow(t0, sz);
    auto t1n  = register_manager::narrow(t1, sz);

    // Pop right -> VTEMP1
    eb.make(ZYDIS_MNEMONIC_MOV, reg(t1), mem(vsp, 0, 64));
    eb.make(ZYDIS_MNEMONIC_ADD, reg(vsp), imm(8));
    // Pop left -> VTEMP0
    eb.make(ZYDIS_MNEMONIC_MOV, reg(t0), mem(vsp, 0, 64));
    eb.make(ZYDIS_MNEMONIC_ADD, reg(vsp), imm(8));

    ZydisMnemonic m = (c.op() == ir::cmp_kind::op_cmp) ? ZYDIS_MNEMONIC_CMP
                                                       : ZYDIS_MNEMONIC_TEST;
    if (sz == ir::ir_size::bit_64) {
        eb.make(m, reg(t0), reg(t1));
    } else {
        eb.make(m, reg(t0n), reg(t1n));
    }
    // No emit_save_flags here -- the lifter emits cmd_save_flags so that the
    // native RFLAGS produced by CMP/TEST is captured before any later op
    // clobbers it.
}

// -----------------------------------------------------------------------------
// unary inc/dec/neg/not/bswap on top of VSP.
// -----------------------------------------------------------------------------
void machine::emit_unary(encode_builder& eb, const ir::cmd_unary& c) {
    auto sz   = c.size();
    auto vsp  = regs_.gpr(role::vsp);
    auto t0   = regs_.gpr(role::vtemp0);
    auto t0n  = register_manager::narrow(t0, sz);

    // Pop -> VTEMP0
    eb.make(ZYDIS_MNEMONIC_MOV, reg(t0), mem(vsp, 0, 64));
    eb.make(ZYDIS_MNEMONIC_ADD, reg(vsp), imm(8));

    ZydisMnemonic m;
    switch (c.op()) {
        case ir::unop::op_inc:   m = ZYDIS_MNEMONIC_INC;   break;
        case ir::unop::op_dec:   m = ZYDIS_MNEMONIC_DEC;   break;
        case ir::unop::op_neg:   m = ZYDIS_MNEMONIC_NEG;   break;
        case ir::unop::op_not:   m = ZYDIS_MNEMONIC_NOT;   break;
        case ir::unop::op_bswap: m = ZYDIS_MNEMONIC_BSWAP; break;
        default: throw std::runtime_error("emit_unary: unknown unop");
    }
    if (c.op() == ir::unop::op_bswap) {
        // BSWAP is defined only for r32/r64; the encoding for r16 is
        // architecturally undefined. The lifter guarantees sz >= 32 for
        // BSWAP, so we narrow the wide GPR to either r32 or r64 here.
        if (sz == ir::ir_size::bit_64) eb.make(m, reg(t0));
        else                            eb.make(m, reg(register_manager::narrow(t0, ir::ir_size::bit_32)));
    } else if (sz == ir::ir_size::bit_64) {
        eb.make(m, reg(t0));
    } else {
        eb.make(m, reg(t0n));
    }
    // Push result using LEA+MOV so native RFLAGS produced by INC/DEC/NEG above
    // survives until a subsequent cmd_save_flags reads it. NOT doesn't change
    // RFLAGS, but using LEA uniformly keeps VFLAGS preservation trivially safe.
    eb.make(ZYDIS_MNEMONIC_LEA, reg(vsp), mem(vsp, -8, 64));
    eb.make(ZYDIS_MNEMONIC_MOV, mem(vsp, 0, 64), reg(t0));
}

// -----------------------------------------------------------------------------
// Label / jmp / jcc.
// -----------------------------------------------------------------------------
void machine::emit_label(encode_builder& eb, const ir::cmd_label& c) {
    auto id = resolve_label(eb, c.id());
    eb.label(id);
}

void machine::emit_jmp(encode_builder& eb, const ir::cmd_jmp& c) {
    auto id = resolve_label(eb, c.target());
    eb.make(ZYDIS_MNEMONIC_JMP, codec::lbl(id));
}

void machine::emit_jcc(encode_builder& eb, const ir::cmd_jcc& c) {
    // Restore flags into RFLAGS before branching.
    emit_load_flags(eb);
    auto id = resolve_label(eb, c.target());
    eb.make(c.cond(), codec::lbl(id));
}

// -----------------------------------------------------------------------------
// cmov : pop src -> VTEMP1, pop dst -> VTEMP0, load_flags, cmovcc, push.
// CMOVcc does not modify flags, so we must keep VFLAGS unchanged. We use LEA
// for stack-pointer bookkeeping after load_flags so RFLAGS stays valid until
// after the conditional move, then we explicitly skip emit_save_flags.
// -----------------------------------------------------------------------------
void machine::emit_cmov(encode_builder& eb, const ir::cmd_cmov& c) {
    auto sz   = c.size();
    auto vsp  = regs_.gpr(role::vsp);
    auto t0   = regs_.gpr(role::vtemp0);
    auto t1   = regs_.gpr(role::vtemp1);
    auto t0n  = register_manager::narrow(t0, sz);
    auto t1n  = register_manager::narrow(t1, sz);

    // Pop right (src) -> VTEMP1, pop left (dst current) -> VTEMP0.
    // Use ADD here -- flags will be reloaded right before CMOVcc.
    eb.make(ZYDIS_MNEMONIC_MOV, reg(t1), mem(vsp, 0, 64));
    eb.make(ZYDIS_MNEMONIC_ADD, reg(vsp), imm(8));
    eb.make(ZYDIS_MNEMONIC_MOV, reg(t0), mem(vsp, 0, 64));
    eb.make(ZYDIS_MNEMONIC_ADD, reg(vsp), imm(8));

    // Reload guest flags into RFLAGS.
    emit_load_flags(eb);

    // CMOVcc t0, t1 (sized). Only 16/32/64 valid; lifter rejects bit_8.
    if (sz == ir::ir_size::bit_64) {
        eb.make(c.cond(), reg(t0), reg(t1));
    } else {
        eb.make(c.cond(), reg(t0n), reg(t1n));
    }

    // Push result using LEA + MOV so the post-CMOV flags are preserved.
    eb.make(ZYDIS_MNEMONIC_LEA, reg(vsp), mem(vsp, -8, 64));
    eb.make(ZYDIS_MNEMONIC_MOV, mem(vsp, 0, 64), reg(t0));
    // No emit_save_flags -- CMOVcc does not change flags.
}

// -----------------------------------------------------------------------------
// setcc : load flags, run SETcc into VTEMP0_8, zero-extend, push.
// -----------------------------------------------------------------------------
void machine::emit_setcc(encode_builder& eb, const ir::cmd_setcc& c) {
    auto vsp  = regs_.gpr(role::vsp);
    auto t0   = regs_.gpr(role::vtemp0);
    auto t0_8 = register_manager::narrow(t0, ir::ir_size::bit_8);
    auto t0_32 = register_manager::narrow(t0, ir::ir_size::bit_32);

    emit_load_flags(eb);
    eb.make(c.cond(), reg(t0_8));
    // Zero-extend low 8 bits to 64 via movzx into 32-bit form.
    eb.make(ZYDIS_MNEMONIC_MOVZX, reg(t0_32), reg(t0_8));

    // Push VTEMP0 (now a zero-extended 0/1).
    eb.make(ZYDIS_MNEMONIC_SUB, reg(vsp), imm(8));
    eb.make(ZYDIS_MNEMONIC_MOV, mem(vsp, 0, 64), reg(t0));
    // SETcc itself doesn't change flags, but the load_flags above already
    // smashed them into RFLAGS; nothing we need to save back.
}

// -----------------------------------------------------------------------------
// movx : extend top-of-VSP from src_size to dst_size (zero or sign).
// -----------------------------------------------------------------------------
void machine::emit_movx(encode_builder& eb, const ir::cmd_movx& c) {
    auto vsp     = regs_.gpr(role::vsp);
    auto t0      = regs_.gpr(role::vtemp0);
    auto t0_src  = register_manager::narrow(t0, c.src_size());
    auto t0_dst  = register_manager::narrow(t0, c.dst_size());

    // Pop into VTEMP0 (low src_size bits hold the value, upper are zero).
    eb.make(ZYDIS_MNEMONIC_MOV, reg(t0), mem(vsp, 0, 64));
    eb.make(ZYDIS_MNEMONIC_ADD, reg(vsp), imm(8));

    if (c.op() == ir::movx_kind::zero) {
        // movzx forms:
        //   r16/r32/r64, r8
        //   r32/r64,     r16
        // mov r32, r32 already zero-extends to 64, so for dst=64,src=32 we
        // do nothing -- the slot already has the upper half cleared from
        // the original push.
        if (c.src_size() == c.dst_size()) {
            // no-op
        } else if (c.src_size() == ir::ir_size::bit_32 &&
                   c.dst_size() == ir::ir_size::bit_64) {
            // upper 32 already zero
        } else {
            // pick the smallest valid destination form for movzx
            ZydisRegister dst_reg = t0_dst;
            if (c.dst_size() == ir::ir_size::bit_64) {
                // movzx r64,r8 / r64,r16 both valid; encoder picks the form
                dst_reg = t0;
            }
            eb.make(ZYDIS_MNEMONIC_MOVZX, reg(dst_reg), reg(t0_src));
        }
    } else {
        // sign-extend forms:
        //   movsx r16/r32/r64, r8
        //   movsx r32/r64,     r16
        //   movsxd r64,        r32      (different mnemonic)
        if (c.src_size() == c.dst_size()) {
            // no-op
        } else if (c.src_size() == ir::ir_size::bit_32 &&
                   c.dst_size() == ir::ir_size::bit_64) {
            eb.make(ZYDIS_MNEMONIC_MOVSXD, reg(t0), reg(t0_src));
        } else {
            ZydisRegister dst_reg = t0_dst;
            if (c.dst_size() == ir::ir_size::bit_64) dst_reg = t0;
            eb.make(ZYDIS_MNEMONIC_MOVSX, reg(dst_reg), reg(t0_src));
        }
    }

    // Push VTEMP0 back (now extended to 64-bit zero-extended).
    eb.make(ZYDIS_MNEMONIC_SUB, reg(vsp), imm(8));
    eb.make(ZYDIS_MNEMONIC_MOV, mem(vsp, 0, 64), reg(t0));
}

// -----------------------------------------------------------------------------
// shift : pop value, shl/shr/sar by an immediate count, save flags, push.
// -----------------------------------------------------------------------------
void machine::emit_shift(encode_builder& eb, const ir::cmd_shift& c) {
    auto sz  = c.size();
    auto vsp = regs_.gpr(role::vsp);
    auto t0  = regs_.gpr(role::vtemp0);
    auto t0n = register_manager::narrow(t0, sz);

    // Pop -> VTEMP0
    eb.make(ZYDIS_MNEMONIC_MOV, reg(t0), mem(vsp, 0, 64));
    eb.make(ZYDIS_MNEMONIC_ADD, reg(vsp), imm(8));

    ZydisMnemonic m;
    switch (c.op()) {
        case ir::shift_kind::shl: m = ZYDIS_MNEMONIC_SHL; break;
        case ir::shift_kind::shr: m = ZYDIS_MNEMONIC_SHR; break;
        case ir::shift_kind::sar: m = ZYDIS_MNEMONIC_SAR; break;
        case ir::shift_kind::rol: m = ZYDIS_MNEMONIC_ROL; break;
        case ir::shift_kind::ror: m = ZYDIS_MNEMONIC_ROR; break;
        default: throw std::runtime_error("emit_shift: unknown shift_kind");
    }

    if (sz == ir::ir_size::bit_64) {
        eb.make(m, reg(t0),  imm(c.count(), 1));
    } else {
        eb.make(m, reg(t0n), imm(c.count(), 1));
    }
    // Push result using LEA+MOV so the shift's native RFLAGS survives until a
    // subsequent cmd_save_flags reads it.
    eb.make(ZYDIS_MNEMONIC_LEA, reg(vsp), mem(vsp, -8, 64));
    eb.make(ZYDIS_MNEMONIC_MOV, mem(vsp, 0, 64), reg(t0));
}

// -----------------------------------------------------------------------------
// load_mem : pop 64-bit EA from top of VSP into VTEMP1, load `sz` bytes from
// [VTEMP1] into VTEMP0 (zero-extended for 8/16/32), push VTEMP0.
// Does NOT touch VFLAGS slot (only native RFLAGS, which is irrelevant once
// the lifter has already issued a cmd_save_flags).
// -----------------------------------------------------------------------------
void machine::emit_mem_load(encode_builder& eb, const ir::cmd_load_mem& c) {
    auto sz    = c.size();
    auto vsp   = regs_.gpr(role::vsp);
    auto t0    = regs_.gpr(role::vtemp0);
    auto t1    = regs_.gpr(role::vtemp1);
    auto t0n   = register_manager::narrow(t0, sz);
    auto t0_32 = register_manager::narrow(t0, ir::ir_size::bit_32);

    // Pop EA -> VTEMP1
    eb.make(ZYDIS_MNEMONIC_MOV, reg(t1), mem(vsp, 0, 64));
    eb.make(ZYDIS_MNEMONIC_ADD, reg(vsp), imm(8));

    if (sz == ir::ir_size::bit_64) {
        eb.make(ZYDIS_MNEMONIC_MOV, reg(t0), mem(t1, 0, 64));
    } else if (sz == ir::ir_size::bit_32) {
        // mov r32, m32 -- zero-extends to r64 automatically.
        eb.make(ZYDIS_MNEMONIC_MOV, reg(t0n), mem(t1, 0, 32));
    } else {
        // 8/16: MOVZX into 32-bit form for clean upper-zero.
        eb.make(ZYDIS_MNEMONIC_MOVZX, reg(t0_32), mem(t1, 0, mem_bits(sz)));
    }

    // Push loaded value.
    eb.make(ZYDIS_MNEMONIC_SUB, reg(vsp), imm(8));
    eb.make(ZYDIS_MNEMONIC_MOV, mem(vsp, 0, 64), reg(t0));
}

// -----------------------------------------------------------------------------
// store_mem : pop 64-bit EA from top of VSP into VTEMP1, pop value into
// VTEMP0, store low `sz` bytes of VTEMP0 to [VTEMP1]. Does NOT touch VFLAGS.
// -----------------------------------------------------------------------------
void machine::emit_mem_store(encode_builder& eb, const ir::cmd_store_mem& c) {
    auto sz    = c.size();
    auto vsp   = regs_.gpr(role::vsp);
    auto t0    = regs_.gpr(role::vtemp0);
    auto t1    = regs_.gpr(role::vtemp1);
    auto t0n   = register_manager::narrow(t0, sz);

    // Pop EA (top) -> VTEMP1
    eb.make(ZYDIS_MNEMONIC_MOV, reg(t1), mem(vsp, 0, 64));
    eb.make(ZYDIS_MNEMONIC_ADD, reg(vsp), imm(8));
    // Pop value -> VTEMP0
    eb.make(ZYDIS_MNEMONIC_MOV, reg(t0), mem(vsp, 0, 64));
    eb.make(ZYDIS_MNEMONIC_ADD, reg(vsp), imm(8));

    if (sz == ir::ir_size::bit_64) {
        eb.make(ZYDIS_MNEMONIC_MOV, mem(t1, 0, 64), reg(t0));
    } else {
        eb.make(ZYDIS_MNEMONIC_MOV, mem(t1, 0, mem_bits(sz)), reg(t0n));
    }
}

// -----------------------------------------------------------------------------
// save_flags : explicit IR-level snapshot of native RFLAGS into VFLAGS slot.
// -----------------------------------------------------------------------------
void machine::emit_save_flags_cmd(encode_builder& eb, const ir::cmd_save_flags&) {
    emit_save_flags(eb);
}

// -----------------------------------------------------------------------------
// push_vsp : push (VSP + adjust) onto VSP. Materialises guest logical RSP for
// [rsp+N] memory operands by accounting for `adjust` = stack_displacement that
// the lifter accumulated during the current x86 instruction.
//
// Emits:
//     lea VTEMP0, [VSP + adjust]
//     lea VSP,    [VSP - 8]
//     mov [VSP],  VTEMP0
//
// LEA preserves native RFLAGS, so this is flag-safe.
// -----------------------------------------------------------------------------
void machine::emit_push_vsp(encode_builder& eb, const ir::cmd_push_vsp& c) {
    auto vsp = regs_.gpr(role::vsp);
    auto t0  = regs_.gpr(role::vtemp0);

    eb.make(ZYDIS_MNEMONIC_LEA,
            reg(t0),
            mem(vsp, static_cast<std::int64_t>(c.adjust()), 64));
    eb.make(ZYDIS_MNEMONIC_LEA, reg(vsp), mem(vsp, -8, 64));
    eb.make(ZYDIS_MNEMONIC_MOV, mem(vsp, 0, 64), reg(t0));
}

// -----------------------------------------------------------------------------
// push_vbase_rva : push (VBASE + rva) onto VSP. Materialises an absolute VA
// from a static RVA -- used by the lifter to translate `[rip + disp]` memory
// operands into emit_push_ea-style stack-pushed addresses.
//
// Emits:
//     mov  VTEMP0, imm64 rva           ; full 64-bit form handles RVAs > 2GiB
//     lea  VTEMP0, [VBASE + VTEMP0*1]  ; preserves RFLAGS (unlike add)
//     lea  VSP,    [VSP - 8]
//     mov  [VSP],  VTEMP0
//
// LEA is used for both the address-add and the VSP decrement so native RFLAGS
// is preserved across this op (VFLAGS is the guest-flag source of truth).
// -----------------------------------------------------------------------------
void machine::emit_push_vbase_rva(encode_builder& eb, const ir::cmd_push_vbase_rva& c) {
    auto vsp   = regs_.gpr(role::vsp);
    auto vbase = regs_.gpr(role::vbase);
    auto t0    = regs_.gpr(role::vtemp0);

    eb.make(ZYDIS_MNEMONIC_MOV, reg(t0), imm(c.rva(), 8));
    eb.make(ZYDIS_MNEMONIC_LEA, reg(t0),
            codec::mem_idx(vbase, t0, /*scale*/ 1, /*disp*/ 0, /*bits*/ 64));
    eb.make(ZYDIS_MNEMONIC_LEA, reg(vsp), mem(vsp, -8, 64));
    eb.make(ZYDIS_MNEMONIC_MOV, mem(vsp, 0, 64), reg(t0));
}

// -----------------------------------------------------------------------------
// compile
// -----------------------------------------------------------------------------
compiled_region machine::compile(const ir::block& blk) {
    encode_builder eb;
    ir_label_map_.clear();
    pending_fixups_.clear();

    bool emitted_enter = false;
    bool emitted_exit  = false;

    for (const auto& cmd_sp : blk) {
        switch (cmd_sp->kind()) {
            case ir::command_kind::vm_enter:
                emit_vm_enter(eb);
                emitted_enter = true;
                break;
            case ir::command_kind::vm_exit:
                emit_vm_exit(eb);
                emitted_exit = true;
                break;
            case ir::command_kind::push_imm:
                emit_push_imm(eb,
                    static_cast<const ir::cmd_push_imm&>(*cmd_sp));
                break;
            case ir::command_kind::push_reg_ctx:
                emit_push_reg_ctx(eb,
                    static_cast<const ir::cmd_push_reg_ctx&>(*cmd_sp));
                break;
            case ir::command_kind::pop_reg_ctx:
                emit_pop_reg_ctx(eb,
                    static_cast<const ir::cmd_pop_reg_ctx&>(*cmd_sp));
                break;
            case ir::command_kind::handler_call:
                emit_handler_call(eb,
                    static_cast<const ir::cmd_handler_call&>(*cmd_sp));
                break;
            case ir::command_kind::cmp_test:
                emit_cmp_test(eb,
                    static_cast<const ir::cmd_cmp_test&>(*cmd_sp));
                break;
            case ir::command_kind::unary:
                emit_unary(eb,
                    static_cast<const ir::cmd_unary&>(*cmd_sp));
                break;
            case ir::command_kind::label:
                emit_label(eb,
                    static_cast<const ir::cmd_label&>(*cmd_sp));
                break;
            case ir::command_kind::jmp:
                emit_jmp(eb,
                    static_cast<const ir::cmd_jmp&>(*cmd_sp));
                break;
            case ir::command_kind::jcc:
                emit_jcc(eb,
                    static_cast<const ir::cmd_jcc&>(*cmd_sp));
                break;
            case ir::command_kind::cmov:
                emit_cmov(eb,
                    static_cast<const ir::cmd_cmov&>(*cmd_sp));
                break;
            case ir::command_kind::setcc:
                emit_setcc(eb,
                    static_cast<const ir::cmd_setcc&>(*cmd_sp));
                break;
            case ir::command_kind::movx:
                emit_movx(eb,
                    static_cast<const ir::cmd_movx&>(*cmd_sp));
                break;
            case ir::command_kind::shift:
                emit_shift(eb,
                    static_cast<const ir::cmd_shift&>(*cmd_sp));
                break;
            case ir::command_kind::load_mem:
                emit_mem_load(eb,
                    static_cast<const ir::cmd_load_mem&>(*cmd_sp));
                break;
            case ir::command_kind::store_mem:
                emit_mem_store(eb,
                    static_cast<const ir::cmd_store_mem&>(*cmd_sp));
                break;
            case ir::command_kind::save_flags:
                emit_save_flags_cmd(eb,
                    static_cast<const ir::cmd_save_flags&>(*cmd_sp));
                break;
            case ir::command_kind::push_vsp:
                emit_push_vsp(eb,
                    static_cast<const ir::cmd_push_vsp&>(*cmd_sp));
                break;
            case ir::command_kind::push_vbase_rva:
                emit_push_vbase_rva(eb,
                    static_cast<const ir::cmd_push_vbase_rva&>(*cmd_sp));
                break;
            case ir::command_kind::exit_to_rva:
                emit_exit_to_rva(eb,
                    static_cast<const ir::cmd_exit_to_rva&>(*cmd_sp));
                // exit_to_rva counts as an exit for the "needs vm_exit" check
                // -- but only if it's a jmp (terminator). For call, the lifter
                // emits a follow-up cmd_vm_enter and lifting continues.
                if (!static_cast<const ir::cmd_exit_to_rva&>(*cmd_sp).is_call()) {
                    emitted_exit = true;
                }
                break;
            case ir::command_kind::exit_to_top:
                emit_exit_to_top(eb,
                    static_cast<const ir::cmd_exit_to_top&>(*cmd_sp));
                if (!static_cast<const ir::cmd_exit_to_top&>(*cmd_sp).is_call()) {
                    emitted_exit = true;
                }
                break;
            case ir::command_kind::xmm_op:
                emit_xmm_op(eb,
                    static_cast<const ir::cmd_xmm_op&>(*cmd_sp));
                break;
            case ir::command_kind::x86_exec:
                throw std::runtime_error(
                    "machine::compile: cmd_x86_exec emitted but pass-through "
                    "is disabled in this build");
        }
    }

    if (!emitted_enter) {
        throw std::runtime_error("machine::compile: block lacks vm_enter");
    }
    if (!emitted_exit) {
        // Tolerate but auto-append.
        emit_vm_exit(eb);
    }

    compiled_region out;
    out.handlers     = eb.finalize();
    out.entry_offset = 0;
    out.fixups       = std::move(pending_fixups_);
    pending_fixups_.clear();
    return out;
}

}  // namespace wenaxvm::vm
