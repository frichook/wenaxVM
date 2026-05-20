#pragma once

// IR command hierarchy. Stack-VM semantics: every operand is materialised on
// a virtual stack (VSP) before any handler runs, and every result is left on
// VSP for the next consumer to pop. There is no register file in the IR.
//
// All arithmetic is stack-based:
//   push lhs   ; push rhs   ; handler_call(ADD,32)   -> result on top of VSP
//   pop dst
//
// MOV is implemented as just "push src, pop dst".

#include <Zydis/Zydis.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace wenaxvm::ir {

enum class ir_size : std::uint8_t {
    bit_8  = 1,
    bit_16 = 2,
    bit_32 = 4,
    bit_64 = 8,
};

constexpr std::uint8_t bytes_of(ir_size s) { return static_cast<std::uint8_t>(s); }
constexpr std::uint16_t bits_of(ir_size s) {
    return static_cast<std::uint16_t>(static_cast<std::uint8_t>(s) * 8);
}

enum class command_kind : std::uint8_t {
    vm_enter,
    vm_exit,
    push_imm,
    push_reg_ctx,    // push value of guest gpr (slot in VREGS) onto VSP
    pop_reg_ctx,     // pop top of VSP into guest gpr slot
    handler_call,    // binary arithmetic that pushes result on top of VSP
    cmp_test,        // binary compare: pops two, sets flags, no push
    unary,           // unary inc/dec/neg/not on top of VSP
    label,           // intra-block label target
    jmp,             // unconditional jump to label
    jcc,             // conditional jump (uses VFLAGS)
    cmov,            // conditional move: pop src, pop dst, cmovcc, push dst
    setcc,           // setcc: push (cond ? 1 : 0) as 8-bit
    movx,            // movzx / movsx / movsxd : pop src(sz_in), push extended sz_out
    shift,           // shl/shr/sar by an immediate count
    load_mem,        // pop 64-bit EA from VSP, load sz bytes from [EA], push (zero-ext 64)
    store_mem,       // pop 64-bit EA from VSP, pop 64-bit value, store low sz bytes
    save_flags,      // snapshot native RFLAGS into VFLAGS slot
    push_vsp,        // push (VSP + adjust) onto VSP -- materialises guest RSP
    push_vbase_rva,  // push (VBASE + rva) onto VSP -- materialises VA from RVA
    exit_to_rva,     // vm_exit body + native jmp/call rel32 to external RVA
    exit_to_top,     // pop target from VSP, vm_exit body, native jmp/call rax
    xmm_op,          // SIMD pass-through (XMM/YMM reg/mem mov & reg-only ALU)
    x86_exec,        // raw x86 pass-through (vm_exit -> native instr -> vm_enter sandwich)
};

class base_command {
 public:
    explicit base_command(command_kind k) : kind_(k) {}
    virtual ~base_command() = default;
    command_kind kind() const noexcept { return kind_; }
    virtual std::string debug_str() const = 0;

 private:
    command_kind kind_;
};

using command_ptr = std::shared_ptr<base_command>;
using block       = std::vector<command_ptr>;

// -----------------------------------------------------------------------------
// vm_enter / vm_exit
// -----------------------------------------------------------------------------
class cmd_vm_enter final : public base_command {
 public:
    cmd_vm_enter() : base_command(command_kind::vm_enter) {}
    std::string debug_str() const override { return "vm_enter"; }
};

class cmd_vm_exit final : public base_command {
 public:
    cmd_vm_exit() : base_command(command_kind::vm_exit) {}
    std::string debug_str() const override { return "vm_exit"; }
};

// -----------------------------------------------------------------------------
// push_imm
// -----------------------------------------------------------------------------
class cmd_push_imm final : public base_command {
 public:
    cmd_push_imm(std::uint64_t v, ir_size sz)
        : base_command(command_kind::push_imm), value_(v), size_(sz) {}
    std::uint64_t value() const noexcept { return value_; }
    ir_size       size()  const noexcept { return size_; }
    std::string   debug_str() const override;

 private:
    std::uint64_t value_;
    ir_size       size_;
};

// -----------------------------------------------------------------------------
// push_reg_ctx / pop_reg_ctx
// -----------------------------------------------------------------------------
class cmd_push_reg_ctx final : public base_command {
 public:
    cmd_push_reg_ctx(ZydisRegister r, ir_size sz)
        : base_command(command_kind::push_reg_ctx), reg_(r), size_(sz) {}
    ZydisRegister reg()  const noexcept { return reg_; }
    ir_size       size() const noexcept { return size_; }
    std::string   debug_str() const override;

 private:
    ZydisRegister reg_;
    ir_size       size_;
};

class cmd_pop_reg_ctx final : public base_command {
 public:
    cmd_pop_reg_ctx(ZydisRegister r, ir_size sz)
        : base_command(command_kind::pop_reg_ctx), reg_(r), size_(sz) {}
    ZydisRegister reg()  const noexcept { return reg_; }
    ir_size       size() const noexcept { return size_; }
    std::string   debug_str() const override;

 private:
    ZydisRegister reg_;
    ir_size       size_;
};

// -----------------------------------------------------------------------------
// handler_call: binary arithmetic that consumes two values from VSP top
// and pushes the result. ZF/SF/CF/OF go into VFLAGS.
// -----------------------------------------------------------------------------
enum class binop : std::uint8_t {
    op_add,
    op_sub,
    op_xor,
    op_and,
    op_or,
    op_mul,    // imul (signed)
};

class cmd_handler_call final : public base_command {
 public:
    cmd_handler_call(binop op, ir_size sz)
        : base_command(command_kind::handler_call), op_(op), size_(sz) {}
    binop   op()   const noexcept { return op_; }
    ir_size size() const noexcept { return size_; }
    std::string debug_str() const override;

 private:
    binop   op_;
    ir_size size_;
};

// -----------------------------------------------------------------------------
// cmd_cmp_test : pops two, sets flags, doesn't push result
// -----------------------------------------------------------------------------
enum class cmp_kind : std::uint8_t {
    op_cmp,    // cmp lhs, rhs  (== sub but no writeback)
    op_test,   // test lhs, rhs (== and but no writeback)
};

class cmd_cmp_test final : public base_command {
 public:
    cmd_cmp_test(cmp_kind op, ir_size sz)
        : base_command(command_kind::cmp_test), op_(op), size_(sz) {}
    cmp_kind op()   const noexcept { return op_; }
    ir_size  size() const noexcept { return size_; }
    std::string debug_str() const override;

 private:
    cmp_kind op_;
    ir_size  size_;
};

// -----------------------------------------------------------------------------
// cmd_unary : pop value, op, push value (flags affected for inc/dec/neg only;
//             not/bswap leave flags untouched)
// -----------------------------------------------------------------------------
enum class unop : std::uint8_t {
    op_inc,
    op_dec,
    op_neg,
    op_not,    // bitwise NOT; doesn't touch flags
    op_bswap,  // byte-reverse (BSWAP r32/r64); doesn't touch flags
};

class cmd_unary final : public base_command {
 public:
    cmd_unary(unop op, ir_size sz)
        : base_command(command_kind::unary), op_(op), size_(sz) {}
    unop    op()   const noexcept { return op_; }
    ir_size size() const noexcept { return size_; }
    std::string debug_str() const override;

 private:
    unop    op_;
    ir_size size_;
};

// -----------------------------------------------------------------------------
// cmd_label / cmd_jmp / cmd_jcc -- intra-block control flow
// -----------------------------------------------------------------------------
class cmd_label final : public base_command {
 public:
    explicit cmd_label(std::uint32_t id)
        : base_command(command_kind::label), id_(id) {}
    std::uint32_t id() const noexcept { return id_; }
    std::string debug_str() const override;

 private:
    std::uint32_t id_;
};

class cmd_jmp final : public base_command {
 public:
    explicit cmd_jmp(std::uint32_t target)
        : base_command(command_kind::jmp), target_(target) {}
    std::uint32_t target() const noexcept { return target_; }
    std::string debug_str() const override;

 private:
    std::uint32_t target_;
};

// Conditions encoded as Zydis Jcc mnemonics. Stored as ZydisMnemonic value.
class cmd_jcc final : public base_command {
 public:
    cmd_jcc(ZydisMnemonic cond, std::uint32_t target)
        : base_command(command_kind::jcc), cond_(cond), target_(target) {}
    ZydisMnemonic cond()   const noexcept { return cond_; }
    std::uint32_t target() const noexcept { return target_; }
    std::string debug_str() const override;

 private:
    ZydisMnemonic cond_;
    std::uint32_t target_;
};

// -----------------------------------------------------------------------------
// cmd_cmov : conditional move (16/32/64).
// Stack layout: push dst (current), push src, cmov, pop dst.
// `cond` is a Zydis CMOVcc mnemonic (e.g. ZYDIS_MNEMONIC_CMOVZ).
// -----------------------------------------------------------------------------
class cmd_cmov final : public base_command {
 public:
    cmd_cmov(ZydisMnemonic cond, ir_size sz)
        : base_command(command_kind::cmov), cond_(cond), size_(sz) {}
    ZydisMnemonic cond() const noexcept { return cond_; }
    ir_size       size() const noexcept { return size_; }
    std::string debug_str() const override;

 private:
    ZydisMnemonic cond_;
    ir_size       size_;
};

// -----------------------------------------------------------------------------
// cmd_setcc : push (cond ? 1 : 0) as an 8-bit (zero-extended) value.
// -----------------------------------------------------------------------------
class cmd_setcc final : public base_command {
 public:
    explicit cmd_setcc(ZydisMnemonic cond)
        : base_command(command_kind::setcc), cond_(cond) {}
    ZydisMnemonic cond() const noexcept { return cond_; }
    std::string debug_str() const override;

 private:
    ZydisMnemonic cond_;
};

// -----------------------------------------------------------------------------
// cmd_movx : zero/sign extend the value on top of VSP from src_size to dst_size.
// -----------------------------------------------------------------------------
enum class movx_kind : std::uint8_t { zero, sign };

class cmd_movx final : public base_command {
 public:
    cmd_movx(movx_kind k, ir_size src_sz, ir_size dst_sz)
        : base_command(command_kind::movx), kind_(k), src_(src_sz), dst_(dst_sz) {}
    movx_kind op()       const noexcept { return kind_; }
    ir_size   src_size() const noexcept { return src_;  }
    ir_size   dst_size() const noexcept { return dst_;  }
    std::string debug_str() const override;

 private:
    movx_kind kind_;
    ir_size   src_;
    ir_size   dst_;
};

// -----------------------------------------------------------------------------
// cmd_shift : shl/shr/sar of top of VSP by an immediate count.
// -----------------------------------------------------------------------------
enum class shift_kind : std::uint8_t { shl, shr, sar, rol, ror };

class cmd_shift final : public base_command {
 public:
    cmd_shift(shift_kind k, ir_size sz, std::uint8_t count)
        : base_command(command_kind::shift), kind_(k), size_(sz), count_(count) {}
    shift_kind   op()    const noexcept { return kind_;  }
    ir_size      size()  const noexcept { return size_;  }
    std::uint8_t count() const noexcept { return count_; }
    std::string debug_str() const override;

 private:
    shift_kind   kind_;
    ir_size      size_;
    std::uint8_t count_;
};

// -----------------------------------------------------------------------------
// cmd_load_mem : pop 64-bit EA from VSP top, load sz bytes from [EA], push the
// loaded value (zero-extended to 64-bit slot). Does not touch VFLAGS.
// -----------------------------------------------------------------------------
class cmd_load_mem final : public base_command {
 public:
    explicit cmd_load_mem(ir_size sz)
        : base_command(command_kind::load_mem), size_(sz) {}
    ir_size size() const noexcept { return size_; }
    std::string debug_str() const override;

 private:
    ir_size size_;
};

// -----------------------------------------------------------------------------
// cmd_store_mem : pop EA from VSP top, pop value, store low sz bytes to [EA].
// Does not touch VFLAGS.
// -----------------------------------------------------------------------------
class cmd_store_mem final : public base_command {
 public:
    explicit cmd_store_mem(ir_size sz)
        : base_command(command_kind::store_mem), size_(sz) {}
    ir_size size() const noexcept { return size_; }
    std::string debug_str() const override;

 private:
    ir_size size_;
};

// -----------------------------------------------------------------------------
// cmd_push_vsp : push (current_VSP + adjust) onto VSP. Used by the lifter to
// materialise the guest's logical RSP for [rsp+N]-style memory operands; the
// lifter passes its tracked stack_displacement as `adjust` so the result lands
// at the value RSP had at the start of the current x86 instruction.
//
// Net VSP change: -8 (one push). The machine emits:
//     lea VTEMP0, [VSP + adjust]
//     lea VSP,    [VSP - 8]
//     mov [VSP],  VTEMP0
// -----------------------------------------------------------------------------
class cmd_push_vsp final : public base_command {
 public:
    explicit cmd_push_vsp(std::int32_t adjust)
        : base_command(command_kind::push_vsp), adjust_(adjust) {}
    std::int32_t adjust() const noexcept { return adjust_; }
    std::string debug_str() const override;

 private:
    std::int32_t adjust_;
};

// -----------------------------------------------------------------------------
// cmd_push_vbase_rva : push (VBASE + rva) onto VSP. Used by the lifter to
// materialise an absolute VA from an RVA constant (target of `[rip+disp]`
// memory operands). VBASE is loaded with the runtime image base at vm_enter;
// adding the target's RVA gives the absolute address regardless of ASLR.
//
// Net VSP change: -8 (one push).
// -----------------------------------------------------------------------------
class cmd_push_vbase_rva final : public base_command {
 public:
    explicit cmd_push_vbase_rva(std::uint64_t rva)
        : base_command(command_kind::push_vbase_rva), rva_(rva) {}
    std::uint64_t rva() const noexcept { return rva_; }
    std::string debug_str() const override;

 private:
    std::uint64_t rva_;
};

// -----------------------------------------------------------------------------
// cmd_exit_to_rva : vm_exit body (restore guest GPRs, switch rsp to VSP) then
// native `jmp rel32` (is_call=false) or `call rel32` (is_call=true) to an
// external RVA. The disp32 is patched at link time once the blob's final
// section RVA is known.
//
// For is_call=true, the lifter MUST emit a subsequent cmd_vm_enter so that
// after the native call returns, the VM context is re-established and lifting
// can continue with post-call guest state.
// -----------------------------------------------------------------------------
class cmd_exit_to_rva final : public base_command {
 public:
    cmd_exit_to_rva(std::uint64_t rva, bool is_call)
        : base_command(command_kind::exit_to_rva), rva_(rva), is_call_(is_call) {}
    std::uint64_t rva()     const noexcept { return rva_; }
    bool          is_call() const noexcept { return is_call_; }
    std::string debug_str() const override;

 private:
    std::uint64_t rva_;
    bool          is_call_;
};

// -----------------------------------------------------------------------------
// cmd_exit_to_top : pop a 64-bit target address from VSP, run vm_exit body,
// then emit native `jmp rax` (is_call=false) or `call rax` (is_call=true).
//
// Used for indirect jmp/call (e.g. `jmp [rip+disp]`, `call rcx`, `call [rax+8]`):
// the lifter first materialises the target onto VSP via emit_push_src, then
// emits this command. machine::emit_exit_to_top pops the target into VTEMP0
// and stashes it in the RAX save slot before running vm_exit_body, so when
// the body finishes restoring guest GPRs from save area, RAX = target. Then
// the final `jmp rax` / `call rax` transfers control.
//
// For is_call=true the lifter must follow with cmd_vm_enter (same pattern as
// cmd_exit_to_rva). Clobbers guest RAX (Win64-volatile, safe for call/tail-jmp).
// -----------------------------------------------------------------------------
class cmd_exit_to_top final : public base_command {
 public:
    explicit cmd_exit_to_top(bool is_call)
        : base_command(command_kind::exit_to_top), is_call_(is_call) {}
    bool is_call() const noexcept { return is_call_; }
    std::string debug_str() const override;

 private:
    bool is_call_;
};

// -----------------------------------------------------------------------------
// cmd_xmm_op : pass-through SIMD instruction.
//
// We don't model XMM/YMM/ZMM lanes in the VM stack/save area; instead we emit
// the native SSE/AVX instruction directly inside the blob. The vector register
// state survives vm_enter / vm_exit untouched (we never read or write any of
// the 16 XMM slots), so this preserves all observable side-effects.
//
//   kind == reg_reg  : emit `mnem dst_xmm, src_xmm` verbatim.
//   kind == reg_mem  : pop EA from VSP (decomposed via emit_push_ea), emit
//                      `mnem dst_xmm, [VTEMP0]` at the chosen size_bits.
//   kind == mem_reg  : pop EA from VSP, emit `mnem [VTEMP0], src_xmm`.
//
// `size_bits` is the memory operand width (32/64/128/256 for ss/sd/dqu/dqu256
// etc.). Ignored when both operands are registers (the mnemonic itself carries
// the width through Zydis register class).
//
// `src_xmm == NONE` in mem_reg mode would be a bug; `dst_xmm == NONE` likewise
// in reg_mem mode.
// -----------------------------------------------------------------------------
enum class xmm_op_kind : std::uint8_t {
    reg_reg,
    reg_mem,
    mem_reg,
};

class cmd_xmm_op final : public base_command {
 public:
    cmd_xmm_op(ZydisMnemonic mnem,
               xmm_op_kind   kind,
               ZydisRegister dst,
               ZydisRegister src,
               std::uint16_t size_bits)
        : base_command(command_kind::xmm_op),
          mnem_(mnem), kind_(kind), dst_(dst), src_(src), size_bits_(size_bits) {}
    ZydisMnemonic mnem()      const noexcept { return mnem_; }
    xmm_op_kind   kind()      const noexcept { return kind_; }
    ZydisRegister dst()       const noexcept { return dst_; }
    ZydisRegister src()       const noexcept { return src_; }
    std::uint16_t size_bits() const noexcept { return size_bits_; }
    std::string   debug_str() const override;

 private:
    ZydisMnemonic mnem_;
    xmm_op_kind   kind_;
    ZydisRegister dst_;
    ZydisRegister src_;
    std::uint16_t size_bits_;
};

// -----------------------------------------------------------------------------
// cmd_x86_exec : raw x86 pass-through.
//
// Fallback strategy for instructions the lifter has no IR handler for
// (SSE/AVX/atomics/bit-test/rep-string/idiv/etc.):
//   1) vm_exit_body    -- restore all guest GPRs + XMMs, switch native rsp
//                         to current VSP. The instruction now sees a real
//                         coherent guest CPU state.
//   2) emit raw bytes  -- the original x86 instruction is re-emitted verbatim.
//                         Writes through guest registers / [rsp+N] / [reg+N]
//                         land in the real guest state because vm_exit_body
//                         just installed it.
//   3) pushfq          -- snapshot the native RFLAGS the instruction produced
//                         onto the guest stack.
//   4) vm_enter        -- re-allocate the VM frame, re-save all GPRs + XMMs.
//                         After this the pushed rflags sits at top of VSP.
//   5) pop top into VFLAGS slot -- so subsequent IR ops see correct flags.
//
// The stored bytes are the exact original encoding. RIP-relative operands are
// NOT supported here (the new emit RVA differs from the original): the lifter
// must reject any instruction with a RIP-relative memory operand before
// constructing a cmd_x86_exec. See translator.cpp.
// -----------------------------------------------------------------------------
class cmd_x86_exec final : public base_command {
 public:
    explicit cmd_x86_exec(std::vector<std::uint8_t> bytes)
        : base_command(command_kind::x86_exec), bytes_(std::move(bytes)) {}
    const std::vector<std::uint8_t>& bytes() const noexcept { return bytes_; }
    std::string debug_str() const override;

 private:
    std::vector<std::uint8_t> bytes_;
};

// -----------------------------------------------------------------------------
// cmd_save_flags : snapshot native RFLAGS into the VFLAGS slot.
// Emitted by the lifter immediately after every flag-producing IR op so that
// later cmd_jcc / cmd_cmov / cmd_setcc see the correct x86 flag state.
// -----------------------------------------------------------------------------
class cmd_save_flags final : public base_command {
 public:
    cmd_save_flags() : base_command(command_kind::save_flags) {}
    std::string debug_str() const override { return "save_flags"; }
};

}  // namespace wenaxvm::ir
