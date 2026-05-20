#include "wenaxvm/ir/commands.h"

#include <Zydis/Zydis.h>

#include <format>

namespace wenaxvm::ir {

namespace {
const char* binop_name(binop o) {
    switch (o) {
        case binop::op_add: return "add";
        case binop::op_sub: return "sub";
        case binop::op_xor: return "xor";
        case binop::op_and: return "and";
        case binop::op_or:  return "or";
        case binop::op_mul: return "imul";
    }
    return "?";
}
const char* cmp_name(cmp_kind o) {
    switch (o) {
        case cmp_kind::op_cmp:  return "cmp";
        case cmp_kind::op_test: return "test";
    }
    return "?";
}
const char* unop_name(unop o) {
    switch (o) {
        case unop::op_inc: return "inc";
        case unop::op_dec: return "dec";
        case unop::op_neg: return "neg";
        case unop::op_not:   return "not";
        case unop::op_bswap: return "bswap";
    }
    return "?";
}
const char* movx_name(movx_kind o) {
    switch (o) {
        case movx_kind::zero: return "movzx";
        case movx_kind::sign: return "movsx";
    }
    return "?";
}
const char* shift_name(shift_kind o) {
    switch (o) {
        case shift_kind::shl: return "shl";
        case shift_kind::shr: return "shr";
        case shift_kind::sar: return "sar";
        case shift_kind::rol: return "rol";
        case shift_kind::ror: return "ror";
    }
    return "?";
}
}  // namespace

std::string cmd_push_imm::debug_str() const {
    return std::format("push.imm{} 0x{:x}", bits_of(size_), value_);
}

std::string cmd_push_reg_ctx::debug_str() const {
    return std::format("push.ctx{} {}", bits_of(size_), ZydisRegisterGetString(reg_));
}

std::string cmd_pop_reg_ctx::debug_str() const {
    return std::format("pop.ctx{} {}", bits_of(size_), ZydisRegisterGetString(reg_));
}

std::string cmd_handler_call::debug_str() const {
    return std::format("call.{}.{}", binop_name(op_), bits_of(size_));
}

std::string cmd_cmp_test::debug_str() const {
    return std::format("{}.{}", cmp_name(op_), bits_of(size_));
}

std::string cmd_unary::debug_str() const {
    return std::format("{}.{}", unop_name(op_), bits_of(size_));
}

std::string cmd_label::debug_str() const {
    return std::format("L{}:", id_);
}

std::string cmd_jmp::debug_str() const {
    return std::format("jmp L{}", target_);
}

std::string cmd_jcc::debug_str() const {
    return std::format("{} L{}", ZydisMnemonicGetString(cond_), target_);
}

std::string cmd_cmov::debug_str() const {
    return std::format("{}.{}", ZydisMnemonicGetString(cond_), bits_of(size_));
}

std::string cmd_setcc::debug_str() const {
    return std::format("{}", ZydisMnemonicGetString(cond_));
}

std::string cmd_movx::debug_str() const {
    return std::format("{}.{}->{}", movx_name(kind_),
                       bits_of(src_), bits_of(dst_));
}

std::string cmd_shift::debug_str() const {
    return std::format("{}.{} {}", shift_name(kind_),
                       bits_of(size_), static_cast<unsigned>(count_));
}

std::string cmd_load_mem::debug_str() const {
    return std::format("load.mem{}", bits_of(size_));
}

std::string cmd_store_mem::debug_str() const {
    return std::format("store.mem{}", bits_of(size_));
}

std::string cmd_push_vsp::debug_str() const {
    if (adjust_ == 0) return "push.vsp";
    return std::format("push.vsp+{:#x}", adjust_);
}

std::string cmd_push_vbase_rva::debug_str() const {
    return std::format("push.vbase+{:#x}", rva_);
}

std::string cmd_exit_to_rva::debug_str() const {
    return std::format("{}.exit {:#x}", is_call_ ? "call" : "jmp", rva_);
}

std::string cmd_exit_to_top::debug_str() const {
    return std::format("{}.exit *top", is_call_ ? "call" : "jmp");
}

std::string cmd_x86_exec::debug_str() const {
    return std::format("x86.exec[{} bytes]", bytes_.size());
}

std::string cmd_xmm_op::debug_str() const {
    const char* mn = ZydisMnemonicGetString(mnem_);
    const char* dn = (dst_ != ZYDIS_REGISTER_NONE) ? ZydisRegisterGetString(dst_) : "?";
    const char* sn = (src_ != ZYDIS_REGISTER_NONE) ? ZydisRegisterGetString(src_) : "?";
    switch (kind_) {
        case xmm_op_kind::reg_reg:
            return std::format("{} {}, {}", mn ? mn : "?", dn, sn);
        case xmm_op_kind::reg_mem:
            return std::format("{} {}, [vsp:m{}]", mn ? mn : "?", dn, size_bits_);
        case xmm_op_kind::mem_reg:
            return std::format("{} [vsp:m{}], {}", mn ? mn : "?", size_bits_, sn);
    }
    return "?xmm_op";
}

}  // namespace wenaxvm::ir
