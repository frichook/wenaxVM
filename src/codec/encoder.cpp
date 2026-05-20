#include "wenaxvm/codec/encoder.h"

#include <cassert>
#include <stdexcept>
#include <string>

namespace wenaxvm::codec {

namespace {

// Map Zydis register to encoder operand-size in bytes.
std::uint8_t reg_size_bytes(ZydisRegister r) {
    auto cls = ZydisRegisterGetClass(r);
    switch (cls) {
        case ZYDIS_REGCLASS_GPR8:  return 1;
        case ZYDIS_REGCLASS_GPR16: return 2;
        case ZYDIS_REGCLASS_GPR32: return 4;
        case ZYDIS_REGCLASS_GPR64: return 8;
        case ZYDIS_REGCLASS_XMM:   return 16;
        case ZYDIS_REGCLASS_FLAGS: return 8;
        case ZYDIS_REGCLASS_IP:    return 8;
        default:                   return 0;
    }
}

}  // namespace

encode_builder::encode_builder() = default;

label_id encode_builder::make_label() {
    label_positions_.push_back(static_cast<std::size_t>(-1));
    return label_id{label_positions_.size() - 1};
}

void encode_builder::label(label_id id) {
    assert(id.value < label_positions_.size());
    label_positions_[id.value] = bytes_.size();
}

bool encode_builder::fill_operand(ZydisEncoderOperand& dst,
                                  const operand&       src,
                                  bool&                has_label_fixup) {
    has_label_fixup = false;

    return std::visit([&](auto&& o) -> bool {
        using T = std::decay_t<decltype(o)>;
        if constexpr (std::is_same_v<T, reg_op>) {
            dst.type      = ZYDIS_OPERAND_TYPE_REGISTER;
            dst.reg.value = o.r;
            return true;
        } else if constexpr (std::is_same_v<T, imm_op>) {
            dst.type      = ZYDIS_OPERAND_TYPE_IMMEDIATE;
            // Zydis validates that the immediate fits in the operand. For
            // sub-64-bit operands we sign-extend the bit pattern so that
            // values like 0xFFFFFFFF (which is -1 in int32) are encoded
            // correctly. Without this, Zydis rejects imm.u >= 0x80000000
            // for 32-bit operands because it doesn't fit in int32.
            std::uint64_t v = o.value;
            std::int64_t  sx;
            switch (o.size_bytes) {
                case 1: sx = static_cast<std::int64_t>(static_cast<std::int8_t>(v & 0xff));  break;
                case 2: sx = static_cast<std::int64_t>(static_cast<std::int16_t>(v & 0xffff)); break;
                case 4: sx = static_cast<std::int64_t>(static_cast<std::int32_t>(v & 0xffffffffu)); break;
                default: sx = static_cast<std::int64_t>(v); break;
            }
            dst.imm.s = sx;
            return true;
        } else if constexpr (std::is_same_v<T, mem_op>) {
            dst.type            = ZYDIS_OPERAND_TYPE_MEMORY;
            dst.mem.base        = o.base;
            dst.mem.index       = o.index;
            // Zydis requires scale == 0 when no index register is present.
            dst.mem.scale       = (o.index == ZYDIS_REGISTER_NONE) ? 0 : o.scale;
            dst.mem.displacement = o.disp;
            dst.mem.size        = static_cast<std::uint16_t>(o.size_bits / 8);
            return true;
        } else if constexpr (std::is_same_v<T, label_ref>) {
            // Caller will encode this differently. We tell it to fill imm 0
            // and schedule a fixup.
            dst.type      = ZYDIS_OPERAND_TYPE_IMMEDIATE;
            dst.imm.s     = 0;
            has_label_fixup = true;
            return true;
        } else {
            return false;
        }
    }, src);
}

encode_builder& encode_builder::make(ZydisMnemonic m) {
    ZydisEncoderRequest req{};
    req.mnemonic       = m;
    req.machine_mode   = ZYDIS_MACHINE_MODE_LONG_64;
    req.operand_count  = 0;
    encode_request(req);
    return *this;
}

encode_builder& encode_builder::make(ZydisMnemonic m, operand op0) {
    ZydisEncoderRequest req{};
    req.mnemonic       = m;
    req.machine_mode   = ZYDIS_MACHINE_MODE_LONG_64;
    req.operand_count  = 1;

    bool fixup_label = false;
    fill_operand(req.operands[0], op0, fixup_label);

    // jmp/call with label_ref -> we will emit and patch rel32.
    if (fixup_label) {
        const auto* lr = std::get_if<label_ref>(&op0);
        // Encode a near jmp/call with rel32=0 then schedule fixup.
        req.branch_type  = (m == ZYDIS_MNEMONIC_CALL) ? ZYDIS_BRANCH_TYPE_NEAR
                                                      : ZYDIS_BRANCH_TYPE_NEAR;
        req.branch_width = ZYDIS_BRANCH_WIDTH_32;
        req.operands[0].imm.s = 0;
        encode_request(req);
        // rel32 sits in the last 4 bytes of the instruction.
        std::size_t patch_at = bytes_.size() - 4;
        fixups_.push_back(fixup{patch_at, bytes_.size(), lr->id, /*rip_lea*/false});
        return *this;
    }
    encode_request(req);
    return *this;
}

encode_builder& encode_builder::make(ZydisMnemonic m, operand op0, operand op1) {
    ZydisEncoderRequest req{};
    req.mnemonic       = m;
    req.machine_mode   = ZYDIS_MACHINE_MODE_LONG_64;
    req.operand_count  = 2;

    bool fixup_0 = false, fixup_1 = false;
    fill_operand(req.operands[0], op0, fixup_0);
    fill_operand(req.operands[1], op1, fixup_1);

    // Special case: lea reg, [rip + label]   (used to load VBASE).
    if (fixup_1) {
        const auto* lr = std::get_if<label_ref>(&op1);
        // Override second operand to be a memory operand with rip base.
        req.operands[1].type      = ZYDIS_OPERAND_TYPE_MEMORY;
        req.operands[1].mem.base  = ZYDIS_REGISTER_RIP;
        req.operands[1].mem.index = ZYDIS_REGISTER_NONE;
        req.operands[1].mem.scale = 0;
        req.operands[1].mem.displacement = 0;
        req.operands[1].mem.size  = 8;

        std::size_t start_pc = bytes_.size();
        encode_request(req);
        // The disp32 lives somewhere in the encoded instruction. For
        // `lea r64, [rip+disp32]` MSVC-style encoding it is the last 4 bytes.
        std::size_t patch_at = bytes_.size() - 4;
        fixups_.push_back(fixup{patch_at, bytes_.size(), lr->id, /*rip_lea*/true});
        (void)start_pc;
        return *this;
    }

    if (fixup_0) {
        throw std::runtime_error("encode_builder: label as first operand not supported");
    }

    // If imm operand has explicit width smaller than 8, hint the encoder.
    if (auto p = std::get_if<imm_op>(&op1)) {
        if (p->size_bytes == 1) {
            req.operand_size_hint = ZYDIS_OPERAND_SIZE_HINT_8;
        } else if (p->size_bytes == 2) {
            req.operand_size_hint = ZYDIS_OPERAND_SIZE_HINT_16;
        } else if (p->size_bytes == 4) {
            req.operand_size_hint = ZYDIS_OPERAND_SIZE_HINT_32;
        }
    }

    encode_request(req);
    return *this;
}

encode_builder& encode_builder::make(ZydisMnemonic m, operand op0, operand op1, operand op2) {
    ZydisEncoderRequest req{};
    req.mnemonic       = m;
    req.machine_mode   = ZYDIS_MACHINE_MODE_LONG_64;
    req.operand_count  = 3;

    bool fx0=false, fx1=false, fx2=false;
    fill_operand(req.operands[0], op0, fx0);
    fill_operand(req.operands[1], op1, fx1);
    fill_operand(req.operands[2], op2, fx2);
    if (fx0 || fx1 || fx2) {
        throw std::runtime_error("encode_builder: 3-operand label not supported");
    }
    encode_request(req);
    return *this;
}

void encode_builder::encode_request(ZydisEncoderRequest& req) {
    std::uint8_t buf[ZYDIS_MAX_INSTRUCTION_LENGTH]{};
    ZyanUSize    len = sizeof(buf);
    auto status = ZydisEncoderEncodeInstruction(&req, buf, &len);
    if (!ZYAN_SUCCESS(status)) {
        std::string msg = "ZydisEncoderEncodeInstruction failed (mnemonic="
                          + std::to_string(int(req.mnemonic))
                          + ", status=0x" ;
        char hex[16];
        std::snprintf(hex, sizeof(hex), "%08X", static_cast<unsigned>(status));
        msg += hex;
        msg += ", ops=" + std::to_string(req.operand_count);
        for (std::uint8_t i = 0; i < req.operand_count; ++i) {
            const auto& op = req.operands[i];
            msg += " [#" + std::to_string(i) + " type=" + std::to_string(int(op.type));
            if (op.type == ZYDIS_OPERAND_TYPE_REGISTER) {
                msg += " reg=" + std::to_string(int(op.reg.value));
            } else if (op.type == ZYDIS_OPERAND_TYPE_MEMORY) {
                msg += " base=" + std::to_string(int(op.mem.base));
                msg += " idx="  + std::to_string(int(op.mem.index));
                msg += " sc="   + std::to_string(int(op.mem.scale));
                msg += " sz="   + std::to_string(int(op.mem.size));
                msg += " disp=" + std::to_string(op.mem.displacement);
            } else if (op.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
                msg += " imm=" + std::to_string(op.imm.u);
            }
            msg += "]";
        }
        msg += " hint=" + std::to_string(int(req.operand_size_hint));
        msg += ")";
        throw std::runtime_error(msg);
    }
    bytes_.insert(bytes_.end(), buf, buf + len);
}

std::vector<std::uint8_t> encode_builder::finalize() {
    for (auto& fx : fixups_) {
        if (fx.target.value >= label_positions_.size()) {
            throw std::runtime_error("encode_builder: undefined label");
        }
        std::size_t tgt = label_positions_[fx.target.value];
        if (tgt == static_cast<std::size_t>(-1)) {
            throw std::runtime_error("encode_builder: unbound label");
        }
        std::int64_t rel = static_cast<std::int64_t>(tgt) -
                           static_cast<std::int64_t>(fx.next_pc);
        if (rel < INT32_MIN || rel > INT32_MAX) {
            throw std::runtime_error("encode_builder: rel32 out of range");
        }
        std::int32_t rel32 = static_cast<std::int32_t>(rel);
        std::memcpy(bytes_.data() + fx.patch_at, &rel32, 4);
    }
    fixups_.clear();
    return std::move(bytes_);
}

}  // namespace wenaxvm::codec
