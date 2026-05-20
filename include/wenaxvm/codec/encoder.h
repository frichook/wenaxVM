#pragma once

// Fluent Zydis encoder. Two-phase builder: callers append `encoded_op` records
// (mnemonic + operand list + optional label fixups), then `materialise()` does
// one greedy pass that picks the smallest Zydis encoding for each op, resolves
// internal label rel32/rel8 displacements, and emits the final byte stream.
//
// Usage:
//     encode_builder b;
//     b.make(ZYDIS_MNEMONIC_MOV, reg(R15), reg(RAX));
//     auto lbl = b.make_label();
//     b.label(lbl);
//     b.make(ZYDIS_MNEMONIC_JMP, label_ref(lbl));
//     auto bytes = b.finalize();           // returns linked byte vector

#include <Zydis/Zydis.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

namespace wenaxvm::codec {

// -----------------------------------------------------------------------------
// Operand builders
// -----------------------------------------------------------------------------
struct reg_op {
    ZydisRegister r;
    explicit reg_op(ZydisRegister rg) : r(rg) {}
};

struct imm_op {
    std::uint64_t value;
    std::uint8_t  size_bytes;  // 1/2/4/8
    imm_op(std::uint64_t v, std::uint8_t bytes = 8) : value(v), size_bytes(bytes) {}
};

struct mem_op {
    ZydisRegister base{ZYDIS_REGISTER_NONE};
    ZydisRegister index{ZYDIS_REGISTER_NONE};
    std::uint8_t  scale{1};
    std::int64_t  disp{0};
    std::uint16_t size_bits{64};
};

struct label_id { std::size_t value; };

struct label_ref {
    label_id id;
    bool     rip_relative_lea{false};  // emit `lea reg, [rip+lbl]` instead of jmp
};

using operand = std::variant<reg_op, imm_op, mem_op, label_ref>;

// -----------------------------------------------------------------------------
// encode_builder
// -----------------------------------------------------------------------------
class encode_builder {
 public:
    encode_builder();

    label_id make_label();

    // Bind a label to the current emit cursor.
    void label(label_id id);

    // Emit an instruction. Variadic operands; up to 4 are typed straight.
    encode_builder& make(ZydisMnemonic m);
    encode_builder& make(ZydisMnemonic m, operand op0);
    encode_builder& make(ZydisMnemonic m, operand op0, operand op1);
    encode_builder& make(ZydisMnemonic m, operand op0, operand op1, operand op2);

    // Returns linked bytes. Patches all label refs as rel32 (jmps) or
    // 32-bit RIP-relative LEAs.
    std::vector<std::uint8_t> finalize();

    // Address of the current emit cursor inside the future linked buffer.
    std::size_t cursor() const noexcept { return bytes_.size(); }

    // Append already-encoded bytes verbatim (used for ad-hoc snippets).
    void append_bytes(const std::uint8_t* data, std::size_t n) {
        bytes_.insert(bytes_.end(), data, data + n);
    }

 private:
    struct fixup {
        std::size_t  patch_at;      // offset of imm32 in bytes_
        std::size_t  next_pc;       // offset right after the instruction
        label_id     target;
        bool         rip_lea;       // lea  reg, [rip+disp32]
    };

    void encode_request(ZydisEncoderRequest& req);
    bool fill_operand(ZydisEncoderOperand& dst, const operand& src, bool& has_label_fixup);

    std::vector<std::uint8_t> bytes_;
    std::vector<fixup>        fixups_;
    std::vector<std::size_t>  label_positions_;  // index = label_id.value
    std::size_t               pending_label_pc_{static_cast<std::size_t>(-1)};
};

// -----------------------------------------------------------------------------
// Operand shortcut helpers (so call sites stay compact)
// -----------------------------------------------------------------------------
inline reg_op reg(ZydisRegister r)            { return reg_op{r}; }
inline imm_op imm(std::uint64_t v, std::uint8_t bytes = 8) { return imm_op{v, bytes}; }
inline mem_op mem(ZydisRegister base, std::int64_t disp = 0, std::uint16_t bits = 64) {
    mem_op m; m.base = base; m.disp = disp; m.size_bits = bits; return m;
}
inline mem_op mem_idx(ZydisRegister base, ZydisRegister index, std::uint8_t scale,
                      std::int64_t disp = 0, std::uint16_t bits = 64) {
    mem_op m; m.base = base; m.index = index; m.scale = scale;
    m.disp = disp; m.size_bits = bits; return m;
}
inline label_ref lbl(label_id id, bool rip_lea = false) {
    return label_ref{id, rip_lea};
}

}  // namespace wenaxvm::codec
