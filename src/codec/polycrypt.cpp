// Polymorphic block cipher + decryptor stub. See include/wenaxvm/codec/polycrypt.h.
//
// Algorithm (Shoggoth-style, see frkngksl/Shoggoth/src/SecondEncryption.cpp):
//   for each 8-byte block b_i:
//       pick op_i = random element of { Xor, Add, Sub, Not, Neg, Inc, Dec, Rol, Ror }
//       pick random operand (imm32 sign-ext / imm64 via temp reg / rot count 1..63)
//       b_i := forward(b_i, op_i)
//   decryptor stub:
//       LEA  r_addr, [rip + body_start]
//       CMP  byte [r_addr - 1], 0           ; sentinel (idempotency)
//       JNZ  done
//       MOV  byte [r_addr - 1], 1           ; raise sentinel BEFORE inverse pass
//       for each block i:
//           emit inverse(op_i) acting on [r_addr + i*8]
//       done:
//       JMP  body_start                     ; body is now plaintext
//       db   0x00                           ; sentinel byte (initial)
//       body_start:
//       <encrypted body>
//
// Constraints inherited from the surrounding VM:
//   * Trampoline enters via `jmp rel32` -- all guest GPRs are unsaved. The
//     body's first instruction is `sub rsp, total_alloc` followed by the
//     guest-GPR save area population by vm_enter.
//   * Win64 ABI integer arg regs (RCX/RDX/R8/R9) and callee-save regs
//     (RBX/RBP/RSI/RDI/R12-R15) MUST NOT be touched by the stub.
//   * Safe scratch pool: {RAX, R10, R11} -- caller-clobber non-arg regs that
//     the ABI guarantees the caller has either saved or considers dead.

#include "wenaxvm/codec/polycrypt.h"

#include <array>
#include <cstring>
#include <stdexcept>

namespace wenaxvm::crypt {

namespace {

using wenaxvm::codec::encode_builder;
using wenaxvm::codec::imm;
using wenaxvm::codec::lbl;
using wenaxvm::codec::mem;
using wenaxvm::codec::reg;

constexpr std::array<ZydisRegister, 3> kScratchPool = {
    ZYDIS_REGISTER_RAX, ZYDIS_REGISTER_R10, ZYDIS_REGISTER_R11,
};

std::uint64_t rotl64(std::uint64_t v, unsigned n) {
    n &= 63;
    return n == 0 ? v : ((v << n) | (v >> (64 - n)));
}
std::uint64_t rotr64(std::uint64_t v, unsigned n) {
    n &= 63;
    return n == 0 ? v : ((v >> n) | (v << (64 - n)));
}

// Sign-extend a 32-bit immediate (held in the low 32 bits of `raw`) to 64 bits.
// This matches the runtime semantics of `OP r/m64, imm32`.
std::uint64_t sign_extend_imm32(std::uint64_t raw) {
    return static_cast<std::uint64_t>(
        static_cast<std::int64_t>(
            static_cast<std::int32_t>(raw & 0xFFFFFFFFu)));
}

// The actual immediate that `apply_forward / apply_inverse` should use for
// Xor/Add/Sub. Hides the imm32-vs-imm64 split.
std::uint64_t effective_imm(const block_op& op) {
    return op.use_reg_form ? op.imm : sign_extend_imm32(op.imm);
}

block_op pick_block_op(std::mt19937_64& rng) {
    block_op op{};
    std::uniform_int_distribution<int> kind_d(0, 8);
    op.kind = static_cast<poly_op_kind>(kind_d(rng));
    std::uniform_int_distribution<int> bool_d(0, 1);

    switch (op.kind) {
        case poly_op_kind::Xor:
        case poly_op_kind::Add:
        case poly_op_kind::Sub:
            op.use_reg_form = (bool_d(rng) != 0);
            // Always populate `imm` with a full 64-bit value; the imm32 path
            // just truncates+sign-extends at use time.
            op.imm = rng();
            break;
        case poly_op_kind::Not:
        case poly_op_kind::Neg:
        case poly_op_kind::Inc:
        case poly_op_kind::Dec:
            op.use_reg_form = (bool_d(rng) != 0);
            op.imm = 0;
            break;
        case poly_op_kind::Rol:
        case poly_op_kind::Ror: {
            std::uniform_int_distribution<unsigned> rot_d(1, 63);
            op.use_reg_form = false;
            op.imm = static_cast<std::uint64_t>(rot_d(rng));
        } break;
    }
    return op;
}

// Pick any scratch reg different from `r_addr`. Used for the temp-mediated
// (use_reg_form=true) variants.
ZydisRegister pick_temp_other_than(std::mt19937_64& rng, ZydisRegister r_addr) {
    std::array<ZydisRegister, 2> others{};
    std::size_t k = 0;
    for (auto r : kScratchPool) {
        if (r != r_addr) others[k++] = r;
    }
    std::uniform_int_distribution<std::size_t> d(0, k - 1);
    return others[d(rng)];
}

// Emit the inverse of `op` on the qword at `[r_addr + disp]`.
void emit_inverse_op(encode_builder& eb,
                     ZydisRegister   r_addr,
                     std::int64_t    disp,
                     const block_op& op,
                     ZydisRegister   temp) {
    const auto m64 = [&] { return mem(r_addr, disp, 64); };

    switch (op.kind) {
        case poly_op_kind::Xor:                              // inverse: XOR
            if (op.use_reg_form) {
                eb.make(ZYDIS_MNEMONIC_MOV, reg(temp), imm(op.imm, 8));
                eb.make(ZYDIS_MNEMONIC_XOR, m64(), reg(temp));
            } else {
                eb.make(ZYDIS_MNEMONIC_XOR, m64(),
                        imm(op.imm & 0xFFFFFFFFu, 4));
            }
            break;

        case poly_op_kind::Add:                              // inverse: SUB
            if (op.use_reg_form) {
                eb.make(ZYDIS_MNEMONIC_MOV, reg(temp), imm(op.imm, 8));
                eb.make(ZYDIS_MNEMONIC_SUB, m64(), reg(temp));
            } else {
                eb.make(ZYDIS_MNEMONIC_SUB, m64(),
                        imm(op.imm & 0xFFFFFFFFu, 4));
            }
            break;

        case poly_op_kind::Sub:                              // inverse: ADD
            if (op.use_reg_form) {
                eb.make(ZYDIS_MNEMONIC_MOV, reg(temp), imm(op.imm, 8));
                eb.make(ZYDIS_MNEMONIC_ADD, m64(), reg(temp));
            } else {
                eb.make(ZYDIS_MNEMONIC_ADD, m64(),
                        imm(op.imm & 0xFFFFFFFFu, 4));
            }
            break;

        case poly_op_kind::Not:                              // inverse: NOT
            if (op.use_reg_form) {
                eb.make(ZYDIS_MNEMONIC_MOV, reg(temp), m64());
                eb.make(ZYDIS_MNEMONIC_NOT, reg(temp));
                eb.make(ZYDIS_MNEMONIC_MOV, m64(), reg(temp));
            } else {
                eb.make(ZYDIS_MNEMONIC_NOT, m64());
            }
            break;

        case poly_op_kind::Neg:                              // inverse: NEG
            if (op.use_reg_form) {
                eb.make(ZYDIS_MNEMONIC_MOV, reg(temp), m64());
                eb.make(ZYDIS_MNEMONIC_NEG, reg(temp));
                eb.make(ZYDIS_MNEMONIC_MOV, m64(), reg(temp));
            } else {
                eb.make(ZYDIS_MNEMONIC_NEG, m64());
            }
            break;

        case poly_op_kind::Inc:                              // inverse: DEC
            if (op.use_reg_form) {
                eb.make(ZYDIS_MNEMONIC_MOV, reg(temp), m64());
                eb.make(ZYDIS_MNEMONIC_DEC, reg(temp));
                eb.make(ZYDIS_MNEMONIC_MOV, m64(), reg(temp));
            } else {
                eb.make(ZYDIS_MNEMONIC_DEC, m64());
            }
            break;

        case poly_op_kind::Dec:                              // inverse: INC
            if (op.use_reg_form) {
                eb.make(ZYDIS_MNEMONIC_MOV, reg(temp), m64());
                eb.make(ZYDIS_MNEMONIC_INC, reg(temp));
                eb.make(ZYDIS_MNEMONIC_MOV, m64(), reg(temp));
            } else {
                eb.make(ZYDIS_MNEMONIC_INC, m64());
            }
            break;

        case poly_op_kind::Rol:                              // inverse: ROR
            eb.make(ZYDIS_MNEMONIC_ROR, m64(),
                    imm(op.imm & 0x3Fu, 1));
            break;

        case poly_op_kind::Ror:                              // inverse: ROL
            eb.make(ZYDIS_MNEMONIC_ROL, m64(),
                    imm(op.imm & 0x3Fu, 1));
            break;
    }
}

}  // namespace

void apply_forward(std::uint64_t& v, const block_op& op) {
    switch (op.kind) {
        case poly_op_kind::Xor: v ^= effective_imm(op);                              break;
        case poly_op_kind::Add: v += effective_imm(op);                              break;
        case poly_op_kind::Sub: v -= effective_imm(op);                              break;
        case poly_op_kind::Not: v  = ~v;                                             break;
        case poly_op_kind::Neg: v  = static_cast<std::uint64_t>(
                                        -static_cast<std::int64_t>(v));              break;
        case poly_op_kind::Inc: v += 1;                                              break;
        case poly_op_kind::Dec: v -= 1;                                              break;
        case poly_op_kind::Rol: v  = rotl64(v, static_cast<unsigned>(op.imm));       break;
        case poly_op_kind::Ror: v  = rotr64(v, static_cast<unsigned>(op.imm));       break;
    }
}

void apply_inverse(std::uint64_t& v, const block_op& op) {
    switch (op.kind) {
        case poly_op_kind::Xor: v ^= effective_imm(op);                              break;  // self-inverse
        case poly_op_kind::Add: v -= effective_imm(op);                              break;
        case poly_op_kind::Sub: v += effective_imm(op);                              break;
        case poly_op_kind::Not: v  = ~v;                                             break;  // self-inverse
        case poly_op_kind::Neg: v  = static_cast<std::uint64_t>(
                                        -static_cast<std::int64_t>(v));              break;  // self-inverse
        case poly_op_kind::Inc: v -= 1;                                              break;
        case poly_op_kind::Dec: v += 1;                                              break;
        case poly_op_kind::Rol: v  = rotr64(v, static_cast<unsigned>(op.imm));       break;
        case poly_op_kind::Ror: v  = rotl64(v, static_cast<unsigned>(op.imm));       break;
    }
}

wrapped_blob wrap_blob(std::span<const std::uint8_t> body, std::mt19937_64& rng) {
    if (body.empty()) {
        throw std::runtime_error("wrap_blob: empty body");
    }

    // 1. Pad body to a multiple of 8 with NOPs. Padding bytes never execute
    //    (the body ends with `ret` long before the tail), they exist only to
    //    round up to a whole number of cipher blocks.
    std::vector<std::uint8_t> padded(body.begin(), body.end());
    while ((padded.size() & 7) != 0) padded.push_back(0x90);
    const std::uint64_t body_size_padded = padded.size();
    const std::uint64_t body_qwords      = body_size_padded / 8;

    // 2. Pick per-block ops.
    std::vector<block_op> ops(body_qwords);
    for (auto& op : ops) op = pick_block_op(rng);

    // 3. Forward-encrypt every block.
    for (std::uint64_t i = 0; i < body_qwords; ++i) {
        std::uint64_t v;
        std::memcpy(&v, padded.data() + i * 8, 8);
        apply_forward(v, ops[i]);
        std::memcpy(padded.data() + i * 8, &v, 8);
    }

    // 4. Pick the pointer-holding scratch register.
    std::uniform_int_distribution<std::size_t> rdist(0, kScratchPool.size() - 1);
    const ZydisRegister r_addr = kScratchPool[rdist(rng)];

    // 5. Build the stub.
    encode_builder eb;
    auto body_start = eb.make_label();
    auto done       = eb.make_label();

    // r_addr <- &body_start (RIP-relative LEA, fixed up by encode_builder).
    eb.make(ZYDIS_MNEMONIC_LEA, reg(r_addr), lbl(body_start, /*rip_lea*/ true));

    // Idempotency check: sentinel byte at [r_addr - 1] starts at 0; after the
    // first run we raise it. Subsequent entries short-circuit to `done`.
    eb.make(ZYDIS_MNEMONIC_CMP, mem(r_addr, -1, 8), imm(0, 1));
    eb.make(ZYDIS_MNEMONIC_JNZ, lbl(done, false));
    eb.make(ZYDIS_MNEMONIC_MOV, mem(r_addr, -1, 8), imm(1, 1));

    // Unrolled inverse pass.
    for (std::uint64_t i = 0; i < body_qwords; ++i) {
        const ZydisRegister temp = pick_temp_other_than(rng, r_addr);
        emit_inverse_op(eb, r_addr, static_cast<std::int64_t>(i * 8), ops[i], temp);
    }

    // Tail: jump into the (now-clear) body.
    eb.label(done);
    eb.make(ZYDIS_MNEMONIC_JMP, lbl(body_start, false));

    // Sentinel byte at offset stub_size - 1, initial value 0x00.
    const std::uint8_t sentinel_init = 0x00;
    eb.append_bytes(&sentinel_init, 1);
    eb.label(body_start);

    auto stub_bytes = eb.finalize();
    const std::uint64_t stub_size = stub_bytes.size();

    // 6. Concatenate stub ++ encrypted body.
    std::vector<std::uint8_t> out;
    out.reserve(stub_bytes.size() + padded.size());
    out.insert(out.end(), stub_bytes.begin(), stub_bytes.end());
    out.insert(out.end(), padded.begin(), padded.end());

    wrapped_blob w;
    w.bytes                   = std::move(out);
    w.record.body_off_in_blob = stub_size;
    w.record.body_size        = body_size_padded;
    w.record.ops              = std::move(ops);
    return w;
}

}  // namespace wenaxvm::crypt
