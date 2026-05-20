#pragma once

// Per-function polymorphic block cipher + decryptor stub.
//
// Model (Shoggoth-style, see frkngksl/Shoggoth/src/SecondEncryption.cpp):
//   * The body is padded to an 8-byte multiple and split into 64-bit blocks.
//   * Each block is encrypted with a single, randomly chosen, REVERSIBLE op
//     from a 9-element pool: { XOR, ADD, SUB, NOT, NEG, INC, DEC, ROL, ROR }.
//   * Op operands are also randomised:
//       - XOR/ADD/SUB pick either a 32-bit immediate (sign-extended to 64 at
//         runtime by the `r/m64, imm32` encoding) OR a full 64-bit immediate
//         pre-loaded into a temp register.
//       - NOT/NEG/INC/DEC pick either the direct r/m64 form or the longer
//         (MOV temp,[mem]; OP temp; MOV [mem],temp) sequence.
//       - ROL/ROR pick a rotation count in 1..63.
//
// The decryptor stub is fully unrolled: one inverse op per block, addressed as
// `[r_addr + i*8]` where r_addr is RIP-LEA'd to body_start. There is NO loop,
// so each function has a structurally unique decryptor whose byte stream is
// a function of the per-block op list.
//
// Idempotency: a sentinel byte sits at `stub_size - 1`, initial value 0x00.
// On entry the stub checks `byte [r_addr - 1] == 0`; if not, it short-circuits
// to the body (already decrypted). On first entry it raises the sentinel to
// 0x01 BEFORE running the inverse ops, so even if the host process is
// interrupted mid-decryption we won't re-encrypt next time.

#include "wenaxvm/codec/encoder.h"

#include <Zydis/Zydis.h>

#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace wenaxvm::crypt {

enum class poly_op_kind : std::uint8_t {
    Xor, Add, Sub, Not, Neg, Inc, Dec, Rol, Ror,
};

// Description of how a single 8-byte block is encrypted. The decryptor emits
// the INVERSE of this (e.g. forward Add -> inverse Sub) on `[r_addr + i*8]`.
struct block_op {
    poly_op_kind  kind{poly_op_kind::Xor};
    // For Xor/Add/Sub:
    //   use_reg_form=false  => `OP r/m64, imm32` (imm32 sign-extended to 64).
    //                          `imm` holds the 32-bit value zero-extended into
    //                          the low half; sign extension is applied at use.
    //   use_reg_form=true   => `MOV temp, imm64 ; OP r/m64, temp`. `imm` holds
    //                          the full 64-bit value.
    // For Not/Neg/Inc/Dec:
    //   use_reg_form=false  => `OP r/m64`.
    //   use_reg_form=true   => `MOV temp,[mem]; OP temp; MOV [mem],temp`.
    // For Rol/Ror:
    //   use_reg_form is unused (always direct). `imm` holds the rotation count
    //   in 1..63.
    bool          use_reg_form{false};
    std::uint64_t imm{0};
};

// Per-function record. The CLI uses this to identify post-link disp32 fixups
// that fall inside an encrypted body and to re-encrypt the affected block(s)
// after writing the resolved disp32 plaintext.
struct crypt_record {
    std::uint64_t          body_off_in_blob{0};   // = stub_size; first byte of encrypted body
    std::uint64_t          body_size{0};          // bytes, multiple of 8
    std::vector<block_op>  ops;                   // ops[i] = forward op for block i
};

struct wrapped_blob {
    std::vector<std::uint8_t> bytes;              // stub ++ encrypted body
    crypt_record              record;
};

// Wrap `body` (the raw vm blob, starting with vm_enter at offset 0) with a
// polymorphic decryptor stub. `rng` controls every randomised choice so the
// caller can derive a deterministic per-function rng from e.g.
// (global_seed ^ function_rva).
wrapped_blob wrap_blob(std::span<const std::uint8_t> body, std::mt19937_64& rng);

// Apply the forward op to one 64-bit block in-place. Used at wrap time AND by
// the CLI when re-encrypting a block after patching a fixup.
void apply_forward(std::uint64_t& v, const block_op& op);

// Apply the inverse op to one 64-bit block in-place. Used by the CLI to peel
// an encrypted block back to plaintext before patching a disp32 inside it.
void apply_inverse(std::uint64_t& v, const block_op& op);

}  // namespace wenaxvm::crypt
