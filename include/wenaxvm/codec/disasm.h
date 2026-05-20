#pragma once

// Thin Zydis decoder wrappers. Filled in M1.3.

#include <Zydis/Zydis.h>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace wenaxvm::codec {

struct decoded_instr {
    ZydisDecodedInstruction inst{};
    ZydisDecodedOperand     operands[ZYDIS_MAX_OPERAND_COUNT]{};
    std::uint64_t           runtime_address{0};
    std::size_t             length{0};
    // Raw original instruction bytes (length bytes are valid). Stored inline
    // so the lifter can emit `cmd_x86_exec` pass-throughs that re-emit the
    // exact original encoding without keeping the source PE buffer alive.
    // x64 instructions are at most ZYDIS_MAX_INSTRUCTION_LENGTH (15) bytes.
    std::uint8_t            raw_bytes[ZYDIS_MAX_INSTRUCTION_LENGTH]{};
};

// Decode a single instruction at the start of `bytes`. Returns nullopt on failure.
std::optional<decoded_instr> decode_one(std::span<const std::uint8_t> bytes,
                                        std::uint64_t                runtime_address);

}  // namespace wenaxvm::codec
