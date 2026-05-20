#pragma once

// Basic block / CFG builder.

#include "wenaxvm/codec/disasm.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_set>
#include <vector>

namespace wenaxvm::disasm {

// A single basic block (kept for M1 callers; M2 lifter uses function_view).
struct basic_block {
    std::uint64_t                     start_rva{0};
    std::vector<codec::decoded_instr> instructions;
};

// Whole-function decode: linear instruction stream + intra-function branch
// targets. Targets are RVAs of decoded instructions that some jmp/jcc inside
// the function points to. The lifter emits an IR label at each such target.
struct function_view {
    std::uint64_t                      start_rva{0};
    std::vector<codec::decoded_instr>  instructions;       // ordered by RVA
    std::unordered_set<std::uint64_t>  branch_targets;     // RVAs targeted
    bool                               complete{false};    // decoded all bytes
};

// Decode `bytes` starting at `start_rva` as a single linear block until
// the first terminator (ret/jmp). Used by M1 path.
basic_block decode_linear(std::span<const std::uint8_t> bytes,
                          std::uint64_t                 start_rva);

// Decode the entire function body, collecting intra-function branch targets.
// Skips 0xCC padding between decoded chunks.
function_view decode_function(std::span<const std::uint8_t> bytes,
                              std::uint64_t                 start_rva);

}  // namespace wenaxvm::disasm
