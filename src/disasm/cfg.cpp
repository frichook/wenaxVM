#include "wenaxvm/disasm/cfg.h"

#include <Zydis/Zydis.h>

namespace wenaxvm::disasm {

basic_block decode_linear(std::span<const std::uint8_t> bytes,
                          std::uint64_t                 start_rva) {
    basic_block bb;
    bb.start_rva = start_rva;

    std::size_t offset = 0;
    while (offset < bytes.size()) {
        auto here = std::span(bytes.data() + offset, bytes.size() - offset);
        auto decoded = codec::decode_one(here, start_rva + offset);
        if (!decoded) break;

        const auto category = decoded->inst.meta.category;
        bb.instructions.push_back(*decoded);
        offset += decoded->length;

        if (category == ZYDIS_CATEGORY_RET ||
            category == ZYDIS_CATEGORY_UNCOND_BR ||
            category == ZYDIS_CATEGORY_COND_BR) {
            break;
        }
    }
    return bb;
}

namespace {

// Return the absolute RVA of a near/short relative branch's target, or
// nullopt if the operand isn't an in-function relative jump.
std::optional<std::uint64_t> branch_target_rva(const codec::decoded_instr& di) {
    const auto& op = di.operands[0];
    if (op.type != ZYDIS_OPERAND_TYPE_IMMEDIATE) return std::nullopt;
    if (!op.imm.is_relative) return std::nullopt;
    std::uint64_t tgt = 0;
    if (ZYAN_FAILED(ZydisCalcAbsoluteAddress(&di.inst, &op, di.runtime_address, &tgt))) {
        return std::nullopt;
    }
    return tgt;
}

}  // namespace

function_view decode_function(std::span<const std::uint8_t> bytes,
                              std::uint64_t                 start_rva) {
    function_view fv;
    fv.start_rva = start_rva;

    const std::uint64_t end_rva = start_rva + bytes.size();
    std::size_t offset = 0;

    while (offset < bytes.size()) {
        // Skip int3 padding so we don't decode it as a real instruction.
        // (MSVC fills tail padding with CC; some inter-block padding is too.)
        if (bytes[offset] == 0xCC) {
            ++offset;
            continue;
        }
        auto here = std::span(bytes.data() + offset, bytes.size() - offset);
        auto decoded = codec::decode_one(here, start_rva + offset);
        if (!decoded) {
            // Undecodable byte: bail out; caller skips function.
            return fv;
        }

        const auto category = decoded->inst.meta.category;
        if (category == ZYDIS_CATEGORY_COND_BR ||
            category == ZYDIS_CATEGORY_UNCOND_BR) {
            if (auto tgt = branch_target_rva(*decoded)) {
                if (*tgt >= start_rva && *tgt < end_rva) {
                    fv.branch_targets.insert(*tgt);
                }
            }
        }

        fv.instructions.push_back(*decoded);
        offset += decoded->length;
    }
    fv.complete = true;
    return fv;
}

}  // namespace wenaxvm::disasm
