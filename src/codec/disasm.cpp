#include "wenaxvm/codec/disasm.h"

namespace wenaxvm::codec {

namespace {
ZydisDecoder& global_decoder() {
    static ZydisDecoder decoder = [] {
        ZydisDecoder d{};
        ZydisDecoderInit(&d, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
        return d;
    }();
    return decoder;
}
}  // namespace

std::optional<decoded_instr> decode_one(std::span<const std::uint8_t> bytes,
                                        std::uint64_t                runtime_address) {
    decoded_instr out{};
    out.runtime_address = runtime_address;
    auto status = ZydisDecoderDecodeFull(&global_decoder(),
                                         bytes.data(),
                                         bytes.size(),
                                         &out.inst,
                                         out.operands);
    if (!ZYAN_SUCCESS(status)) return std::nullopt;
    out.length = out.inst.length;
    // Copy the raw original bytes so downstream consumers (e.g. cmd_x86_exec
    // pass-through) can re-emit the exact encoding verbatim.
    std::size_t copy_n = out.length;
    if (copy_n > sizeof(out.raw_bytes)) copy_n = sizeof(out.raw_bytes);
    for (std::size_t i = 0; i < copy_n; ++i) out.raw_bytes[i] = bytes[i];
    return out;
}

}  // namespace wenaxvm::codec
