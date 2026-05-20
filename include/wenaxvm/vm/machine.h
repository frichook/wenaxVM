#pragma once

// VM machine: lifts IR -> native x64 code.
//
// M1 mode: emit one straight-line blob: VEnter prologue + inlined IR ops +
// VExit epilogue + ret. There is no separate bytecode region and no
// threaded-code dispatch -- those are M2/M3 work. The `compiled_region`
// holds the entire blob in `handlers`; `bytecode` is left empty for now.

#include "wenaxvm/codec/encoder.h"
#include "wenaxvm/codec/polycrypt.h"
#include "wenaxvm/ir/commands.h"
#include "wenaxvm/vm/register_manager.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace wenaxvm::vm {

// Post-link fixup: at `patch_at` (offset inside the blob) sits a 4-byte disp32
// slot that the CLI patches AFTER the final section RVA is known. The formula
// for the patched value is:
//     disp32 = target_rva - (section_va + next_pc)
// where `next_pc` is the offset just after the instruction containing the slot
// (RIP-relative addressing semantics).
//
// `target_rva == 0` is the special case used by `lea vbase, [rip+0]` at
// vm_enter, where we want vbase = RuntimeImageBase, i.e. (next_pc_va + disp)
// = ImageBase + 0, i.e. disp = -(section_va + next_pc).
//
// `target_rva != 0` is used by external `jmp/call rel32` from the blob to a
// foreign RVA in the original image (function tail-call / external call).
struct post_link_fixup {
    std::uint64_t patch_at;     // blob-relative offset of the 4-byte slot
    std::uint64_t next_pc;      // blob-relative offset right after the instruction
    std::uint64_t target_rva;   // RVA the disp must compute; 0 for image_base lea
};

struct compiled_region {
    std::vector<std::uint8_t>      handlers;   // entire emitted x64 blob (M1)
    std::vector<std::uint8_t>      bytecode;   // VM bytecode (empty in M1)
    std::uint64_t                  entry_offset{0};  // offset of vm_enter inside `handlers`
    std::vector<post_link_fixup>   fixups;     // patched by CLI after add_section
    // Polymorphic XOR wrapper bookkeeping. Populated by the CLI when it wraps
    // `handlers` with a decryptor stub; `body_off_in_blob` then equals the
    // stub size, and fixups have already been shifted to point past the stub.
    std::optional<wenaxvm::crypt::crypt_record> crypt;
};

class machine {
 public:
    explicit machine(register_manager regs);
    compiled_region compile(const ir::block& blk);

    // EagleVM-style shared-stack frame layout (M3 Phase 3A).
    //
    // At vm_enter, native rsp = G (= guest RSP at original function entry).
    // After `sub rsp, total_alloc`, layout (low -> high address):
    //   [rsp + 0)                          - VFLAGS slot (8 bytes)
    //   [rsp + 8 .. rsp + save_top)        - guest GPR save area (128 bytes,
    //                                        16 slots, slot4=RSP)
    //   [rsp + 136 .. rsp + ta)            - VM operand-stack headroom (1024 B)
    //   [rsp + ta] == G                    - VSP initial value (grows down)
    //
    // VREGS = rsp + 8 (slot 0 base). VFLAGS at [VREGS - 8].
    // total_alloc is 8 mod 16 so post-sub rsp is 16-byte aligned (entry rsp is
    // 8-mod-16 due to the call-pushed return address). 1160 mod 16 == 8 OK.
    static constexpr std::size_t vflags_bytes     = 8;              // VFLAGS slot
    static constexpr std::size_t save_area_bytes  = 16 * 8;         // 128, slot 4 (RSP) unused
    static constexpr std::size_t vm_headroom_bytes = 1024;          // VM operand-stack workspace
    static constexpr std::size_t total_alloc      =
        vflags_bytes + save_area_bytes + vm_headroom_bytes;         // 1160 (= 8 mod 16)
    static constexpr std::size_t save_area_off    = vflags_bytes;   // first GPR slot offset from rsp

 private:
    void emit_vm_enter(codec::encode_builder& eb);
    void emit_vm_exit (codec::encode_builder& eb);
    void emit_push_imm    (codec::encode_builder& eb, const ir::cmd_push_imm&     c);
    void emit_push_reg_ctx(codec::encode_builder& eb, const ir::cmd_push_reg_ctx& c);
    void emit_pop_reg_ctx (codec::encode_builder& eb, const ir::cmd_pop_reg_ctx&  c);
    void emit_handler_call(codec::encode_builder& eb, const ir::cmd_handler_call& c);
    void emit_cmp_test    (codec::encode_builder& eb, const ir::cmd_cmp_test&     c);
    void emit_unary       (codec::encode_builder& eb, const ir::cmd_unary&        c);
    void emit_jmp         (codec::encode_builder& eb, const ir::cmd_jmp&          c);
    void emit_jcc         (codec::encode_builder& eb, const ir::cmd_jcc&          c);
    void emit_label       (codec::encode_builder& eb, const ir::cmd_label&        c);
    void emit_cmov        (codec::encode_builder& eb, const ir::cmd_cmov&         c);
    void emit_setcc       (codec::encode_builder& eb, const ir::cmd_setcc&        c);
    void emit_movx        (codec::encode_builder& eb, const ir::cmd_movx&         c);
    void emit_shift       (codec::encode_builder& eb, const ir::cmd_shift&        c);
    void emit_mem_load    (codec::encode_builder& eb, const ir::cmd_load_mem&     c);
    void emit_mem_store   (codec::encode_builder& eb, const ir::cmd_store_mem&    c);
    void emit_save_flags_cmd(codec::encode_builder& eb, const ir::cmd_save_flags& c);
    void emit_push_vsp    (codec::encode_builder& eb, const ir::cmd_push_vsp&    c);
    void emit_push_vbase_rva(codec::encode_builder& eb, const ir::cmd_push_vbase_rva& c);
    void emit_exit_to_rva  (codec::encode_builder& eb, const ir::cmd_exit_to_rva&  c);
    void emit_exit_to_top  (codec::encode_builder& eb, const ir::cmd_exit_to_top&  c);
    void emit_xmm_op       (codec::encode_builder& eb, const ir::cmd_xmm_op&       c);

    // Helper: emit the vm-exit body (restore guest GPRs, switch rsp to current
    // VSP) WITHOUT the final `ret`. Shared between emit_vm_exit (which appends
    // ret) and emit_exit_to_rva (which appends native jmp/call rel32).
    void emit_vm_exit_body(codec::encode_builder& eb);

    // Helper: emit `lea vbase, [rip+0]` and record a post-link fixup so the
    // disp32 resolves to vbase = RuntimeImageBase. Called from emit_vm_enter
    // (including any re-entry vm_enter after an external native call).
    void emit_load_image_base(codec::encode_builder& eb);

    // Flag-stash helpers: store/restore RFLAGS via VFLAGS slot using
    // seto+lahf / add+sahf (no rsp manipulation).
    void emit_save_flags(codec::encode_builder& eb);
    void emit_load_flags(codec::encode_builder& eb);

    // Look up encode_builder label for an IR label id, allocating on demand.
    codec::label_id resolve_label(codec::encode_builder& eb, std::uint32_t ir_id);

    register_manager regs_;

    // IR-label-id -> encode_builder label_id, valid for the duration of one
    // compile() call.
    std::vector<codec::label_id> ir_label_map_;

    // Post-link fixups accumulated during the current compile() call. Drained
    // into the returned compiled_region.
    std::vector<post_link_fixup> pending_fixups_;
};

}  // namespace wenaxvm::vm
