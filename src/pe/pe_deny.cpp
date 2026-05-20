#include "wenaxvm/pe/pe_deny.h"

#include "linuxpe"

#include <cstring>

namespace wenaxvm::pe {

namespace {

// Convert an absolute VA (e.g. SecurityCookie, GuardCFCheckFunctionPointer)
// stored in LoadConfig into an RVA, returning 0 on failure.
std::uint32_t va_to_rva(std::uint64_t va, std::uint64_t image_base) {
    if (va == 0 || va < image_base) return 0;
    std::uint64_t off = va - image_base;
    if (off > 0xFFFFFFFFull) return 0;
    return static_cast<std::uint32_t>(off);
}

// Walk the unwind_info chain at `unwind_rva` and collect any RVA stored in
// `exception_handler_rva()` when UNW_FLAG_EHANDLER or UNW_FLAG_UHANDLER is
// set. Chained entries (UNW_FLAG_CHAININFO) are followed recursively.
void walk_unwind_handler(const win::image_x64_t* img,
                         std::uint32_t           unwind_rva,
                         deny_set&               out,
                         int                     depth = 0) {
    if (depth > 8) return;        // pathological chain guard
    if (unwind_rva == 0) return;
    const auto* uw = img->rva_to_ptr<const win::unwind_info_t>(
        unwind_rva, sizeof(win::unwind_info_t));
    if (!uw) return;

    if (uw->chained) {
        // language-specific data slot contains a runtime_function_t for the
        // parent function; chase it.
        const auto* chain = reinterpret_cast<const win::runtime_function_t*>(
            uw->get_language_specific_data());
        if (chain) {
            walk_unwind_handler(img, chain->rva_unwind_info, out, depth + 1);
        }
        return;
    }
    if (uw->ex_handler || uw->term_handler) {
        std::uint32_t h = uw->exception_handler_rva();
        if (h && out.rvas.insert(h).second) ++out.eh_handlers;
    }
}

}  // namespace

deny_set collect_pe_denies(std::span<const std::uint8_t> image_bytes) {
    deny_set out;
    if (image_bytes.size() < sizeof(win::dos_header_t)) return out;

    const auto* img = reinterpret_cast<const win::image_x64_t*>(image_bytes.data());
    const auto* nt  = img->get_nt_headers();
    if (!nt) return out;
    if (nt->signature != 0x00004550) return out;
    if (nt->optional_header.magic != win::OPT_HDR64_MAGIC) return out;

    const std::uint64_t image_base = nt->optional_header.image_base;

    // 1. Entry point ----------------------------------------------------------
    if (auto ep = nt->optional_header.entry_point; ep != 0) {
        if (out.rvas.insert(ep).second) ++out.entry_point;
    }

    // 2. TLS callbacks --------------------------------------------------------
    if (auto* td = img->get_directory(win::directory_entry_tls)) {
        const auto* tls = img->rva_to_ptr<const win::tls_directory_x64_t>(
            td->rva, td->size);
        if (tls && tls->address_callbacks != 0) {
            // address_callbacks is a VA pointing to a null-terminated array
            // of VAs (each a TLS callback function VA).
            std::uint32_t cb_array_rva = va_to_rva(tls->address_callbacks,
                                                   image_base);
            if (cb_array_rva) {
                // Read up to 32 entries to avoid runaway.
                for (int i = 0; i < 32; ++i) {
                    const auto* p = img->rva_to_ptr<const std::uint64_t>(
                        cb_array_rva + static_cast<std::uint32_t>(i * 8),
                        sizeof(std::uint64_t));
                    if (!p) break;
                    std::uint64_t cb_va;
                    std::memcpy(&cb_va, p, sizeof(cb_va));
                    if (cb_va == 0) break;
                    std::uint32_t cb_rva = va_to_rva(cb_va, image_base);
                    if (cb_rva && out.rvas.insert(cb_rva).second) ++out.tls_callbacks;
                }
            }
        }
    }

    // 3. Exception directory (.pdata) handlers --------------------------------
    if (auto* ed = img->get_directory(win::directory_entry_exception)) {
        const auto* base = img->rva_to_ptr<const win::runtime_function_t>(
            ed->rva, ed->size);
        std::size_t count = ed->size / sizeof(win::runtime_function_t);
        for (std::size_t i = 0; i < count; ++i) {
            walk_unwind_handler(img, base[i].rva_unwind_info, out);
        }
    }

    // 4. Load Config singletons + GuardCF function table ----------------------
    if (auto* lc = img->get_directory(win::directory_entry_load_config)) {
        const auto* cfg = img->rva_to_ptr<const win::load_config_directory_x64_t>(
            lc->rva, lc->size);
        if (cfg) {
            auto try_va = [&](std::uint64_t va) {
                std::uint32_t r = va_to_rva(va, image_base);
                if (r && out.rvas.insert(r).second) ++out.load_config_singletons;
            };
            // Each of these is a VA the loader / CRT touches directly:
            //   security_cookie               -- __security_init_cookie / __security_check_cookie
            //   guard_cf_check_function_ptr   -- __guard_check_icall_fptr  (CFG)
            //   guard_cf_dispatch_function_ptr-- __guard_dispatch_icall_fptr
            try_va(cfg->security_cookie);
            try_va(cfg->guard_cf_check_function_ptr);
            try_va(cfg->guard_cf_dispatch_function_ptr);

            // GuardCFFunctionTable: array of {RVA + N stride-bytes} of every
            // address-taken function the loader is allowed to call indirectly.
            // EVERY ENTRY HERE IS AN INDIRECT-CALL TARGET — virtualising any
            // of them via a 5-byte jmp trampoline breaks CFG.
            std::uint64_t tbl_va = cfg->guard_cf_function_table.virtual_address;
            std::uint64_t tbl_n  = cfg->guard_cf_function_table.count;
            if (tbl_va && tbl_n) {
                std::uint32_t tbl_rva = va_to_rva(tbl_va, image_base);
                if (tbl_rva) {
                    // Stride = 4 + ((guard_flags >> GUARD_CF_FUNCTION_TABLE_SIZE_SHIFT) & 0xF)
                    // GUARD_CF_FUNCTION_TABLE_SIZE_SHIFT = 28.
                    std::uint32_t stride =
                        4 + ((cfg->guard_flags >> 28) & 0xFu);
                    // Cap to avoid pathological counts (10M = 40MB scan).
                    if (tbl_n > 10'000'000ULL) tbl_n = 10'000'000ULL;
                    for (std::uint64_t i = 0; i < tbl_n; ++i) {
                        std::uint32_t off = static_cast<std::uint32_t>(i * stride);
                        const auto* p = img->rva_to_ptr<const std::uint32_t>(
                            tbl_rva + off, sizeof(std::uint32_t));
                        if (!p) break;
                        std::uint32_t fn_rva;
                        std::memcpy(&fn_rva, p, sizeof(fn_rva));
                        if (fn_rva && out.rvas.insert(fn_rva).second) ++out.cfg_entries;
                    }
                }
            }
        }
    }

    return out;
}

}  // namespace wenaxvm::pe
