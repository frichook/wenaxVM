// wenaxvm-cli -- M1 pipeline driver.
//
//   wenaxvm-cli <input.exe> --regex <PATTERN> --out <out.exe>
//                          [--pdb <input.pdb>] [--seed 0x...]
//                          [--dump-ir <file>]  [--dump-bytecode <file>]
//
// For each PDB function matching --regex we:
//   1. read its raw bytes from the input PE
//   2. linear-decode to a basic_block (M1)
//   3. lift x86 -> IR
//   4. compile IR -> handler blob (VEnter + inlined ops + VExit)
//   5. append it to a new section in the PE
//   6. overwrite the original bytes with `jmp rel32` to the new section
//
// In M1 we only handle the first match and skip the rest.

#include "wenaxvm/codec/disasm.h"
#include "wenaxvm/codec/encoder.h"
#include "wenaxvm/codec/polycrypt.h"
#include "wenaxvm/disasm/cfg.h"
#include "wenaxvm/ir/commands.h"
#include "wenaxvm/ir/translator.h"
#include "wenaxvm/pdb/crt_blacklist.h"
#include "wenaxvm/pdb/pdb_resolver.h"
#include "wenaxvm/pe/pe_deny.h"
#include "wenaxvm/pe/pe_patcher.h"
#include "wenaxvm/version.h"
#include "wenaxvm/vm/machine.h"
#include "wenaxvm/vm/register_manager.h"

#include <Zydis/Zydis.h>
#include <linuxpe>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <regex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct cli_args {
    fs::path        input;
    fs::path        pdb;            // optional override; default <input>.pdb
    fs::path        output;
    std::string     regex_pattern;
    fs::path        only_list;      // --virt-only-functions <list.txt>
    fs::path        deny_list;      // --virt-deny-names <list.txt>
    bool            no_auto_deny{false};  // disable PE-metadata + CRT deny
    bool            no_polycrypt{false};  // disable per-function XOR wrapper
    std::uint64_t   seed{0x1337};
    fs::path        dump_ir;        // optional
    fs::path        dump_bc;        // optional
    bool            dry_run{false}; // coverage scan only, no PE write
};

// Read a name list: one symbol per line. Blank lines and lines beginning with
// '#' (comments) are ignored. Trailing CR/whitespace stripped.
std::vector<std::string> read_name_list(const fs::path& p) {
    std::vector<std::string> out;
    std::ifstream f(p);
    if (!f) return out;
    std::string line;
    while (std::getline(f, line)) {
        // strip CR + trailing whitespace
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' ||
                                 line.back() == '\t'))
            line.pop_back();
        // strip leading whitespace
        std::size_t a = 0;
        while (a < line.size() && (line[a] == ' ' || line[a] == '\t')) ++a;
        if (a > 0) line.erase(0, a);
        if (line.empty() || line.front() == '#') continue;
        out.push_back(std::move(line));
    }
    return out;
}

bool parse_args(int argc, char** argv, cli_args& out) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: wenaxvm-cli <input.exe> (--regex PATTERN | "
            "--virt-only-functions LIST.txt) --out OUT.exe "
            "[--pdb PATH] [--seed 0xN] [--dump-ir FILE] [--dump-bytecode FILE]\n"
            "\n"
            "  --regex PATTERN        virtualise every PDB symbol whose name\n"
            "                         matches PATTERN (broad selector).\n"
            "  --virt-only-functions  exact-name allow-list; one symbol per\n"
            "                         line, blank/'#' lines ignored. Default-\n"
            "                         deny: nothing else is touched.\n"
            "  --virt-deny-names FILE additional exact-name deny list applied\n"
            "                         on top of --regex.\n"
            "  --no-auto-deny         disable PE-metadata + CRT auto-deny\n"
            "                         (only affects --regex mode).\n");
        return false;
    }
    out.input = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string_view a = argv[i];
        auto next = [&](const char* opt) -> const char* {
            if (++i >= argc) {
                std::fprintf(stderr, "missing value for %s\n", opt);
                return nullptr;
            }
            return argv[i];
        };
        if (a == "--regex") {
            auto v = next("--regex"); if (!v) return false;
            out.regex_pattern = v;
        } else if (a == "--out") {
            auto v = next("--out"); if (!v) return false;
            out.output = v;
        } else if (a == "--pdb") {
            auto v = next("--pdb"); if (!v) return false;
            out.pdb = v;
        } else if (a == "--seed") {
            auto v = next("--seed"); if (!v) return false;
            out.seed = std::strtoull(v, nullptr, 0);
        } else if (a == "--virt-only-functions") {
            auto v = next("--virt-only-functions"); if (!v) return false;
            out.only_list = v;
        } else if (a == "--virt-deny-names") {
            auto v = next("--virt-deny-names"); if (!v) return false;
            out.deny_list = v;
        } else if (a == "--no-auto-deny") {
            out.no_auto_deny = true;
        } else if (a == "--no-polycrypt") {
            out.no_polycrypt = true;
        } else if (a == "--dump-ir") {
            auto v = next("--dump-ir"); if (!v) return false;
            out.dump_ir = v;
        } else if (a == "--dump-bytecode") {
            auto v = next("--dump-bytecode"); if (!v) return false;
            out.dump_bc = v;
        } else if (a == "--dry-run") {
            out.dry_run = true;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return false;
        }
    }
    if (out.regex_pattern.empty() && out.only_list.empty()) {
        std::fprintf(stderr,
            "selector required: pass --regex PATTERN or "
            "--virt-only-functions LIST.txt\n");
        return false;
    }
    if (!out.regex_pattern.empty() && !out.only_list.empty()) {
        std::fprintf(stderr,
            "--regex and --virt-only-functions are mutually exclusive\n");
        return false;
    }
    if (out.output.empty() && !out.dry_run) {
        std::fprintf(stderr, "--out is required (unless --dry-run)\n");
        return false;
    }
    if (out.pdb.empty()) {
        out.pdb = out.input;
        out.pdb.replace_extension(".pdb");
    }
    return true;
}


void dump_ir(const fs::path& path, const wenaxvm::ir::block& blk) {
    if (path.empty()) return;
    std::ofstream f(path, std::ios::trunc);
    for (const auto& c : blk) f << c->debug_str() << '\n';
}

void dump_bc(const fs::path& path, std::span<const std::uint8_t> bytes) {
    if (path.empty()) return;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        char hex[4];
        std::snprintf(hex, sizeof(hex), "%02X ", bytes[i]);
        f.write(hex, 3);
        if ((i & 15) == 15) f.put('\n');
    }
}

}  // namespace

int main(int argc, char** argv) {
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
    spdlog::set_level(spdlog::level::debug);
    spdlog::info("wenaxvm-cli {} -- M1 pipeline", wenaxvm::kVersion);

    cli_args args;
    if (!parse_args(argc, argv, args)) return 1;

    spdlog::info("input  : {}", args.input.string());
    spdlog::info("pdb    : {}", args.pdb.string());
    spdlog::info("output : {}", args.output.string());
    if (!args.regex_pattern.empty())
        spdlog::info("regex  : {}", args.regex_pattern);
    if (!args.only_list.empty())
        spdlog::info("only   : {}", args.only_list.string());
    spdlog::info("seed   : 0x{:x}", args.seed);

    // --- Load PE ----------------------------------------------------------
    wenaxvm::pe::pe_patcher patcher;
    if (!patcher.load(args.input)) return 2;

    // --- Load PDB ---------------------------------------------------------
    wenaxvm::pdb::pdb_resolver pdb;
    if (!pdb.load(args.pdb)) return 3;
    spdlog::info("pdb: loaded {} symbols", pdb.all().size());

    // --- Find target function --------------------------------------------
    // Two mutually exclusive selectors:
    //   --regex                 : broad pattern match across PDB
    //   --virt-only-functions   : exact-name allow-list (opt-in, default-deny;
    //                             this is the VMProtect/EagleVM/covirt model)
    std::vector<wenaxvm::pdb::func_info> matches;
    if (!args.only_list.empty()) {
        auto names = read_name_list(args.only_list);
        if (names.empty()) {
            spdlog::error("name list {} is empty / unreadable",
                          args.only_list.string());
            return 5;
        }
        spdlog::info("allow-list: {} names", names.size());
        std::vector<std::string> missing;
        matches = pdb.find_by_names(names, &missing);
        for (const auto& n : missing) {
            spdlog::warn("allow-list: '{}' not found in PDB", n);
        }
        if (matches.empty()) {
            spdlog::error("no allow-list names resolved against PDB");
            return 5;
        }
    } else {
        std::regex re;
        try { re = std::regex(args.regex_pattern); }
        catch (const std::exception& e) {
            spdlog::error("bad regex: {}", e.what());
            return 4;
        }
        matches = pdb.find_by_regex(re);
        if (matches.empty()) {
            spdlog::error("no PDB symbols matched /{}/.", args.regex_pattern);
            return 5;
        }
    }

    // --- Build AUTO-DENY sets (regex mode only) --------------------------
    // --virt-only-functions is already explicit opt-in; trust it.
    if (!args.regex_pattern.empty()) {
        std::unordered_set<std::uint32_t> deny_rvas;
        std::unordered_set<std::string>   deny_names_user;

        if (!args.no_auto_deny) {
            // 1. PE metadata deny set: entry_point, TLS callbacks, EH handlers,
            //    SecurityCookie, GuardCF* (incl. GuardCFFunctionTable entries).
            //    Read the PE file once into bytes for the walker.
            std::ifstream fpe(args.input, std::ios::binary | std::ios::ate);
            if (fpe) {
                auto sz = fpe.tellg();
                std::vector<std::uint8_t> buf(static_cast<std::size_t>(sz));
                fpe.seekg(0);
                fpe.read(reinterpret_cast<char*>(buf.data()), sz);
                auto ds = wenaxvm::pe::collect_pe_denies(buf);
                spdlog::info("auto-deny PE: entry={} tls={} eh={} cfg={} lc={}"
                             " ({} unique rvas)",
                             ds.entry_point, ds.tls_callbacks, ds.eh_handlers,
                             ds.cfg_entries, ds.load_config_singletons,
                             ds.rvas.size());
                deny_rvas = std::move(ds.rvas);
            }
        } else {
            spdlog::warn("--no-auto-deny: PE+CRT auto-deny disabled");
        }

        if (!args.deny_list.empty()) {
            auto names = read_name_list(args.deny_list);
            spdlog::info("user deny-list: {} names from {}",
                         names.size(), args.deny_list.string());
            for (auto& n : names) deny_names_user.insert(std::move(n));
        }

        // Filter matches.
        std::vector<wenaxvm::pdb::func_info> kept;
        kept.reserve(matches.size());
        std::size_t skip_pe = 0, skip_crt = 0, skip_user = 0;
        for (const auto& m : matches) {
            if (deny_rvas.count(m.rva))                  { ++skip_pe;   continue; }
            if (!args.no_auto_deny &&
                wenaxvm::pdb::is_crt_blacklisted(m.name)) {
                ++skip_crt;
                spdlog::debug("auto-deny CRT: {} @ 0x{:x}", m.name, m.rva);
                continue;
            }
            if (deny_names_user.count(m.name))           { ++skip_user; continue; }
            kept.push_back(m);
        }
        spdlog::info("filter: matched={} kept={} (skipped pe={} crt={} user={})",
                     matches.size(), kept.size(),
                     skip_pe, skip_crt, skip_user);
        matches.swap(kept);
        if (matches.empty()) {
            spdlog::error("all matches were denied; nothing to virtualise");
            return 5;
        }
    }
    if (!args.dry_run) {
        for (const auto& m : matches) {
            spdlog::info("match: {} @ rva 0x{:x} size {}", m.name, m.rva, m.size);
        }
    }

    // --- Read the whole PE once -- we need raw bytes per match ----------
    std::vector<std::uint8_t> pe_bytes;
    {
        std::ifstream f(args.input, std::ios::binary | std::ios::ate);
        auto sz = f.tellg();
        pe_bytes.resize(static_cast<std::size_t>(sz));
        f.seekg(0);
        f.read(reinterpret_cast<char*>(pe_bytes.data()), sz);
    }
    auto* full_img = reinterpret_cast<const win::image_x64_t*>(pe_bytes.data());

    // --- Resolve CFG fptr RVAs from LoadConfig --------------------------
    // MSVC emits `call qword ptr [__guard_dispatch_icall_fptr]` as the
    // indirect-call dispatcher idiom; the slot holds `_guard_dispatch_icall_nop`
    // (= `jmp rax`). Lifting that as the generic indirect-mem path clobbers
    // guest RAX with the dispatcher's address, causing an infinite jmp-rax
    // loop. The translator special-cases these two slots and rewrites the
    // call to use guest RAX directly. See translator.cpp lift_call.
    wenaxvm::ir::translator_config tr_cfg{};
    if (auto* lc = full_img->get_directory(win::directory_entry_load_config)) {
        const auto* cfg = full_img->rva_to_ptr<
            const win::load_config_directory_x64_t>(lc->rva, lc->size);
        if (cfg) {
            const auto* nt = full_img->get_nt_headers();
            std::uint64_t image_base = nt ? nt->optional_header.image_base : 0;
            auto va_to_rva = [&](std::uint64_t va) -> std::uint32_t {
                if (!image_base || va < image_base) return 0;
                std::uint64_t off = va - image_base;
                return off > 0xFFFFFFFFull
                    ? 0u
                    : static_cast<std::uint32_t>(off);
            };
            tr_cfg.cfg_check_fptr_rva =
                va_to_rva(cfg->guard_cf_check_function_ptr);
            tr_cfg.cfg_dispatch_fptr_rva =
                va_to_rva(cfg->guard_cf_dispatch_function_ptr);
        }
    }
    if (tr_cfg.cfg_check_fptr_rva || tr_cfg.cfg_dispatch_fptr_rva) {
        spdlog::info("CFG fptr slots: check=0x{:x} dispatch=0x{:x}",
                     tr_cfg.cfg_check_fptr_rva, tr_cfg.cfg_dispatch_fptr_rva);
    } else {
        spdlog::info("CFG fptr slots: not present (LoadConfig missing or unset)");
    }

    // --- Coverage scan (dry-run): try lift on every match ---------------
    if (args.dry_run) {
        spdlog::info("dry-run: scanning {} symbols", matches.size());
        std::unordered_map<std::string, std::size_t> bucket;
        std::size_t pass = 0, fail = 0, skipped = 0;
        std::size_t total_insn = 0, lifted_insn = 0;
        wenaxvm::ir::translator tr(tr_cfg);
        // Optional artifact: a flat list of names that DID lift, plus their
        // size and instruction count. Useful for building --virt-only-functions
        // whitelists without re-walking the PDB by hand. Written to
        // <dump-ir>.lifted.txt if --dump-ir is provided.
        std::vector<std::tuple<std::string, std::uint32_t, std::size_t>> lifted_names;

        for (const auto& m : matches) {
            if (m.size == 0) { ++bucket["[unknown_size]"]; ++skipped; continue; }
            const auto* p = full_img->rva_to_ptr<const std::uint8_t>(m.rva, m.size);
            if (!p)         { ++bucket["[unmapped_rva]"]; ++skipped; continue; }

            std::span<const std::uint8_t> sp(p, m.size);
            auto fv = wenaxvm::disasm::decode_function(sp, m.rva);
            auto blk = tr.lift(fv);
            total_insn += fv.instructions.size();

            if (!blk.empty()) {
                ++pass;
                lifted_insn += fv.instructions.size();
                lifted_names.emplace_back(m.name, m.size, fv.instructions.size());
                continue;
            }
            ++fail;
            const auto& f = tr.last_failure();
            switch (f.reason) {
                case wenaxvm::ir::fail_info::undecodable:
                    ++bucket["[undecodable]"]; break;
                case wenaxvm::ir::fail_info::empty_function:
                    ++bucket["[empty]"]; break;
                case wenaxvm::ir::fail_info::unsupported_insn:
                    ++bucket[ZydisMnemonicGetString(f.mnemonic)]; break;
                default:
                    ++bucket["[unknown_reason]"]; break;
            }
        }

        std::vector<std::pair<std::string, std::size_t>> sorted(bucket.begin(),
                                                                bucket.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](auto& a, auto& b){ return a.second > b.second; });

        spdlog::info("---- coverage summary ----");
        spdlog::info("total matches : {}", matches.size());
        spdlog::info("lifted        : {} ({:.1f}%)",
                     pass, 100.0 * pass / std::max<std::size_t>(matches.size(), 1));
        spdlog::info("failed        : {}", fail);
        spdlog::info("skipped       : {} (no size / unmapped rva)", skipped);
        spdlog::info("instructions  : {} total, {} in lifted functions",
                     total_insn, lifted_insn);
        spdlog::info("CFG peepholes : dispatch={} check={}",
                     wenaxvm::ir::g_cfg_dispatch_peephole_hits.load(
                         std::memory_order_relaxed),
                     wenaxvm::ir::g_cfg_check_peephole_hits.load(
                         std::memory_order_relaxed));
        spdlog::info("---- top blockers ----");
        std::size_t shown = 0;
        for (auto& kv : sorted) {
            if (shown++ >= 30) break;
            spdlog::info("  {:>6}  {}", kv.second, kv.first);
        }
        if (!args.dump_ir.empty()) {
            // Sort by instruction count ascending: small leaf-like functions
            // first. Good starting candidates for --virt-only-functions.
            std::sort(lifted_names.begin(), lifted_names.end(),
                      [](const auto& a, const auto& b){
                          return std::get<2>(a) < std::get<2>(b);
                      });
            fs::path lp = args.dump_ir; lp += ".lifted.txt";
            std::ofstream lf(lp, std::ios::trunc);
            lf << "# name  size_bytes  insn_count\n";
            for (const auto& [n, sz, ic] : lifted_names) {
                lf << n << "\t" << sz << "\t" << ic << "\n";
            }
            spdlog::info("wrote lifted-name list: {}", lp.string());
        }
        return 0;
    }

    // -------------------------------------------------------------------
    // Batch virtualisation: process every match, accumulate handler blobs
    // into a single `.wenax0` section, then patch a 5-byte rel32 trampoline
    // at each successfully-lifted function. Failures (unsupported insn,
    // size < 5, etc.) are logged and skipped -- not fatal.
    // -------------------------------------------------------------------
    wenaxvm::vm::register_manager rm(args.seed);
    spdlog::info("register shuffle: vip={} vsp={} vregs={} vcs={} vcsret={} "
                 "vbase={} vflags={} vtemp0={} vtemp1={} vtempf={}",
                 ZydisRegisterGetString(rm.gpr(wenaxvm::vm::role::vip)),
                 ZydisRegisterGetString(rm.gpr(wenaxvm::vm::role::vsp)),
                 ZydisRegisterGetString(rm.gpr(wenaxvm::vm::role::vregs)),
                 ZydisRegisterGetString(rm.gpr(wenaxvm::vm::role::vcs)),
                 ZydisRegisterGetString(rm.gpr(wenaxvm::vm::role::vcsret)),
                 ZydisRegisterGetString(rm.gpr(wenaxvm::vm::role::vbase)),
                 ZydisRegisterGetString(rm.gpr(wenaxvm::vm::role::vflags)),
                 ZydisRegisterGetString(rm.gpr(wenaxvm::vm::role::vtemp0)),
                 ZydisRegisterGetString(rm.gpr(wenaxvm::vm::role::vtemp1)),
                 ZydisRegisterGetString(rm.gpr(wenaxvm::vm::role::vtempf)));

    struct compiled_match {
        std::string   name;
        std::uint32_t rva;
        std::uint32_t size;
        std::size_t   blob_offset;   // offset inside the concatenated section
    };
    std::vector<compiled_match>             compiled;
    std::vector<std::uint8_t>               section_blob;
    // Section-local fixups: per blob, machine returns blob-relative offsets;
    // we shift by blob_offset so they point into the concatenated section.
    std::vector<wenaxvm::vm::post_link_fixup> all_fixups;
    // Per-function encrypted regions, in section-local coords. Used to
    // re-encrypt disp32 fixup bytes that land inside an encrypted body, so
    // that when the runtime decryptor runs the patched body resolves to the
    // correct disp32 plaintext.
    struct crypt_range_local {
        std::size_t                          blob_off;   // section-local start of stub
        std::uint64_t                        body_off;   // section-local start of encrypted body
        std::uint64_t                        body_size;  // bytes (multiple of 8)
        std::vector<wenaxvm::crypt::block_op> ops;       // per-block forward op
    };
    std::vector<crypt_range_local>          crypt_ranges;
    wenaxvm::ir::translator     tr(tr_cfg);
    wenaxvm::vm::machine        m(rm);

    std::size_t skipped_small = 0, skipped_unmapped = 0;
    std::size_t skipped_decode = 0, skipped_lift = 0, skipped_compile = 0;

    // Build a sorted-by-rva index of all PDB symbols. We use it as the upper
    // bound for the "padding-probe" fallback that lets us replace a function
    // smaller than 5 bytes with a 5-byte `jmp rel32` by extending the patch
    // window into trailing 0xCC / 0x90 alignment padding.
    std::vector<std::uint32_t> all_rvas;
    all_rvas.reserve(pdb.all().size());
    for (const auto& kv : pdb.all()) all_rvas.push_back(kv.second.rva);
    std::sort(all_rvas.begin(), all_rvas.end());
    auto next_symbol_rva = [&](std::uint32_t rva) -> std::uint32_t {
        auto it = std::upper_bound(all_rvas.begin(), all_rvas.end(), rva);
        return (it != all_rvas.end()) ? *it : (rva + 0x100);
    };
    auto effective_patch_size = [&](std::uint32_t rva,
                                    std::uint32_t declared) -> std::uint32_t {
        if (declared >= 5) return declared;
        std::uint32_t need = 5 - declared;
        std::uint32_t limit = next_symbol_rva(rva);
        if (rva + declared >= limit) return declared;
        std::uint32_t avail = limit - (rva + declared);
        if (avail < need) return declared;
        const auto* tail = full_img->rva_to_ptr<const std::uint8_t>(
            rva + declared, avail);
        if (!tail) return declared;
        std::uint32_t extra = 0;
        while (extra < avail && (tail[extra] == 0xCC || tail[extra] == 0x90)) {
            ++extra;
            if (declared + extra >= 5) break;
        }
        return declared + extra;
    };

    for (const auto& target : matches) {
        if (target.size == 0) { ++skipped_unmapped; continue; }
        std::uint32_t patch_size = effective_patch_size(target.rva, target.size);
        if (patch_size < 5) {
            spdlog::warn("skip {} @ 0x{:x}: only {} bytes (no usable padding), "
                         "can't fit jmp rel32",
                         target.name, target.rva, target.size);
            ++skipped_small; continue;
        }
        if (patch_size > target.size) {
            spdlog::debug("padding-probe extended {} @ 0x{:x} from {} to {} bytes",
                          target.name, target.rva, target.size, patch_size);
        }
        const auto* code_ptr = full_img->rva_to_ptr<const std::uint8_t>(
            target.rva, target.size);
        if (!code_ptr) { ++skipped_unmapped; continue; }

        std::span<const std::uint8_t> code_span(code_ptr, target.size);
        auto fv = wenaxvm::disasm::decode_function(code_span, target.rva);
        if (!fv.complete) {
            spdlog::warn("skip {} @ 0x{:x}: undecodable body", target.name, target.rva);
            ++skipped_decode; continue;
        }
        auto ir_blk = tr.lift(fv);
        if (ir_blk.empty()) {
            const auto& f = tr.last_failure();
            const char* why = "?";
            switch (f.reason) {
                case wenaxvm::ir::fail_info::undecodable:     why = "undecodable"; break;
                case wenaxvm::ir::fail_info::empty_function:  why = "empty"; break;
                case wenaxvm::ir::fail_info::unsupported_insn:
                    why = ZydisMnemonicGetString(f.mnemonic); break;
                default:                                       why = "?"; break;
            }
            spdlog::warn("skip {} @ 0x{:x}: lifter rejected ({})",
                         target.name, target.rva, why);
            ++skipped_lift; continue;
        }

        wenaxvm::vm::compiled_region region;
        try {
            region = m.compile(ir_blk);
        } catch (const std::exception& e) {
            spdlog::warn("skip {} @ 0x{:x}: machine::compile threw ({})",
                         target.name, target.rva, e.what());
            ++skipped_compile; continue;
        }

        // Optional dump for the FIRST successful match (BEFORE wrapping --
        // we want the clear bytecode for debugging).
        if (compiled.empty()) {
            dump_ir(args.dump_ir, ir_blk);
            dump_bc(args.dump_bc, region.handlers);
        }

        // Per-function polymorphic XOR wrap. Deterministic per (seed, rva)
        // so re-runs of the same binary produce the same output.
        if (!args.no_polycrypt) {
            try {
                std::mt19937_64 fn_rng(args.seed ^
                                       static_cast<std::uint64_t>(target.rva));
                auto wrapped = wenaxvm::crypt::wrap_blob(region.handlers, fn_rng);
                const std::uint64_t stub_size = wrapped.record.body_off_in_blob;
                // Shift every fixup so it still points at the encrypted body.
                for (auto& fx : region.fixups) {
                    fx.patch_at += stub_size;
                    fx.next_pc  += stub_size;
                }
                region.handlers     = std::move(wrapped.bytes);
                region.entry_offset = 0;  // stub is the new entry
                region.crypt        = wrapped.record;
            } catch (const std::exception& e) {
                spdlog::warn("skip {} @ 0x{:x}: polycrypt wrap threw ({})",
                             target.name, target.rva, e.what());
                ++skipped_compile; continue;
            }
        }

        std::size_t off = section_blob.size();
        section_blob.insert(section_blob.end(),
                            region.handlers.begin(), region.handlers.end());
        // Shift each fixup's blob-local offsets to section-local positions.
        for (auto fx : region.fixups) {
            fx.patch_at += off;
            fx.next_pc  += off;
            all_fixups.push_back(fx);
        }
        // Translate the crypt record's body offset into section-local coords
        // and remember per-function so the fixup loop can XOR-mask disp32s
        // that land inside the encrypted region.
        compiled.push_back({target.name, target.rva, patch_size, off});
        spdlog::debug("lifted {} @ 0x{:x} -> blob[+0x{:x}] ({} bytes, {} fixups{})",
                      target.name, target.rva, off,
                      region.handlers.size(), region.fixups.size(),
                      region.crypt ? ", crypted" : "");
        if (region.crypt) {
            crypt_ranges.push_back({off,
                                    off + region.crypt->body_off_in_blob,
                                    region.crypt->body_size,
                                    std::move(region.crypt->ops)});
        }
    }

    spdlog::info("---- virtualisation summary ----");
    spdlog::info("total matches      : {}", matches.size());
    spdlog::info("virtualised        : {}", compiled.size());
    spdlog::info("skipped tiny       : {}", skipped_small);
    spdlog::info("skipped unmapped   : {}", skipped_unmapped);
    spdlog::info("skipped decode err : {}", skipped_decode);
    spdlog::info("skipped by lifter  : {}", skipped_lift);
    spdlog::info("skipped by compile : {}", skipped_compile);
    spdlog::info("CFG peepholes fired: dispatch={} check={}",
                 wenaxvm::ir::g_cfg_dispatch_peephole_hits.load(
                     std::memory_order_relaxed),
                 wenaxvm::ir::g_cfg_check_peephole_hits.load(
                     std::memory_order_relaxed));

    if (compiled.empty()) {
        spdlog::error("no functions were virtualised");
        return 11;
    }

    // -------------------------------------------------------------------
    // Append the single concatenated handler blob as `.wenax0`.
    // -------------------------------------------------------------------
    // RWX: code + exec + read + write. WRITE is required because the
    // polymorphic decryptor stub patches the encrypted body in place at first
    // execution. (If --no-polycrypt is set the section technically only needs
    // exec+read, but giving it write doesn't hurt and keeps the layout
    // consistent across runs for debugging.)
    constexpr std::uint32_t kCharCodeExecReadWrite = 0xE0000020u;
    std::uint32_t section_va = patcher.add_section(".wenax0",
                                                   section_blob,
                                                   kCharCodeExecReadWrite);
    if (section_va == 0) {
        spdlog::error("add_section failed");
        return 12;
    }
    spdlog::info("appended .wenax0 at RVA 0x{:x} ({} bytes)",
                 section_va, section_blob.size());

    // -------------------------------------------------------------------
    // Apply post-link fixups: disp32 slots inside the .wenax0 section that
    // depend on the final section RVA. Two flavours:
    //   target_rva == 0 : `lea vbase, [rip+0]` -- patch so vbase = ImageBase.
    //   target_rva != 0 : `jmp/call rel32` from blob to a foreign image RVA.
    //
    // Formula:  disp32 = target_rva - (section_va + next_pc)
    // The CLI writes 4 bytes at section_va + fx.patch_at via pe_patcher::patch_at.
    // -------------------------------------------------------------------
    std::size_t fixups_applied = 0;
    std::size_t fixups_xored   = 0;
    // Binary-searchable index over crypt_ranges sorted by body_off.
    auto find_crypt_range = [&](std::size_t patch_at) -> const crypt_range_local* {
        for (const auto& cr : crypt_ranges) {
            if (patch_at >= cr.body_off &&
                patch_at <  cr.body_off + cr.body_size) {
                return &cr;
            }
        }
        return nullptr;
    };
    for (const auto& fx : all_fixups) {
        std::int64_t section_next_va =
            static_cast<std::int64_t>(section_va) + static_cast<std::int64_t>(fx.next_pc);
        std::int64_t disp =
            static_cast<std::int64_t>(fx.target_rva) - section_next_va;
        if (disp < INT32_MIN || disp > INT32_MAX) {
            spdlog::warn("fixup out of rel32 range (target=0x{:x} next=0x{:x})",
                         fx.target_rva, section_next_va);
            continue;
        }
        std::int32_t disp32 = static_cast<std::int32_t>(disp);
        std::uint8_t bytes4[4];
        std::memcpy(bytes4, &disp32, 4);

        // If this fixup lands inside an encrypted body, we cannot just write
        // the disp32 plaintext: it would be exec'd as plaintext bytes inside
        // an encrypted block, which the runtime decryptor would then garble.
        // Instead, for each cipher block the fixup overlaps, we:
        //   (a) read the currently encrypted 8 bytes,
        //   (b) apply the inverse of the block's op (-> plaintext),
        //   (c) patch the relevant byte(s) inside the plaintext,
        //   (d) apply the forward op (-> new ciphertext),
        //   (e) write all 8 bytes back through the patcher.
        if (const auto* cr = find_crypt_range(fx.patch_at)) {
            const std::uint64_t off_in_body = fx.patch_at - cr->body_off;
            const std::size_t   block_lo    = static_cast<std::size_t>(off_in_body / 8);
            const std::size_t   block_hi    = static_cast<std::size_t>((off_in_body + 4 - 1) / 8);

            for (std::size_t bi = block_lo; bi <= block_hi; ++bi) {
                const std::size_t block_off_in_section = cr->body_off + bi * 8;
                // (a) read current ciphertext from our local section_blob copy
                std::uint64_t v;
                std::memcpy(&v, &section_blob[block_off_in_section], 8);
                // (b) inverse op -> plaintext
                wenaxvm::crypt::apply_inverse(v, cr->ops[bi]);

                std::uint8_t plain[8];
                std::memcpy(plain, &v, 8);
                // (c) patch within this block: clip the fixup window to the
                //     block boundaries.
                const std::size_t patch_lo_sec = fx.patch_at;
                const std::size_t patch_hi_sec = fx.patch_at + 4;
                const std::size_t blk_lo_sec   = block_off_in_section;
                const std::size_t blk_hi_sec   = block_off_in_section + 8;
                const std::size_t lo = std::max(patch_lo_sec, blk_lo_sec);
                const std::size_t hi = std::min(patch_hi_sec, blk_hi_sec);
                for (std::size_t off = lo; off < hi; ++off) {
                    plain[off - blk_lo_sec] = bytes4[off - fx.patch_at];
                }
                std::memcpy(&v, plain, 8);
                // (d) forward op -> new ciphertext
                wenaxvm::crypt::apply_forward(v, cr->ops[bi]);

                std::uint8_t encrypted[8];
                std::memcpy(encrypted, &v, 8);
                // Keep our local section_blob mirror in sync (in case a later
                // fixup lands in the same block).
                std::memcpy(&section_blob[block_off_in_section], encrypted, 8);
                // (e) write back through the patcher.
                const std::uint32_t block_rva =
                    static_cast<std::uint32_t>(section_va + block_off_in_section);
                if (!patcher.patch_at(block_rva,
                                      std::span<const std::uint8_t>(encrypted, 8))) {
                    spdlog::warn("patch_at failed for crypt block at rva 0x{:x}",
                                 block_rva);
                }
            }
            ++fixups_xored;
            ++fixups_applied;
            continue;  // bytes4 path already done for this fixup
        }

        std::uint32_t patch_rva =
            static_cast<std::uint32_t>(section_va + fx.patch_at);
        if (!patcher.patch_at(patch_rva, std::span<const std::uint8_t>(bytes4, 4))) {
            spdlog::warn("patch_at failed for fixup at rva 0x{:x}", patch_rva);
            continue;
        }
        ++fixups_applied;
    }
    spdlog::info("applied post-link fixups: {}/{} ({} re-encrypted in-body)",
                 fixups_applied, all_fixups.size(), fixups_xored);

    // -------------------------------------------------------------------
    // Patch trampolines.
    // -------------------------------------------------------------------
    std::size_t patched = 0;
    for (const auto& cm : compiled) {
        std::uint64_t target_va = section_va + cm.blob_offset;
        std::int64_t rel = static_cast<std::int64_t>(target_va)
                         - static_cast<std::int64_t>(cm.rva + 5);
        if (rel < INT32_MIN || rel > INT32_MAX) {
            spdlog::warn("rel32 out of range for {} -- skipped patch", cm.name);
            continue;
        }
        std::vector<std::uint8_t> tramp(cm.size, 0xCC);
        tramp[0] = 0xE9;
        std::int32_t rel32 = static_cast<std::int32_t>(rel);
        std::memcpy(tramp.data() + 1, &rel32, 4);
        if (!patcher.patch_at(cm.rva, tramp)) {
            spdlog::warn("patch_at failed for {}", cm.name);
            continue;
        }
        ++patched;
    }
    spdlog::info("patched trampolines: {}/{}", patched, compiled.size());

    if (!patcher.save(args.output)) {
        spdlog::error("save failed");
        return 14;
    }
    spdlog::info("wrote {}", args.output.string());
    return 0;
}
