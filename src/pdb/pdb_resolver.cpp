#include "wenaxvm/pdb/pdb_resolver.h"

// raw-pdb headers.
#include "PDB.h"
#include "PDB_RawFile.h"
#include "PDB_InfoStream.h"
#include "PDB_DBIStream.h"
#include "PDB_ImageSectionStream.h"
#include "PDB_ModuleInfoStream.h"
#include "PDB_ModuleSymbolStream.h"
#include "PDB_PublicSymbolStream.h"
#include "PDB_CoalescedMSFStream.h"
#include "PDB_SectionContributionStream.h"
#include "PDB_DBITypes.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <vector>

namespace wenaxvm::pdb {

namespace {

// Read entire file into memory. For 14MB PDBs this is fine.
std::vector<std::uint8_t> read_all(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = f.tellg();
    if (sz <= 0) return {};
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(sz));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

}  // namespace

bool pdb_resolver::load(const std::filesystem::path& pdb_path) {
    funcs_.clear();

    static std::vector<std::uint8_t> g_buf;  // keep buffer alive (raw-pdb keeps pointers into it)
    g_buf = read_all(pdb_path);
    if (g_buf.empty()) {
        std::fprintf(stderr, "pdb_resolver: cannot read %s\n", pdb_path.string().c_str());
        return false;
    }

    if (PDB::ValidateFile(g_buf.data(), g_buf.size()) != PDB::ErrorCode::Success) {
        std::fprintf(stderr, "pdb_resolver: ValidateFile failed\n");
        return false;
    }

    static PDB::RawFile raw = PDB::CreateRawFile(g_buf.data());
    if (PDB::HasValidDBIStream(raw) != PDB::ErrorCode::Success) {
        std::fprintf(stderr, "pdb_resolver: no DBI stream\n");
        return false;
    }

    PDB::InfoStream info(raw);
    if (info.UsesDebugFastLink()) {
        std::fprintf(stderr, "pdb_resolver: /DEBUG:FASTLINK not supported\n");
        return false;
    }

    static PDB::DBIStream dbi = PDB::CreateDBIStream(raw);
    if (dbi.HasValidImageSectionStream(raw)    != PDB::ErrorCode::Success ||
        dbi.HasValidSymbolRecordStream(raw)    != PDB::ErrorCode::Success ||
        dbi.HasValidPublicSymbolStream(raw)    != PDB::ErrorCode::Success ||
        dbi.HasValidSectionContributionStream(raw) != PDB::ErrorCode::Success) {
        std::fprintf(stderr, "pdb_resolver: DBI sub-streams missing\n");
        return false;
    }

    const PDB::ImageSectionStream sectionStream = dbi.CreateImageSectionStream(raw);
    const PDB::ModuleInfoStream   moduleStream  = dbi.CreateModuleInfoStream(raw);
    const PDB::CoalescedMSFStream symbolRecords = dbi.CreateSymbolRecordStream(raw);
    const PDB::PublicSymbolStream publicStream  = dbi.CreatePublicSymbolStream(raw);

    // 1. Pull function symbols from each module.
    auto modules = moduleStream.GetModules();
    for (const auto& mod : modules) {
        if (!mod.HasSymbolStream()) continue;
        auto modSymStream = mod.CreateSymbolStream(raw);
        modSymStream.ForEachSymbol([&](const PDB::CodeView::DBI::Record* record) {
            const char*   name = nullptr;
            std::uint32_t rva  = 0;
            std::uint32_t size = 0;
            std::uint16_t section = 0;

            using Kind = PDB::CodeView::DBI::SymbolRecordKind;
            switch (record->header.kind) {
                case Kind::S_GPROC32:
                    name = record->data.S_GPROC32.name;
                    rva  = sectionStream.ConvertSectionOffsetToRVA(
                              record->data.S_GPROC32.section,
                              record->data.S_GPROC32.offset);
                    size = record->data.S_GPROC32.codeSize;
                    section = record->data.S_GPROC32.section;
                    break;
                case Kind::S_LPROC32:
                    name = record->data.S_LPROC32.name;
                    rva  = sectionStream.ConvertSectionOffsetToRVA(
                              record->data.S_LPROC32.section,
                              record->data.S_LPROC32.offset);
                    size = record->data.S_LPROC32.codeSize;
                    section = record->data.S_LPROC32.section;
                    break;
                case Kind::S_GPROC32_ID:
                    name = record->data.S_GPROC32_ID.name;
                    rva  = sectionStream.ConvertSectionOffsetToRVA(
                              record->data.S_GPROC32_ID.section,
                              record->data.S_GPROC32_ID.offset);
                    size = record->data.S_GPROC32_ID.codeSize;
                    section = record->data.S_GPROC32_ID.section;
                    break;
                case Kind::S_LPROC32_ID:
                    name = record->data.S_LPROC32_ID.name;
                    rva  = sectionStream.ConvertSectionOffsetToRVA(
                              record->data.S_LPROC32_ID.section,
                              record->data.S_LPROC32_ID.offset);
                    size = record->data.S_LPROC32_ID.codeSize;
                    section = record->data.S_LPROC32_ID.section;
                    break;
                default:
                    return;
            }

            if (!name || rva == 0) return;
            func_info fi;
            fi.name    = name;
            fi.rva     = rva;
            fi.size    = size;
            fi.section = section;
            funcs_.emplace(fi.name, fi);
        });
    }

    // 2. Add any public symbols not seen above.
    auto hashRecords = publicStream.GetRecords();
    for (const auto& hashRecord : hashRecords) {
        const auto* record = publicStream.GetRecord(symbolRecords, hashRecord);
        if (record->header.kind != PDB::CodeView::DBI::SymbolRecordKind::S_PUB32) {
            continue;
        }
        // Only function-flagged publics.
        if ((PDB_AS_UNDERLYING(record->data.S_PUB32.flags) &
             PDB_AS_UNDERLYING(PDB::CodeView::DBI::PublicSymbolFlags::Function)) == 0) {
            continue;
        }
        std::uint32_t rva = sectionStream.ConvertSectionOffsetToRVA(
            record->data.S_PUB32.section,
            record->data.S_PUB32.offset);
        if (rva == 0) continue;
        const char* name = record->data.S_PUB32.name;
        if (!name) continue;

        // Skip if we already have something with this RVA (module symbol wins).
        bool seen = false;
        for (const auto& kv : funcs_) {
            if (kv.second.rva == rva) { seen = true; break; }
        }
        if (seen) continue;

        func_info fi;
        fi.name    = name;
        fi.rva     = rva;
        fi.size    = 0;  // unknown; M1 sandbox cares only about RVA match
        fi.section = record->data.S_PUB32.section;
        funcs_.emplace(fi.name, fi);
    }

    return true;
}

std::vector<func_info> pdb_resolver::find_by_regex(const std::regex& re) const {
    std::vector<func_info> hits;
    for (const auto& kv : funcs_) {
        if (std::regex_search(kv.first, re)) {
            hits.push_back(kv.second);
        }
    }
    std::sort(hits.begin(), hits.end(),
              [](const func_info& a, const func_info& b){ return a.rva < b.rva; });
    return hits;
}

std::vector<func_info> pdb_resolver::find_by_names(
    const std::vector<std::string>& names,
    std::vector<std::string>* missing) const {
    std::vector<func_info> hits;
    hits.reserve(names.size());
    for (const auto& n : names) {
        auto it = funcs_.find(n);
        if (it == funcs_.end()) {
            if (missing) missing->push_back(n);
            continue;
        }
        hits.push_back(it->second);
    }
    std::sort(hits.begin(), hits.end(),
              [](const func_info& a, const func_info& b){ return a.rva < b.rva; });
    return hits;
}

}  // namespace wenaxvm::pdb
