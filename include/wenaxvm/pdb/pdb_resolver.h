#pragma once

// PDB symbol resolver via raw-pdb. Filled in M1.9.

#include <cstdint>
#include <filesystem>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

namespace wenaxvm::pdb {

struct func_info {
    std::string   name;
    std::uint32_t rva{0};
    std::uint32_t size{0};
    std::uint16_t section{0};
};

class pdb_resolver {
 public:
    bool load(const std::filesystem::path& pdb_path);
    std::vector<func_info> find_by_regex(const std::regex& re) const;
    // Exact-name lookup. Missing names are reported via `missing` (if non-null).
    // The opt-in equivalent of VMProtect's per-MapFunction set_compilation_type:
    // ONLY the listed names are eligible for virtualisation, nothing else.
    std::vector<func_info> find_by_names(
        const std::vector<std::string>& names,
        std::vector<std::string>* missing = nullptr) const;
    const std::unordered_map<std::string, func_info>& all() const noexcept { return funcs_; }

 private:
    std::unordered_map<std::string, func_info> funcs_;
};

}  // namespace wenaxvm::pdb
