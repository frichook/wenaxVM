#pragma once

// PE patcher built on linux-pe. Filled in M1.10.

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace wenaxvm::pe {

class pe_patcher {
 public:
    bool load(const std::filesystem::path& pe_path);

    // Append a new section with the given name and contents.
    // Returns the new section RVA, or 0 on failure.
    std::uint32_t add_section(const std::string&            name,
                              std::span<const std::uint8_t> bytes,
                              std::uint32_t                 characteristics);

    // Overwrite bytes at `rva` with `payload`.
    bool patch_at(std::uint32_t                  rva,
                  std::span<const std::uint8_t>  payload);

    bool save(const std::filesystem::path& out_path) const;

 private:
    std::vector<std::uint8_t> image_;
};

}  // namespace wenaxvm::pe
