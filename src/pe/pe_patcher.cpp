#include "wenaxvm/pe/pe_patcher.h"

#include "linuxpe"

#include <cstdio>
#include <cstring>
#include <fstream>

namespace wenaxvm::pe {

namespace {

constexpr std::uint32_t align_up(std::uint32_t v, std::uint32_t a) {
    return (v + a - 1u) & ~(a - 1u);
}

}  // namespace

bool pe_patcher::load(const std::filesystem::path& pe_path) {
    image_.clear();
    std::ifstream f(pe_path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "pe_patcher: cannot open %s\n", pe_path.string().c_str());
        return false;
    }
    auto sz = f.tellg();
    if (sz <= 0) return false;
    image_.resize(static_cast<std::size_t>(sz));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(image_.data()), sz);

    // Quick sanity: must be PE32+ (we only support x64 in M1).
    auto* img = reinterpret_cast<win::image_x64_t*>(image_.data());
    auto* nt  = img->get_nt_headers();
    if (nt->signature != 0x00004550 /* "PE\0\0" */) {
        std::fprintf(stderr, "pe_patcher: bad PE signature\n");
        return false;
    }
    if (nt->optional_header.magic != win::OPT_HDR64_MAGIC) {
        std::fprintf(stderr, "pe_patcher: not PE32+\n");
        return false;
    }
    return true;
}

std::uint32_t pe_patcher::add_section(const std::string&            name,
                                      std::span<const std::uint8_t> bytes,
                                      std::uint32_t                 characteristics) {
    if (image_.empty()) return 0;

    // Snapshot the fields we need BEFORE any resize, since resizes
    // invalidate any pointer or reference into image_.
    std::uint32_t file_alignment;
    std::uint32_t section_alignment;
    std::uint32_t size_headers;
    std::uint16_t cur_n;
    std::size_t   shdr_off;
    std::uint32_t end_va = 0;
    {
        auto* img = reinterpret_cast<win::image_x64_t*>(image_.data());
        auto* nt  = img->get_nt_headers();
        auto& oh  = nt->optional_header;
        file_alignment    = oh.file_alignment;
        section_alignment = oh.section_alignment;
        size_headers      = oh.size_headers;
        cur_n             = nt->file_header.num_sections;
        shdr_off = reinterpret_cast<std::uint8_t*>(nt->get_sections()) - image_.data();

        for (std::uint16_t i = 0; i < cur_n; ++i) {
            auto* s = nt->get_section(i);
            std::uint32_t e = s->virtual_address + s->virtual_size;
            if (e > end_va) end_va = e;
        }
    }

    std::size_t new_shdr_end = shdr_off + (cur_n + 1u) * sizeof(win::section_header_t);
    if (new_shdr_end > size_headers) {
        std::fprintf(stderr, "pe_patcher: no room for another section header\n");
        return 0;
    }

    std::uint32_t new_va     = align_up(end_va, section_alignment);
    std::uint32_t new_raw_off = align_up(static_cast<std::uint32_t>(image_.size()),
                                         file_alignment);
    std::uint32_t raw_size   = align_up(static_cast<std::uint32_t>(bytes.size()),
                                        file_alignment);

    // Grow the buffer to hold the new section payload.
    image_.resize(static_cast<std::size_t>(new_raw_off) + raw_size, 0u);
    std::memcpy(image_.data() + new_raw_off, bytes.data(), bytes.size());

    // Re-fetch pointers AFTER resize, then patch headers in-place.
    {
        auto* img = reinterpret_cast<win::image_x64_t*>(image_.data());
        auto* nt  = img->get_nt_headers();

        win::section_header_t sh{};
        std::memset(sh.name.short_name, 0, sizeof(sh.name.short_name));
        std::size_t name_len = std::min<std::size_t>(name.size(), sizeof(sh.name.short_name));
        std::memcpy(sh.name.short_name, name.data(), name_len);
        sh.virtual_size            = static_cast<std::uint32_t>(bytes.size());
        sh.virtual_address         = new_va;
        sh.size_raw_data           = raw_size;
        sh.ptr_raw_data            = new_raw_off;
        sh.ptr_relocs              = 0;
        sh.ptr_line_numbers        = 0;
        sh.num_relocs              = 0;
        sh.num_line_numbers        = 0;
        sh.characteristics.flags   = characteristics;

        auto* sections = nt->get_sections();
        sections[cur_n] = sh;
        nt->file_header.num_sections = cur_n + 1;

        std::uint32_t new_image_size =
            align_up(new_va + sh.virtual_size, section_alignment);
        if (new_image_size > nt->optional_header.size_image) {
            nt->optional_header.size_image = new_image_size;
        }

        img->update_checksum(image_.size());
    }

    return new_va;
}

bool pe_patcher::patch_at(std::uint32_t rva, std::span<const std::uint8_t> payload) {
    if (image_.empty() || payload.empty()) return false;

    auto* img = reinterpret_cast<win::image_x64_t*>(image_.data());
    auto* dst = img->rva_to_ptr<std::uint8_t>(rva, payload.size());
    if (!dst) {
        std::fprintf(stderr, "pe_patcher: rva 0x%X out of range\n", rva);
        return false;
    }
    std::memcpy(dst, payload.data(), payload.size());

    img->update_checksum(image_.size());
    return true;
}

bool pe_patcher::save(const std::filesystem::path& out_path) const {
    std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
    if (!f) {
        std::fprintf(stderr, "pe_patcher: cannot write %s\n", out_path.string().c_str());
        return false;
    }
    f.write(reinterpret_cast<const char*>(image_.data()),
            static_cast<std::streamsize>(image_.size()));
    return true;
}

}  // namespace wenaxvm::pe
