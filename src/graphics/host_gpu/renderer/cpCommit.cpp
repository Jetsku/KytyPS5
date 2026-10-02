#include "graphics/host_gpu/renderer/cpCommit.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/slotVector.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <utility>

namespace Libs::Graphics::CpCommit {

namespace {

constexpr std::array<std::pair<std::string_view, Part>, 8> PartNames {{
    {"dccguest", Part::DccGuest},
    {"targetalloc", Part::TargetAlloc},
    {"streamread", Part::StreamRead},
    {"slots", Part::Slots},
    {"draws", Part::Draws},
    {"texdcc", Part::TexDcc},
    {"bindslots", Part::BindSlots},
    {"metaerase", Part::MetaErase},
}};

uint32_t ParseParts() {
	const auto* value = std::getenv("KYTY_CP_COMMIT");
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	uint32_t result = 0;
	if (std::strcmp(value, "1") == 0 || std::strcmp(value, "all") == 0) {
		result = AllParts;
	} else {
		std::string_view list(value);
		while (!list.empty()) {
			const auto comma = list.find(',');
			const auto name  = list.substr(0, comma);
			list = comma == std::string_view::npos ? std::string_view {} : list.substr(comma + 1);
			if (name.empty()) {
				continue;
			}
			bool known = false;
			for (const auto& [part_name, part]: PartNames) {
				if (name == part_name) {
					result |= static_cast<uint32_t>(part);
					known = true;
				}
			}
			if (!known) {
				EXIT("KYTY_CP_COMMIT: unknown part '%.*s' (expected 0, 1, all or a list of dccguest, "
				     "targetalloc, streamread, slots, draws, texdcc, bindslots, metaerase)\n",
				     static_cast<int>(name.size()), name.data());
			}
		}
	}
	std::printf("KYTY_CP_COMMIT: parts 0x%x\n", result);
	Common::g_slot_vector_dense.store((result & static_cast<uint32_t>(Part::Slots)) != 0,
	                                  std::memory_order_relaxed);
	return result;
}

} // namespace

uint32_t Parts() {
	static const uint32_t parts = ParseParts();
	return parts;
}

} // namespace Libs::Graphics::CpCommit
