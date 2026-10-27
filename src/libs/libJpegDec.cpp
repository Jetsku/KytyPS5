#include "common/abi.h"
#include "common/logging/log.h"
#include "libs/libs.h"
#include "loader/symbolDatabase.h"
#include "stb_image.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

// libSceJpegDec (QueryMemorySize, Create, Delete, ParseHeader, Decode). Unresolved, its imports
// returned 0 and wrote nothing, so the pictures a game decodes with it came out 0x0 (Gran Turismo
// 7's car dealer photos and menu pictures). The structures follow the PS4 toolchain's jpeg.h; the
// decoding is stb_image's (its implementation is compiled in window.cpp), with the down scales of
// 2, 4 and 8 as box filters. Registered only with KYTY_JPEG=1 (see InitJpegDec_1).
namespace Libs {

LIB_VERSION("JpegDec", 1, "JpegDec", 1, 1);

namespace JpegDec {

constexpr int32_t JPEG_DEC_ERROR_INVALID_ADDR        = -2140864511; // 0x80650001
constexpr int32_t JPEG_DEC_ERROR_INVALID_SIZE        = -2140864510; // 0x80650002
constexpr int32_t JPEG_DEC_ERROR_INVALID_PARAM       = -2140864509; // 0x80650003
constexpr int32_t JPEG_DEC_ERROR_INVALID_HANDLE      = -2140864508; // 0x80650004
constexpr int32_t JPEG_DEC_ERROR_INVALID_WORK_MEMORY = -2140864507; // 0x80650005
constexpr int32_t JPEG_DEC_ERROR_INVALID_DATA        = -2140864496; // 0x80650010
constexpr int32_t JPEG_DEC_ERROR_DECODE_ERROR        = -2140864494; // 0x80650012

constexpr uint16_t JPEG_DEC_COLOR_SPACE_GRAYSCALE = 2;
constexpr uint16_t JPEG_DEC_COLOR_SPACE_RGB       = 3;
constexpr uint16_t JPEG_DEC_PIXEL_FORMAT_R8G8B8A8 = 0;
constexpr uint16_t JPEG_DEC_PIXEL_FORMAT_B8G8R8A8 = 1;
constexpr uint32_t JPEG_DEC_CSC_ORDINARY          = 1;

struct JpegDecCreateParam {
	uint32_t this_size;
	uint32_t attribute;
	uint32_t max_image_width;
};

struct JpegDecParseParam {
	const void* jpeg_mem_addr;
	uint32_t    jpeg_mem_size;
	uint16_t    decode_mode;
	uint16_t    down_scale;
};

struct JpegDecDecodeParam {
	const void* jpeg_mem_addr;
	void*       image_mem_addr;
	void*       coefficient_mem_addr;
	uint32_t    jpeg_mem_size;
	uint32_t    image_mem_size;
	uint32_t    coefficient_mem_size;
	uint16_t    decode_mode;
	uint16_t    down_scale;
	uint16_t    pixel_format;
	uint16_t    alpha_value;
	uint32_t    image_pitch;
};

struct JpegDecImageInfo {
	uint32_t image_width;
	uint32_t image_height;
	uint16_t color_space;
	uint16_t component_number;
	uint8_t  component_id[4];
	uint8_t  sampling_factor[4];
	uint32_t coefficient_mem_size;
	uint32_t suitable_csc_attribute;
	uint32_t output_image_width;
	uint32_t output_image_height;
};

static_assert(sizeof(JpegDecCreateParam) == 0xc);
static_assert(sizeof(JpegDecParseParam) == 0x10);
static_assert(sizeof(JpegDecDecodeParam) == 0x30);
static_assert(sizeof(JpegDecImageInfo) == 0x24);

// In the memory the game gives Create, at its first 8-byte boundary.
struct JpegDecContext {
	uint64_t magic;
	uint32_t attribute;
	uint32_t max_image_width;
};

constexpr uint64_t JPEG_DEC_CONTEXT_MAGIC = 0x4b59545944474a50ull;
constexpr uint32_t JPEG_DEC_MEMORY_SIZE   = 0x40;
constexpr uint32_t JPEG_DEC_LOGGED_CALLS  = 16;

static_assert(sizeof(JpegDecContext) + 7u <= JPEG_DEC_MEMORY_SIZE);

struct JpegHeader {
	uint32_t width       = 0;
	uint32_t height      = 0;
	uint32_t components  = 0;
	bool     progressive = false;
	uint8_t  component_id[4] {};
	uint8_t  sampling_factor[4] {};
};

// The frame header (any SOFn but DHT, JPG and DAC) of a JPEG.
static bool ParseJpegHeader(const void* data, uint32_t size, JpegHeader* header) {
	if (data == nullptr || header == nullptr || size < 4) {
		return false;
	}
	const auto* bytes = static_cast<const uint8_t*>(data);
	if (bytes[0] != 0xff || bytes[1] != 0xd8) {
		return false;
	}
	uint32_t offset = 2;
	while (offset + 4 <= size) {
		if (bytes[offset] != 0xff) {
			return false;
		}
		const uint8_t marker = bytes[offset + 1];
		if (marker == 0xff) {
			offset++; // fill byte
			continue;
		}
		if (marker == 0xd8 || marker == 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
			offset += 2; // markers without a length
			continue;
		}
		if (marker == 0xd9 || marker == 0xda) {
			return false; // the end of the image or a scan before any frame header
		}
		const uint32_t length = (static_cast<uint32_t>(bytes[offset + 2]) << 8u) | bytes[offset + 3];
		if (length < 2 || offset + 2 + length > size) {
			return false;
		}
		const bool is_frame =
		    marker >= 0xc0 && marker <= 0xcf && marker != 0xc4 && marker != 0xc8 && marker != 0xcc;
		if (is_frame) {
			if (length < 8) {
				return false;
			}
			const auto* frame   = bytes + offset + 4;
			header->height      = (static_cast<uint32_t>(frame[1]) << 8u) | frame[2];
			header->width       = (static_cast<uint32_t>(frame[3]) << 8u) | frame[4];
			header->components  = frame[5];
			header->progressive = marker == 0xc2 || marker == 0xc6 || marker == 0xca || marker == 0xce;
			if (header->width == 0 || header->height == 0 || header->components == 0 ||
			    length < 8 + 3 * header->components) {
				return false;
			}
			for (uint32_t i = 0; i < std::min<uint32_t>(header->components, 4); i++) {
				header->component_id[i]    = frame[6 + i * 3];
				header->sampling_factor[i] = frame[7 + i * 3];
			}
			return true;
		}
		offset += 2 + length;
	}
	return false;
}

static uint32_t ScaledSize(uint32_t size, uint32_t scale) {
	return (size + scale - 1) / scale;
}

static uint32_t DownScale(uint16_t scale) {
	return scale == 2 || scale == 4 || scale == 8 ? scale : 1u;
}

static void FillImageInfo(const JpegHeader& header, uint32_t scale, JpegDecImageInfo* info) {
	if (info == nullptr) {
		return;
	}
	*info              = JpegDecImageInfo {};
	info->image_width  = header.width;
	info->image_height = header.height;
	info->color_space  = header.components == 1 ? JPEG_DEC_COLOR_SPACE_GRAYSCALE : JPEG_DEC_COLOR_SPACE_RGB;
	info->component_number = static_cast<uint16_t>(header.components);
	std::memcpy(info->component_id, header.component_id, sizeof(info->component_id));
	std::memcpy(info->sampling_factor, header.sampling_factor, sizeof(info->sampling_factor));
	// The host decoder needs no coefficient buffer.
	info->coefficient_mem_size   = 0;
	info->suitable_csc_attribute = JPEG_DEC_CSC_ORDINARY;
	info->output_image_width     = ScaledSize(header.width, scale);
	info->output_image_height    = ScaledSize(header.height, scale);
}

static JpegDecContext* ContextOf(void* handle) {
	if (handle == nullptr || (reinterpret_cast<uintptr_t>(handle) & 7u) != 0) {
		return nullptr;
	}
	auto* ctx = static_cast<JpegDecContext*>(handle);
	return ctx->magic == JPEG_DEC_CONTEXT_MAGIC ? ctx : nullptr;
}

static bool LogCall(std::atomic<uint32_t>* counter) {
	return counter->fetch_add(1, std::memory_order_relaxed) < JPEG_DEC_LOGGED_CALLS;
}

static int32_t KYTY_SYSV_ABI JpegDecQueryMemorySize(const JpegDecCreateParam* param) {
	PRINT_NAME();

	if (param == nullptr) {
		return JPEG_DEC_ERROR_INVALID_PARAM;
	}

	LOGF("\t this_size       = %" PRIu32 "\n"
	     "\t attribute       = %" PRIu32 "\n"
	     "\t max_image_width = %" PRIu32 "\n",
	     param->this_size, param->attribute, param->max_image_width);

	return static_cast<int32_t>(JPEG_DEC_MEMORY_SIZE);
}

static int32_t KYTY_SYSV_ABI JpegDecCreate(const JpegDecCreateParam* param, void* memory_address,
                                           uint32_t memory_size, void** handle) {
	PRINT_NAME();

	if (param == nullptr) {
		return JPEG_DEC_ERROR_INVALID_PARAM;
	}

	LOGF("\t this_size       = %" PRIu32 "\n"
	     "\t attribute       = %" PRIu32 "\n"
	     "\t max_image_width = %" PRIu32 "\n"
	     "\t memory_address  = %p\n"
	     "\t memory_size     = %" PRIu32 "\n",
	     param->this_size, param->attribute, param->max_image_width, memory_address, memory_size);

	if (memory_address == nullptr || handle == nullptr) {
		return JPEG_DEC_ERROR_INVALID_ADDR;
	}

	const auto address = reinterpret_cast<uintptr_t>(memory_address);
	const auto aligned = (address + 7u) & ~uintptr_t {7u};
	if (aligned - address + sizeof(JpegDecContext) > memory_size) {
		return JPEG_DEC_ERROR_INVALID_WORK_MEMORY;
	}

	auto* ctx            = reinterpret_cast<JpegDecContext*>(aligned);
	ctx->magic           = JPEG_DEC_CONTEXT_MAGIC;
	ctx->attribute       = param->attribute;
	ctx->max_image_width = param->max_image_width;
	*handle              = ctx;

	return 0;
}

static int32_t KYTY_SYSV_ABI JpegDecDelete(void* handle) {
	PRINT_NAME();

	auto* ctx = ContextOf(handle);
	if (ctx == nullptr) {
		return JPEG_DEC_ERROR_INVALID_HANDLE;
	}
	ctx->magic = 0;
	return 0;
}

static int32_t KYTY_SYSV_ABI JpegDecParseHeader(const JpegDecParseParam* param, JpegDecImageInfo* image_info) {
	if (param == nullptr) {
		return JPEG_DEC_ERROR_INVALID_PARAM;
	}
	if (param->jpeg_mem_addr == nullptr || image_info == nullptr) {
		return JPEG_DEC_ERROR_INVALID_ADDR;
	}

	JpegHeader    header {};
	const bool    parsed = ParseJpegHeader(param->jpeg_mem_addr, param->jpeg_mem_size, &header);
	const int32_t result = parsed ? 0 : JPEG_DEC_ERROR_INVALID_DATA;
	if (parsed) {
		FillImageInfo(header, DownScale(param->down_scale), image_info);
	}

	static std::atomic<uint32_t> parses {0};
	static std::atomic<uint32_t> failures {0};
	if (LogCall(parsed ? &parses : &failures)) {
		LOGF("JpegDec: parse header %p, %" PRIu32 " bytes, decode mode %u, down scale %u: %" PRIu32 "x%" PRIu32
		     ", %" PRIu32 " components%s, result 0x%08" PRIx32 "\n",
		     param->jpeg_mem_addr, param->jpeg_mem_size, static_cast<unsigned>(param->decode_mode),
		     static_cast<unsigned>(param->down_scale), header.width, header.height, header.components,
		     header.progressive ? " (progressive)" : "", static_cast<uint32_t>(result));
	}
	return result;
}

static int32_t DecodeImage(void* handle, const JpegDecDecodeParam* param, JpegDecImageInfo* image_info,
                           JpegHeader* header) {
	if (ContextOf(handle) == nullptr) {
		return JPEG_DEC_ERROR_INVALID_HANDLE;
	}
	if (param == nullptr) {
		return JPEG_DEC_ERROR_INVALID_PARAM;
	}
	if (param->jpeg_mem_addr == nullptr || param->image_mem_addr == nullptr) {
		return JPEG_DEC_ERROR_INVALID_ADDR;
	}
	if (param->pixel_format != JPEG_DEC_PIXEL_FORMAT_R8G8B8A8 &&
	    param->pixel_format != JPEG_DEC_PIXEL_FORMAT_B8G8R8A8) {
		return JPEG_DEC_ERROR_INVALID_PARAM;
	}
	if (!ParseJpegHeader(param->jpeg_mem_addr, param->jpeg_mem_size, header)) {
		return JPEG_DEC_ERROR_INVALID_DATA;
	}

	const uint32_t scale      = DownScale(param->down_scale);
	const uint32_t out_width  = ScaledSize(header->width, scale);
	const uint32_t out_height = ScaledSize(header->height, scale);
	const uint64_t min_pitch  = uint64_t {out_width} * 4u;
	const uint64_t pitch      = param->image_pitch == 0 ? min_pitch : param->image_pitch;
	if (pitch < min_pitch) {
		return JPEG_DEC_ERROR_INVALID_PARAM;
	}
	if (pitch * (out_height - 1u) + min_pitch > param->image_mem_size) {
		return JPEG_DEC_ERROR_INVALID_SIZE;
	}

	int   width      = 0;
	int   height     = 0;
	int   components = 0;
	auto* rgba       = stbi_load_from_memory(static_cast<const stbi_uc*>(param->jpeg_mem_addr),
	                                         static_cast<int>(param->jpeg_mem_size), &width, &height, &components, 4);
	if (rgba == nullptr || static_cast<uint32_t>(width) != header->width ||
	    static_cast<uint32_t>(height) != header->height) {
		if (rgba != nullptr) {
			stbi_image_free(rgba);
		}
		return JPEG_DEC_ERROR_DECODE_ERROR;
	}

	// A JPEG has no alpha: every pixel gets the alpha value asked for.
	const auto alpha   = static_cast<uint8_t>(std::min<uint16_t>(param->alpha_value, 255u));
	const bool swap_rb = param->pixel_format == JPEG_DEC_PIXEL_FORMAT_B8G8R8A8;
	auto*      dst_row = static_cast<uint8_t*>(param->image_mem_addr);
	for (uint32_t y = 0; y < out_height; y++, dst_row += pitch) {
		auto*          dst    = dst_row;
		const uint32_t y_end  = std::min(header->height, (y + 1) * scale);
		for (uint32_t x = 0; x < out_width; x++, dst += 4) {
			// With a down scale, the average of the block the output pixel covers.
			const uint32_t x_end = std::min(header->width, (x + 1) * scale);
			uint32_t       sum[3] {};
			uint32_t       count = 0;
			for (uint32_t sy = y * scale; sy < y_end; sy++) {
				const auto* src = rgba + (size_t {sy} * header->width + x * scale) * 4u;
				for (uint32_t sx = x * scale; sx < x_end; sx++, src += 4) {
					sum[0] += src[0];
					sum[1] += src[1];
					sum[2] += src[2];
					count++;
				}
			}
			dst[0] = static_cast<uint8_t>((sum[0] + count / 2) / count);
			dst[1] = static_cast<uint8_t>((sum[1] + count / 2) / count);
			dst[2] = static_cast<uint8_t>((sum[2] + count / 2) / count);
			dst[3] = alpha;
			if (swap_rb) {
				std::swap(dst[0], dst[2]);
			}
		}
	}
	stbi_image_free(rgba);

	FillImageInfo(*header, scale, image_info);

	// As PngDec: the output's width and height packed, when they fit.
	return out_width > 32767u || out_height > 32767u ? 0 : static_cast<int32_t>((out_width << 16u) | out_height);
}

static int32_t KYTY_SYSV_ABI JpegDecDecode(void* handle, const JpegDecDecodeParam* param,
                                           JpegDecImageInfo* image_info) {
	JpegHeader    header {};
	const int32_t result = DecodeImage(handle, param, image_info, &header);

	static std::atomic<uint32_t> decoded {0};
	static std::atomic<uint32_t> failed {0};
	if (LogCall(result < 0 ? &failed : &decoded)) {
		if (param != nullptr) {
			LOGF("JpegDec: decode %p, %" PRIu32 " bytes -> %p, %" PRIu32 " bytes, pitch %" PRIu32
			     ", decode mode %u, down scale %u, pixel format %u, alpha %u: %" PRIu32 "x%" PRIu32 ", %" PRIu32
			     " components%s, result 0x%08" PRIx32 "\n",
			     param->jpeg_mem_addr, param->jpeg_mem_size, param->image_mem_addr, param->image_mem_size,
			     param->image_pitch, static_cast<unsigned>(param->decode_mode),
			     static_cast<unsigned>(param->down_scale), static_cast<unsigned>(param->pixel_format),
			     static_cast<unsigned>(param->alpha_value), header.width, header.height, header.components,
			     header.progressive ? " (progressive)" : "", static_cast<uint32_t>(result));
		} else {
			LOGF("JpegDec: decode without parameters, result 0x%08" PRIx32 "\n", static_cast<uint32_t>(result));
		}
	}
	return result;
}

// KYTY_JPEG=1 (default off): libSceJpegDec (and libSceJpegEnc, libJpegEnc.cpp) are registered;
// otherwise their imports stay unresolved stubs that return 0, as before.
static bool JpegLibsEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_JPEG");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

} // namespace JpegDec

LIB_DEFINE(InitJpegDec_1) {
	if (!JpegDec::JpegLibsEnabled()) {
		return;
	}

	PRINT_NAME_ENABLE(true);

	LIB_FUNC("uNAUmANZMEw", JpegDec::JpegDecQueryMemorySize);
	LIB_FUNC("JPh3Zgg0Zwc", JpegDec::JpegDecCreate);
	LIB_FUNC("1kzQRoWEgSA", JpegDec::JpegDecDecode);
	LIB_FUNC("Hwh11+m5KoI", JpegDec::JpegDecDelete);
	LIB_FUNC("LSinoSQH790", JpegDec::JpegDecParseHeader);
}

} // namespace Libs
