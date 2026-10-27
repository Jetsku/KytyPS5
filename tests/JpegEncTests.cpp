#include "libs/libJpegEnc.cpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#include "stb_image.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace Loader {
void SymbolDatabase::Add(const SymbolResolve&, uint64_t, const std::string&) {}
namespace Timer {
double GetTimeMs() {
	return 0.0;
}
} // namespace Timer
} // namespace Loader

namespace {
using namespace Libs::JpegEnc;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "JpegEncTests:%d: %s\n", __LINE__, #condition);                   \
			std::abort();                                                                          \
		}                                                                                          \
	} while (false)

struct Encoder {
	// An odd start, as a game's allocation may be: the handle is at the next 32-byte boundary.
	alignas(64) std::array<uint8_t, JPEG_ENC_MEMORY_SIZE + 1> memory {};
	void* handle = nullptr;

	Encoder() {
		const JpegEncCreateParam create {sizeof(JpegEncCreateParam), 0};
		CHECK(JpegEncQueryMemorySize(&create) == static_cast<int32_t>(JPEG_ENC_MEMORY_SIZE));
		CHECK(JpegEncCreate(&create, memory.data() + 1, JPEG_ENC_MEMORY_SIZE, &handle) == 0);
		CHECK(handle != nullptr && (reinterpret_cast<uintptr_t>(handle) & 31u) == 0);
	}

	~Encoder() {
		CHECK(JpegEncDelete(handle) == 0);
		CHECK(JpegEncDelete(handle) == JPEG_ENC_ERROR_INVALID_HANDLE);
	}
};

JpegEncEncodeParam EncodeParam(const std::vector<uint8_t>& image, uint32_t width, uint32_t height,
                               uint32_t pitch, uint16_t format, std::vector<uint8_t>& jpeg) {
	JpegEncEncodeParam param {};
	param.image_mem_addr    = image.data();
	param.jpeg_mem_addr     = jpeg.data();
	param.image_mem_size    = static_cast<uint32_t>(image.size());
	param.jpeg_mem_size     = static_cast<uint32_t>(jpeg.size());
	param.image_width       = width;
	param.image_height      = height;
	param.image_pitch       = pitch;
	param.pixel_format      = format;
	param.encode_mode       = JPEG_ENC_ENCODE_MODE_NORMAL;
	param.color_space       = format == JPEG_ENC_PIXEL_FORMAT_Y8 ? JPEG_ENC_COLOR_SPACE_GRAYSCALE
	                                                             : JPEG_ENC_COLOR_SPACE_YCC;
	param.sampling_type     = format == JPEG_ENC_PIXEL_FORMAT_Y8 ? 0 : JPEG_ENC_SAMPLING_TYPE_420;
	param.compression_ratio = 1;
	return param;
}

// The colour the test pictures have at a pixel: smooth, so a JPEG keeps it closely.
std::array<int, 3> Colour(uint32_t x, uint32_t y) {
	return {static_cast<int>(40 + x * 2), static_cast<int>(200 - y * 2),
	        static_cast<int>(60 + (x + y))};
}

// The JPEG decoded, compared with Colour: the mean difference of its channels.
double MeanError(const std::vector<uint8_t>& jpeg, int32_t size, uint32_t width, uint32_t height,
                 bool gray) {
	int      w          = 0;
	int      h          = 0;
	int      components = 0;
	uint8_t* rgb        = stbi_load_from_memory(jpeg.data(), size, &w, &h, &components, 3);
	CHECK(rgb != nullptr && w == static_cast<int>(width) && h == static_cast<int>(height));
	double total = 0.0;
	for (uint32_t y = 0; y < height; y++) {
		for (uint32_t x = 0; x < width; x++) {
			const auto  expected = Colour(x, y);
			const auto* pixel    = rgb + (size_t {y} * width + x) * 3u;
			for (int c = 0; c < 3; c++) {
				const int want =
				    gray ? (expected[0] * 299 + expected[1] * 587 + expected[2] * 114) / 1000
				         : expected[c];
				total += std::abs(static_cast<int>(pixel[c]) - want);
			}
		}
	}
	stbi_image_free(rgb);
	return total / (static_cast<double>(width) * height * 3.0);
}

void TestRgba() {
	Encoder        encoder;
	constexpr auto width  = 40u;
	constexpr auto height = 24u;
	for (const bool bgr: {false, true}) {
		const uint32_t       pitch = width * 4u + 16u;
		std::vector<uint8_t> image(size_t {pitch} * height, 0xee);
		for (uint32_t y = 0; y < height; y++) {
			for (uint32_t x = 0; x < width; x++) {
				const auto colour = Colour(x, y);
				auto*      pixel  = image.data() + size_t {y} * pitch + x * 4u;
				pixel[0]          = static_cast<uint8_t>(bgr ? colour[2] : colour[0]);
				pixel[1]          = static_cast<uint8_t>(colour[1]);
				pixel[2]          = static_cast<uint8_t>(bgr ? colour[0] : colour[2]);
				pixel[3]          = 0x80;
			}
		}
		std::vector<uint8_t> jpeg(64 * 1024, 0);
		auto                 param = EncodeParam(
            image, width, height, pitch,
            bgr ? JPEG_ENC_PIXEL_FORMAT_B8G8R8A8 : JPEG_ENC_PIXEL_FORMAT_R8G8B8A8, jpeg);
		JpegEncOutputInfo info {0xdeadbeef, 0xdeadbeef};
		const int32_t     size = JpegEncEncode(encoder.handle, &param, &info);
		CHECK(size > 0 && info.data_size == static_cast<uint32_t>(size) &&
		      info.processed_height == height);
		CHECK(jpeg[0] == 0xff && jpeg[1] == 0xd8 && jpeg[size - 2] == 0xff &&
		      jpeg[size - 1] == 0xd9);
		CHECK(MeanError(jpeg, size, width, height, false) < 6.0);
	}
}

void TestGrayAndYuyv() {
	Encoder        encoder;
	constexpr auto width  = 33u; // odd: the last Y8U8Y8V8 pair has one pixel
	constexpr auto height = 17u;
	{
		std::vector<uint8_t> image(size_t {width} * height);
		for (uint32_t y = 0; y < height; y++) {
			for (uint32_t x = 0; x < width; x++) {
				const auto c = Colour(x, y);
				image[size_t {y} * width + x] =
				    static_cast<uint8_t>((c[0] * 299 + c[1] * 587 + c[2] * 114) / 1000);
			}
		}
		std::vector<uint8_t> jpeg(64 * 1024, 0);
		auto param = EncodeParam(image, width, height, width, JPEG_ENC_PIXEL_FORMAT_Y8, jpeg);
		const int32_t size = JpegEncEncode(encoder.handle, &param, nullptr);
		CHECK(size > 0);
		CHECK(MeanError(jpeg, size, width, height, true) < 6.0);
	}
	{
		// Full-range YCbCr of each pixel, the pair's chroma from its first pixel.
		const uint32_t       pitch = (width + 1u) / 2u * 4u;
		std::vector<uint8_t> image(size_t {pitch} * height);
		for (uint32_t y = 0; y < height; y++) {
			for (uint32_t x = 0; x < width; x++) {
				const auto c                  = Colour(x & ~1u, y);
				const auto luma               = (c[0] * 299 + c[1] * 587 + c[2] * 114) / 1000;
				auto*      pair               = image.data() + size_t {y} * pitch + (x / 2u) * 4u;
				pair[(x & 1u) != 0 ? 2u : 0u] = static_cast<uint8_t>(luma);
				pair[1] =
				    static_cast<uint8_t>(std::clamp(128 + (c[2] - luma) * 564 / 1000, 0, 255));
				pair[3] =
				    static_cast<uint8_t>(std::clamp(128 + (c[0] - luma) * 713 / 1000, 0, 255));
			}
		}
		std::vector<uint8_t> jpeg(64 * 1024, 0);
		auto param = EncodeParam(image, width, height, pitch, JPEG_ENC_PIXEL_FORMAT_Y8U8Y8V8, jpeg);
		const int32_t size = JpegEncEncode(encoder.handle, &param, nullptr);
		CHECK(size > 0);
		// The pairs share a colour, so the error is the difference between neighbours too.
		CHECK(MeanError(jpeg, size, width, height, false) < 8.0);
	}
}

void TestOutputCapacity() {
	Encoder              encoder;
	constexpr auto       width  = 64u;
	constexpr auto       height = 64u;
	std::vector<uint8_t> image(size_t {width} * height * 4u);
	for (size_t i = 0; i < image.size(); i++) {
		image[i] = static_cast<uint8_t>((i * 2654435761u) >> 24u); // noise: hard to compress
	}
	std::vector<uint8_t> roomy(256 * 1024, 0);
	auto                 param =
	    EncodeParam(image, width, height, width * 4u, JPEG_ENC_PIXEL_FORMAT_R8G8B8A8, roomy);
	const int32_t best = JpegEncEncode(encoder.handle, &param, nullptr);
	CHECK(best > 0);
	// Less room than the best quality takes: made again with more compression, and it fits.
	std::vector<uint8_t> tight(static_cast<size_t>(best) * 2u / 3u, 0);
	param = EncodeParam(image, width, height, width * 4u, JPEG_ENC_PIXEL_FORMAT_R8G8B8A8, tight);
	const int32_t smaller = JpegEncEncode(encoder.handle, &param, nullptr);
	CHECK(smaller > 0 && smaller <= static_cast<int32_t>(tight.size()) && tight[0] == 0xff &&
	      tight[1] == 0xd8);
	// No room at all: an error, and the output info says nothing was made.
	std::vector<uint8_t> tiny(64, 0);
	param = EncodeParam(image, width, height, width * 4u, JPEG_ENC_PIXEL_FORMAT_R8G8B8A8, tiny);
	JpegEncOutputInfo info {7, 7};
	CHECK(JpegEncEncode(encoder.handle, &param, &info) == JPEG_ENC_ERROR_INVALID_SIZE);
	CHECK(info.data_size == 7 && info.processed_height == 7);
}

void TestValidation() {
	const JpegEncCreateParam bad_size {4, 0};
	CHECK(JpegEncQueryMemorySize(&bad_size) == JPEG_ENC_ERROR_INVALID_SIZE);
	const JpegEncCreateParam bad_attribute {sizeof(JpegEncCreateParam), 1};
	CHECK(JpegEncQueryMemorySize(&bad_attribute) == JPEG_ENC_ERROR_INVALID_PARAM);
	CHECK(JpegEncQueryMemorySize(nullptr) == JPEG_ENC_ERROR_INVALID_ADDR);
	const JpegEncCreateParam create {sizeof(JpegEncCreateParam), 0};
	std::array<uint8_t, 8>   small {};
	void*                    handle = nullptr;
	CHECK(JpegEncCreate(&create, small.data(), static_cast<uint32_t>(small.size()), &handle) ==
	      JPEG_ENC_ERROR_INVALID_SIZE);
	CHECK(JpegEncCreate(&create, nullptr, JPEG_ENC_MEMORY_SIZE, &handle) ==
	      JPEG_ENC_ERROR_INVALID_ADDR);

	Encoder              encoder;
	std::vector<uint8_t> image(16 * 4 * 4, 0x40);
	std::vector<uint8_t> jpeg(16 * 1024, 0);
	auto param = EncodeParam(image, 16, 4, 64, JPEG_ENC_PIXEL_FORMAT_R8G8B8A8, jpeg);
	CHECK(JpegEncEncode(nullptr, &param, nullptr) == JPEG_ENC_ERROR_INVALID_HANDLE);
	CHECK(JpegEncEncode(image.data(), &param, nullptr) == JPEG_ENC_ERROR_INVALID_HANDLE);
	CHECK(JpegEncEncode(encoder.handle, nullptr, nullptr) == JPEG_ENC_ERROR_INVALID_ADDR);
	auto narrow        = param;
	narrow.image_pitch = 60;
	CHECK(JpegEncEncode(encoder.handle, &narrow, nullptr) == JPEG_ENC_ERROR_INVALID_SIZE);
	auto tall         = param;
	tall.image_height = 5;
	CHECK(JpegEncEncode(encoder.handle, &tall, nullptr) == JPEG_ENC_ERROR_INVALID_SIZE);
	auto format         = param;
	format.pixel_format = 2;
	CHECK(JpegEncEncode(encoder.handle, &format, nullptr) == JPEG_ENC_ERROR_INVALID_PARAM);
	auto colour        = param;
	colour.color_space = 3;
	CHECK(JpegEncEncode(encoder.handle, &colour, nullptr) == JPEG_ENC_ERROR_INVALID_PARAM);
	auto empty        = param;
	empty.image_width = 0;
	CHECK(JpegEncEncode(encoder.handle, &empty, nullptr) == JPEG_ENC_ERROR_INVALID_PARAM);
	CHECK(JpegEncEncode(encoder.handle, &param, nullptr) > 0);
}

} // namespace

int main() {
	TestRgba();
	TestGrayAndYuyv();
	TestOutputCapacity();
	TestValidation();
	std::puts("JpegEncTests: all cases passed");
}
