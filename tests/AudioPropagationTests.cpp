#include "libs/audioPropagation.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

namespace AP = Libs::AudioPropagation;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "AudioPropagationTests: failed: %s\n", message);
		std::abort();
	}
}

AP::RenderParams Params(AP::Handle source, std::vector<float>& bed) {
	AP::RenderParams params {};
	params.desc              = {0x010107d6u, 0, sizeof(AP::RenderParams)};
	params.source            = source;
	params.output            = bed.data();
	params.output_size_bytes = bed.size() * sizeof(float);
	params.format            = 2;
	return params;
}

AP::Attribute InputAttribute(const AP::InputBuffer& input) {
	AP::Attribute attribute {};
	attribute.id    = AP::SourceAttributeInput;
	attribute.value = &input;
	attribute.size  = sizeof(input);
	return attribute;
}

void Handles() {
	AP::Reset();
	const auto system = AP::Create(AP::Kind::System);
	const auto room   = AP::Create(AP::Kind::Room);
	const auto portal = AP::Create(AP::Kind::Portal);
	Check(system != 0 && room != 0 && portal != 0, "handles are nonzero");
	Check(system != room && room != portal && system != portal, "handles are distinct");
	Check(AP::IsLive(portal, AP::Kind::Portal), "a created portal is live");
	Check(!AP::IsLive(portal, AP::Kind::Room), "a portal is not a room");
	Check(AP::LiveCount() == 3, "three objects");
	Check(AP::Destroy(portal), "a live portal is destroyed");
	Check(!AP::Destroy(portal), "a destroyed portal is not live");
	Check(!AP::IsLive(portal, AP::Kind::Portal), "a destroyed portal is gone");
	Check(!AP::Destroy(0), "handle 0 is never live");
	Check(AP::LiveCount() == 2, "two objects");
}

// The source's mono input becomes channel 0 (W) of the 16-channel bed, everything else is zero.
void RenderDirect() {
	AP::Reset();
	AP::SetRenderMode(AP::RenderMode::Direct);
	const auto         source = AP::Create(AP::Kind::Source);
	std::vector<float> input {0.25f, -0.5f, 1.0f, 0.0f};
	AP::InputBuffer    buffer {input.data(), static_cast<uint32_t>(input.size() * sizeof(float)), 0, 0};
	const auto         attribute = InputAttribute(buffer);
	AP::SetSourceAttributes(source, &attribute, 1);

	std::vector<float> bed(input.size() * 16, 7.0f);
	Check(AP::Render(Params(source, bed)) == 0, "render succeeds");
	for (size_t frame = 0; frame < input.size(); frame++) {
		Check(bed[frame * 16] == input[frame], "channel 0 carries the input");
		for (size_t channel = 1; channel < 16; channel++) {
			Check(bed[frame * 16 + channel] == 0.0f, "the other channels are silent");
		}
	}

	// The input is consumed by the render: a render without a new input is silent.
	std::fill(bed.begin(), bed.end(), 7.0f);
	Check(AP::Render(Params(source, bed)) == 0, "second render succeeds");
	for (const float sample: bed) {
		Check(sample == 0.0f, "a render without input is silent");
	}
}

void RenderSilentAndUnknown() {
	AP::Reset();
	const auto         source = AP::Create(AP::Kind::Source);
	std::vector<float> input(8, 0.5f);
	AP::InputBuffer    buffer {input.data(), static_cast<uint32_t>(input.size() * sizeof(float)), 0, 0};
	const auto         attribute = InputAttribute(buffer);

	AP::SetRenderMode(AP::RenderMode::Silent);
	AP::SetSourceAttributes(source, &attribute, 1);
	std::vector<float> bed(input.size() * 16, 3.0f);
	Check(AP::Render(Params(source, bed)) == 0, "silent render succeeds");
	for (const float sample: bed) {
		Check(sample == 0.0f, "silent mode zeroes the bed");
	}

	AP::SetRenderMode(AP::RenderMode::Direct);
	std::fill(bed.begin(), bed.end(), 3.0f);
	Check(AP::Render(Params(source + 1000, bed)) == 0, "an unknown source renders");
	for (const float sample: bed) {
		Check(sample == 0.0f, "an unknown source is silent");
	}

	// Attributes of a non-source handle are ignored.
	const auto room = AP::Create(AP::Kind::Room);
	AP::SetSourceAttributes(room, &attribute, 1);
	std::fill(bed.begin(), bed.end(), 3.0f);
	Check(AP::Render(Params(room, bed)) == 0, "a room renders nothing");
	Check(bed[0] == 0.0f, "a room is silent");

	AP::RenderParams null_output {};
	null_output.source            = source;
	null_output.output_size_bytes = 64;
	Check(AP::Render(null_output) == AP::ErrorInvalidParam, "a null bed with a size is rejected");
	null_output.output_size_bytes = 0;
	Check(AP::Render(null_output) == 0, "an empty bed is accepted");
}

// Other attributes in the list do not disturb the input; a short input fills only its frames.
void AttributeList() {
	AP::Reset();
	AP::SetRenderMode(AP::RenderMode::Direct);
	const auto         source = AP::Create(AP::Kind::Source);
	std::vector<float> input {1.0f, 2.0f};
	AP::InputBuffer    buffer {input.data(), static_cast<uint32_t>(input.size() * sizeof(float)), 0, 0};
	float              position[4] {1.0f, 2.0f, 3.0f, 0.0f};
	AP::Attribute      list[3] {};
	list[0]       = InputAttribute(buffer);
	list[1].id    = 2;
	list[1].value = position;
	list[1].size  = sizeof(position);
	list[2].id    = 3;
	list[2].value = position;
	list[2].size  = sizeof(position);
	AP::SetSourceAttributes(source, list, 3);
	std::vector<float> bed(2 * 16, 5.0f);
	Check(AP::Render(Params(source, bed)) == 0, "render after an attribute list");
	Check(bed[0] == 1.0f && bed[16] == 2.0f, "input frames in channel 0");
}

} // namespace

int main() {
	Handles();
	RenderDirect();
	RenderSilentAndUnknown();
	AttributeList();
	std::printf("AudioPropagationTests: all passed\n");
	return 0;
}
