#include "common/wheelConfig.h"

#include <algorithm>
#include <charconv>
#include <fmt/format.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <type_traits>

namespace Config::Wheel {
namespace {

using Json = nlohmann::json;

constexpr std::array<std::string_view, CONTROL_COUNT> ControlNames = {
    "steering", "throttle", "brake",  "clutch",  "handbrake", "cross",  "circle", "square",
    "triangle", "up",       "down",   "left",    "right",     "l1",     "r1",     "l2",
    "r2",       "l3",       "r3",     "options", "touchpad",  "gear_1", "gear_2", "gear_3",
    "gear_4",   "gear_5",   "gear_6", "gear_7",  "gear_r",
};

constexpr std::array<std::string_view, 4> BindingTypeNames  = {"none", "axis", "button", "hat"};
constexpr std::array<std::string_view, 5> HatDirectionNames = {"", "up", "down", "left", "right"};
constexpr std::array<std::string_view, 3> FfbDirectionNames = {"steering_axis", "cartesian",
                                                               "polar"};
constexpr std::array<std::string_view, 3> ForceSenseNames   = {"unknown", "normal", "reversed"};

bool                  g_enabled = false;
std::filesystem::path g_config_path {"wheel.json"};

template <typename T, size_t N>
std::optional<T> EnumFromName(const std::array<std::string_view, N>& names, std::string_view name) {
	const auto it = std::find(names.begin(), names.end(), name);
	return it != names.end() ? std::optional<T>(static_cast<T>(it - names.begin())) : std::nullopt;
}

std::optional<uint64_t> ParseU64(std::string_view text) {
	int base = 10;
	if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
		text.remove_prefix(2);
		base = 16;
	}
	uint64_t value          = 0;
	const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value, base);
	if (error != std::errc {} || end != text.data() + text.size()) {
		return std::nullopt;
	}
	return value;
}

// A missing key, or a value of the wrong type, keeps what *out holds. The type is checked first:
// some builds have no exceptions, and a mismatched get would end the process.
template <typename T>
void Read(const Json& object, const char* key, T* out) {
	const auto it = object.find(key);
	if (it == object.end()) {
		return;
	}
	bool fits = false;
	if constexpr (std::is_same_v<T, bool>) {
		fits = it->is_boolean();
	} else if constexpr (std::is_integral_v<T>) {
		fits = it->is_number_integer();
	} else if constexpr (std::is_floating_point_v<T>) {
		fits = it->is_number();
	} else {
		fits = it->is_string();
	}
	if (fits) {
		*out = it->get<T>();
	}
}

// Read as a signed number and clamped, so a negative value does not wrap.
template <typename T>
void ReadClamped(const Json& object, const char* key, T* out, int64_t min, int64_t max) {
	int64_t value = *out;
	Read(object, key, &value);
	*out = static_cast<T>(std::clamp(value, min, max));
}

DeviceId ReadDevice(const Json& object, const char* key) {
	std::string text;
	Read(object, key, &text);
	return DeviceId::FromU64(ParseU64(text).value_or(0));
}

std::string DeviceToString(DeviceId device) {
	return fmt::format("0x{:016X}", device.AsU64());
}

Json BindingToJson(const Binding& b, bool axis) {
	Json j;
	j["device"] = DeviceToString(b.device);
	j["type"]   = BindingTypeName(b.type);
	j["index"]  = b.index;
	if (b.type == BindingType::Hat) {
		j["hat"] = HatDirectionName(b.hat);
	}
	j["reverse"]  = b.reverse;
	j["positive"] = b.positive;
	if (axis) {
		j["dead_zone"] = b.dead_zone;
		j["curve"]     = b.curve;
	}
	return j;
}

Binding BindingFromJson(const Json& j) {
	Binding b;
	if (!j.is_object()) {
		return b;
	}
	std::string type;
	std::string hat;
	b.device = ReadDevice(j, "device");
	Read(j, "type", &type);
	Read(j, "hat", &hat);
	Read(j, "reverse", &b.reverse);
	Read(j, "positive", &b.positive);
	Read(j, "dead_zone", &b.dead_zone);
	Read(j, "curve", &b.curve);
	ReadClamped(j, "index", &b.index, 0, 1023);
	b.type      = EnumFromName<BindingType>(BindingTypeNames, type).value_or(BindingType::None);
	b.hat       = EnumFromName<HatDirection>(HatDirectionNames, hat).value_or(HatDirection::None);
	b.dead_zone = std::clamp(b.dead_zone, 0.0f, 0.9f);
	b.curve     = std::clamp(b.curve, 0.2f, 5.0f);
	return b;
}

void SetError(std::string* error, std::string text) {
	if (error != nullptr) {
		*error = std::move(text);
	}
}

} // namespace

void SetEnabled(bool enabled) {
	g_enabled = enabled;
}

bool IsEnabled() {
	return g_enabled;
}

void SetConfigPath(const std::filesystem::path& path) {
	g_config_path = path;
}

std::filesystem::path GetConfigPath() {
	return g_config_path;
}

uint64_t DeviceId::AsU64() const {
	uint64_t value = product_id;
	value |= static_cast<uint64_t>(vendor_id) << 16u;
	value |= static_cast<uint64_t>(num_axes & 0x3ffu) << 32u;
	value |= static_cast<uint64_t>(num_hats & 0x3ffu) << 42u;
	value |= static_cast<uint64_t>(num_buttons & 0x3ffu) << 52u;
	return value;
}

DeviceId DeviceId::FromU64(uint64_t value) {
	DeviceId id;
	id.product_id  = static_cast<uint16_t>(value & 0xffffu);
	id.vendor_id   = static_cast<uint16_t>((value >> 16u) & 0xffffu);
	id.num_axes    = static_cast<uint16_t>((value >> 32u) & 0x3ffu);
	id.num_hats    = static_cast<uint16_t>((value >> 42u) & 0x3ffu);
	id.num_buttons = static_cast<uint16_t>((value >> 52u) & 0x3ffu);
	return id;
}

std::string_view ControlName(Control c) {
	return ControlNames[static_cast<int>(c)];
}

std::optional<Control> ControlFromName(std::string_view name) {
	return EnumFromName<Control>(ControlNames, name);
}

std::string_view BindingTypeName(BindingType type) {
	return BindingTypeNames[static_cast<int>(type)];
}

std::string_view HatDirectionName(HatDirection hat) {
	return HatDirectionNames[static_cast<int>(hat)];
}

std::string_view FfbDirectionName(FfbDirection direction) {
	return FfbDirectionNames[static_cast<int>(direction)];
}

std::string_view ForceSenseName(ForceSense sense) {
	return ForceSenseNames[static_cast<int>(sense)];
}

std::string ToJson(const WheelConfig& config) {
	Json j;
	j["version"]          = CONFIG_VERSION;
	j["model"]            = config.model;
	j["background_input"] = config.background_input;

	Json& ffb                  = j["ffb"];
	ffb["device"]              = DeviceToString(config.ffb_device);
	ffb["gain"]                = config.ffb_gain;
	ffb["direction"]           = FfbDirectionName(config.ffb_direction);
	ffb["reverse"]             = config.reverse_effects;
	ffb["force_sense"]         = ForceSenseName(config.force_sense);
	ffb["software_conditions"] = config.software_conditions;

	Json& rotation               = j["rotation"];
	rotation["physical_degrees"] = config.physical_rotation_deg;
	rotation["soft_lock"]        = config.soft_lock;

	Json& bindings = j["bindings"];
	bindings       = Json::object();
	for (int i = 0; i < CONTROL_COUNT; i++) {
		const auto c                          = static_cast<Control>(i);
		bindings[std::string(ControlName(c))] = BindingToJson(config.Get(c), IsAxisControl(c));
	}
	// Invalid UTF-8 (a hand-edited model name) is replaced rather than thrown.
	return j.dump(2, ' ', false, Json::error_handler_t::replace);
}

std::optional<WheelConfig> FromJson(std::string_view text, std::string* error) {
	const Json j = Json::parse(text, nullptr, false);
	if (j.is_discarded() || !j.is_object()) {
		SetError(error, "not a JSON object");
		return std::nullopt;
	}

	int version = CONFIG_VERSION;
	Read(j, "version", &version);
	if (version > CONFIG_VERSION) {
		SetError(error, fmt::format("version {} is newer than {}", version, CONFIG_VERSION));
		return std::nullopt;
	}
	WheelConfig config;
	Read(j, "model", &config.model);
	Read(j, "background_input", &config.background_input);

	if (auto it = j.find("ffb"); it != j.end() && it->is_object()) {
		std::string direction;
		std::string sense;
		config.ffb_device = ReadDevice(*it, "device");
		ReadClamped(*it, "gain", &config.ffb_gain, 0, 100);
		Read(*it, "direction", &direction);
		Read(*it, "reverse", &config.reverse_effects);
		Read(*it, "force_sense", &sense);
		Read(*it, "software_conditions", &config.software_conditions);
		config.ffb_direction = EnumFromName<FfbDirection>(FfbDirectionNames, direction)
		                           .value_or(FfbDirection::SteeringAxis);
		config.force_sense =
		    EnumFromName<ForceSense>(ForceSenseNames, sense).value_or(ForceSense::Unknown);
	}
	if (auto it = j.find("rotation"); it != j.end() && it->is_object()) {
		ReadClamped(*it, "physical_degrees", &config.physical_rotation_deg, 90, 2880);
		Read(*it, "soft_lock", &config.soft_lock);
	}
	if (auto it = j.find("bindings"); it != j.end() && it->is_object()) {
		for (const auto& [name, value]: it->items()) {
			if (auto c = ControlFromName(name)) {
				config.Get(*c) = BindingFromJson(value);
			}
		}
	}
	return config;
}

std::optional<WheelConfig> Load(const std::filesystem::path& path, std::string* error) {
	std::ifstream file(path, std::ios::binary);
	if (!file) {
		SetError(error, "cannot open the file");
		return std::nullopt;
	}
	std::stringstream text;
	text << file.rdbuf();
	return FromJson(text.str(), error);
}

bool Save(const std::filesystem::path& path, const WheelConfig& config, std::string* error) {
	// Written beside and renamed over the old file, so an emulator re-reading it never sees half.
	auto temp = path;
	temp += ".tmp";
	bool written = false;
	{
		std::ofstream file(temp, std::ios::binary | std::ios::trunc);
		file << ToJson(config) << '\n';
		file.close();
		written = !file.fail();
	}
	std::error_code ec;
	if (written) {
		std::filesystem::rename(temp, path, ec);
	}
	if (!written || ec) {
		SetError(error, written ? ec.message() : "cannot write the file");
		std::filesystem::remove(temp, ec);
		return false;
	}
	return true;
}

std::optional<WheelConfig> ImportRpcs3G27(std::string_view yaml, std::string* error) {
	struct Key {
		std::string_view name;
		Control          control;
	};
	// RPCS3's keys (LogitechG27Config.h): "pause" (Start) becomes Options, "select" the touch pad
	// press. Plus, minus, the dial and PS have no counterpart and are skipped.
	static constexpr std::array<Key, 27> Keys = {{
	    {"steering", Control::Steering},
	    {"throttle", Control::Throttle},
	    {"brake", Control::Brake},
	    {"clutch", Control::Clutch},
	    {"shift_up", Control::R1},
	    {"shift_down", Control::L1},
	    {"up", Control::Up},
	    {"down", Control::Down},
	    {"left", Control::Left},
	    {"right", Control::Right},
	    {"triangle", Control::Triangle},
	    {"cross", Control::Cross},
	    {"square", Control::Square},
	    {"circle", Control::Circle},
	    {"l2", Control::L2},
	    {"l3", Control::L3},
	    {"r2", Control::R2},
	    {"r3", Control::R3},
	    {"select", Control::TouchPad},
	    {"pause", Control::Options},
	    {"shifter_1", Control::Gear1},
	    {"shifter_2", Control::Gear2},
	    {"shifter_3", Control::Gear3},
	    {"shifter_4", Control::Gear4},
	    {"shifter_5", Control::Gear5},
	    {"shifter_6", Control::Gear6},
	    {"shifter_r", Control::GearReverse},
	}};

	auto trim = [](std::string_view s) {
		const auto first = s.find_first_not_of(" \t\r");
		const auto last  = s.find_last_not_of(" \t\r");
		s                = first == std::string_view::npos ? std::string_view {}
		                                                   : s.substr(first, last - first + 1);
		if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
			s = s.substr(1, s.size() - 2);
		}
		return s;
	};
	auto device = [](std::string_view text) {
		return DeviceId::FromU64(ParseU64(text).value_or(0));
	};

	WheelConfig config;
	Binding*    current = nullptr;
	for (size_t pos = 0; pos < yaml.size();) {
		const size_t end  = std::min(yaml.find('\n', pos), yaml.size());
		const auto   line = yaml.substr(pos, end - pos);
		pos               = end + 1;

		const auto colon = line.find(':');
		if (colon == std::string_view::npos) {
			continue;
		}
		const bool indented = line[0] == ' ' || line[0] == '\t';
		const auto key      = trim(line.substr(0, colon));
		const auto value    = trim(line.substr(colon + 1));

		if (!indented) {
			current = nullptr;
			if (value.empty()) {
				const auto it = std::find_if(Keys.begin(), Keys.end(),
				                             [&](const Key& k) { return k.name == key; });
				current       = it != Keys.end() ? &config.Get(it->control) : nullptr;
			} else if (key == "reverse_effects") {
				config.reverse_effects = value == "true";
			} else if (key == "ffb_direction_type") {
				config.ffb_direction = EnumFromName<FfbDirection>(FfbDirectionNames, value)
				                           .value_or(FfbDirection::SteeringAxis);
			} else if (key == "ffb_device_type_id") {
				config.ffb_device = device(value);
			}
		} else if (current != nullptr) {
			if (key == "device_type_id") {
				current->device = device(value);
			} else if (key == "type") {
				current->type =
				    EnumFromName<BindingType>(BindingTypeNames, value).value_or(BindingType::None);
			} else if (key == "id") {
				current->index =
				    static_cast<int>(std::min<uint64_t>(ParseU64(value).value_or(0), 1023));
			} else if (key == "hat") {
				current->hat = EnumFromName<HatDirection>(HatDirectionNames, value)
				                   .value_or(HatDirection::None);
			} else if (key == "reverse") {
				current->reverse = value == "true";
			}
		}
	}

	// RPCS3 keeps a type on unmapped controls with device 0: those stay unbound.
	bool any = false;
	for (auto& b: config.bindings) {
		if (b.device.IsNone()) {
			b = Binding {};
		}
		any |= b.IsBound();
	}
	if (!any) {
		SetError(error, "no mapped control found");
		return std::nullopt;
	}
	return config;
}

} // namespace Config::Wheel
