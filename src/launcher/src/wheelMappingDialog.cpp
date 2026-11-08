#include "wheelMappingDialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QTreeWidget>
#include <QVBoxLayout>

#include <SDL3/SDL.h>
#include <algorithm>
#include <cmath>

using Config::Wheel::Binding;
using Config::Wheel::BindingType;
using Config::Wheel::Control;
using Config::Wheel::DeviceId;
using Config::Wheel::FfbDirection;
using Config::Wheel::ForceSense;
using Config::Wheel::HatDirection;

namespace {

constexpr int CaptureTimeoutMs = 5500;
// An axis counts as moved once it has travelled a fifth of its range, as in RPCS3.
constexpr int AxisThreshold = 0x7FFF / 5;
// An axis counts as alive once its readings have spanned this much (as the emulator decides).
constexpr int AxisActivity = 2000;

QString ControlLabel(Control c) {
	switch (c) {
		case Control::Steering: return QObject::tr("Steering");
		case Control::Throttle: return QObject::tr("Throttle");
		case Control::Brake: return QObject::tr("Brake");
		case Control::Clutch: return QObject::tr("Clutch");
		case Control::Handbrake: return QObject::tr("Handbrake");
		case Control::Cross: return QObject::tr("Cross ✕");
		case Control::Circle: return QObject::tr("Circle ○");
		case Control::Square: return QObject::tr("Square □");
		case Control::Triangle: return QObject::tr("Triangle △");
		case Control::Up: return QObject::tr("D-pad up");
		case Control::Down: return QObject::tr("D-pad down");
		case Control::Left: return QObject::tr("D-pad left");
		case Control::Right: return QObject::tr("D-pad right");
		case Control::L1: return QObject::tr("L1 (left paddle)");
		case Control::R1: return QObject::tr("R1 (right paddle)");
		case Control::L2: return QObject::tr("L2");
		case Control::R2: return QObject::tr("R2");
		case Control::L3: return QObject::tr("L3");
		case Control::R3: return QObject::tr("R3");
		case Control::Options: return QObject::tr("Options");
		case Control::TouchPad: return QObject::tr("Touchpad press");
		case Control::Gear1: return QObject::tr("Gear 1");
		case Control::Gear2: return QObject::tr("Gear 2");
		case Control::Gear3: return QObject::tr("Gear 3");
		case Control::Gear4: return QObject::tr("Gear 4");
		case Control::Gear5: return QObject::tr("Gear 5");
		case Control::Gear6: return QObject::tr("Gear 6");
		case Control::Gear7: return QObject::tr("Gear 7");
		case Control::GearReverse: return QObject::tr("Gear R");
		case Control::Count: break;
	}
	return {};
}

int16_t ReadAxis(SDL_Joystick* j, int index) {
	int16_t v = SDL_GetJoystickAxis(j, index);
	return v == -32767 ? int16_t {-32768} : v;
}

bool HatHas(uint8_t hat, HatDirection dir) {
	switch (dir) {
		case HatDirection::Up: return (hat & SDL_HAT_UP) != 0;
		case HatDirection::Down: return (hat & SDL_HAT_DOWN) != 0;
		case HatDirection::Left: return (hat & SDL_HAT_LEFT) != 0;
		case HatDirection::Right: return (hat & SDL_HAT_RIGHT) != 0;
		case HatDirection::None: break;
	}
	return false;
}

} // namespace

WheelMappingDialog::WheelMappingDialog(const QString& config_path, bool enabled, QWidget* parent)
    : QDialog(parent), m_path(config_path) {
	setWindowTitle(tr("Racing wheel"));
	m_sdl_ready = SDL_InitSubSystem(SDL_INIT_JOYSTICK | SDL_INIT_HAPTIC);
	if (QFile::exists(m_path)) {
		std::string error;
		if (auto loaded = Config::Wheel::Load(m_path.toStdWString(), &error)) {
			m_config = *loaded;
		} else {
			QMessageBox::warning(this, tr("Racing wheel"),
			                     tr("%1 could not be read (%2); starting from an empty mapping.")
			                         .arg(m_path, QString::fromStdString(error)));
		}
	}
	BuildUi(enabled);
	RefreshDevices();
	LoadToUi();
	m_refresh_clock.start();
	m_timer.setInterval(25);
	connect(&m_timer, &QTimer::timeout, this, &WheelMappingDialog::Tick);
	if (m_sdl_ready) {
		m_timer.start();
	} else {
		m_status->setText(tr("SDL joystick support could not start: %1").arg(SDL_GetError()));
	}
}

WheelMappingDialog::~WheelMappingDialog() {
	m_timer.stop();
	for (auto& d: m_devices) {
		SDL_CloseJoystick(d.joystick);
	}
	if (m_sdl_ready) {
		SDL_QuitSubSystem(SDL_INIT_JOYSTICK | SDL_INIT_HAPTIC);
	}
}

bool WheelMappingDialog::Enabled() const {
	return !m_emulated->currentData().toString().isEmpty();
}

void WheelMappingDialog::BuildUi(bool enabled) {
	auto* layout = new QVBoxLayout(this);

	// The wheel the game is shown; Off leaves it the pad alone.
	auto* emulated_row = new QHBoxLayout;
	m_emulated         = new QComboBox;
	m_emulated->addItem(tr("Off"), QString());
	m_emulated->addItem(tr("Fanatec Gran Turismo DD Pro"),
	                    QString::fromLatin1(Config::Wheel::MODEL_GT_DD_PRO.data(),
	                                        Config::Wheel::MODEL_GT_DD_PRO.size()));
	const int model = m_emulated->findData(QString::fromStdString(m_config.model));
	m_emulated->setCurrentIndex(enabled ? std::max(model, 1) : 0);
	emulated_row->addWidget(new QLabel(tr("Emulated wheel")));
	emulated_row->addWidget(m_emulated);
	emulated_row->addStretch();
	layout->addLayout(emulated_row);

	auto* note =
	    new QLabel(tr("Map the wheel here, not in the controller mapping. Pedals at rest "
	                  "should read 0 %; tick Reverse on a pedal that reads 100 % at rest."));
	note->setWordWrap(true);
	layout->addWidget(note);

	m_table = new QTreeWidget;
	m_table->setColumnCount(3);
	m_table->setHeaderLabels({tr("Control"), tr("Bound to"), tr("Value")});
	m_table->setRootIsDecorated(false);
	m_table->header()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
	m_table->header()->setSectionResizeMode(1, QHeaderView::Stretch);
	m_table->setMinimumSize(620, 360);
	for (int i = 0; i < Config::Wheel::CONTROL_COUNT; i++) {
		auto* item = new QTreeWidgetItem(m_table);
		item->setText(0, ControlLabel(static_cast<Control>(i)));
		item->setData(0, Qt::UserRole, i);
	}
	layout->addWidget(m_table);

	auto* row       = new QHBoxLayout;
	m_map_button    = new QPushButton(tr("Map"));
	m_clear_button  = new QPushButton(tr("Clear"));
	m_row_reverse   = new QCheckBox(tr("Reverse"));
	m_row_dead_zone = new QDoubleSpinBox;
	m_row_dead_zone->setRange(0.0, 50.0);
	m_row_dead_zone->setSuffix(QStringLiteral(" %"));
	m_row_dead_zone->setDecimals(1);
	m_row_dead_zone->setToolTip(tr("Steering: around the centre, in percent of half the travel. "
	                               "Pedals: at the released end, in percent of the whole travel."));
	m_row_curve = new QDoubleSpinBox;
	m_row_curve->setRange(0.2, 5.0);
	m_row_curve->setSingleStep(0.1);
	m_row_curve->setToolTip(tr("1 is linear; above 1 is finer near the centre or the released "
	                           "end, below 1 is quicker."));
	row->addWidget(m_map_button);
	row->addWidget(m_clear_button);
	row->addWidget(m_row_reverse);
	row->addWidget(new QLabel(tr("Dead zone")));
	row->addWidget(m_row_dead_zone);
	row->addWidget(new QLabel(tr("Curve")));
	row->addWidget(m_row_curve);
	row->addStretch();
	layout->addLayout(row);

	auto* ffb      = new QGroupBox(tr("Force feedback and rotation"));
	auto* form     = new QFormLayout(ffb);
	auto* gain_row = new QHBoxLayout;
	m_gain         = new QSlider(Qt::Horizontal);
	m_gain->setRange(0, 100);
	m_gain_label = new QLabel;
	gain_row->addWidget(m_gain);
	gain_row->addWidget(m_gain_label);
	form->addRow(tr("Strength"), gain_row);
	m_ffb_device = new QComboBox;
	form->addRow(tr("Force-feedback device"), m_ffb_device);
	m_direction = new QComboBox;
	m_direction->addItem(tr("Steering axis (default)"),
	                     static_cast<int>(FfbDirection::SteeringAxis));
	m_direction->addItem(tr("Cartesian"), static_cast<int>(FfbDirection::Cartesian));
	m_direction->addItem(tr("Polar"), static_cast<int>(FfbDirection::Polar));
	m_direction->setToolTip(tr("Change only if the wheel feels no force with the default; some "
	                           "DirectInput drivers ignore the steering-axis encoding."));
	form->addRow(tr("Direction encoding"), m_direction);
	m_reverse = new QCheckBox(tr("Reverse force direction"));
	m_reverse->setToolTip(tr("Mirrors the game's forces. Run the force test first; tick this only "
	                         "if the wheel still pulls away from the centre in the game."));
	form->addRow(QString(), m_reverse);
	m_rotation = new QSpinBox;
	m_rotation->setRange(90, 2880);
	m_rotation->setSuffix(tr(" degrees"));
	m_rotation->setToolTip(tr("The rotation set in the wheel's own software. When the game sets a "
	                          "car's steering lock, steering is scaled to it."));
	form->addRow(tr("Wheel rotation"), m_rotation);
	m_sense_label = new QLabel;
	form->addRow(tr("Force direction"), m_sense_label);
	m_soft_lock = new QCheckBox(tr("Soft end stop at the car's steering lock"));
	m_soft_lock->setToolTip(tr("Pushes back past the car's lock. Works once the force test has "
	                           "measured the wheel's force direction."));
	form->addRow(QString(), m_soft_lock);
	m_background = new QCheckBox(
	    tr("Keep reading controllers when the game window has no focus (all of them, pads too)"));
	form->addRow(QString(), m_background);
	m_software_conditions = new QCheckBox(
	    tr("Compute damper and friction in the emulator (for drivers that mishandle them)"));
	m_software_conditions->setToolTip(
	    tr("Works once the force test has measured the wheel's force direction."));
	form->addRow(QString(), m_software_conditions);
	layout->addWidget(ffb);

	m_status = new QLabel;
	m_status->setWordWrap(true);
	layout->addWidget(m_status);

	auto* buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel);
	auto* import =
	    buttons->addButton(tr("Import RPCS3 G27 mapping..."), QDialogButtonBox::ActionRole);
	auto* test_button = buttons->addButton(tr("Test force feedback"), QDialogButtonBox::ActionRole);
	layout->addWidget(buttons);

	connect(buttons, &QDialogButtonBox::accepted, this, [this] {
		if (Enabled() && !m_config.Get(Control::Steering).IsBound() &&
		    QMessageBox::question(this, tr("Racing wheel"),
		                          tr("Steering is not mapped, so the game will not see the wheel. "
		                             "Save anyway?")) != QMessageBox::Yes) {
			return;
		}
		if (Save()) {
			accept();
		}
	});
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
	connect(import, &QPushButton::clicked, this, &WheelMappingDialog::ImportRpcs3);
	connect(test_button, &QPushButton::clicked, this, &WheelMappingDialog::TestForce);
	connect(m_map_button, &QPushButton::clicked, this, &WheelMappingDialog::StartCapture);
	connect(m_clear_button, &QPushButton::clicked, this, &WheelMappingDialog::ClearBinding);
	connect(m_table, &QTreeWidget::currentItemChanged, this, [this] {
		m_loading      = true;
		const auto  c  = SelectedControl();
		const bool  ok = c != Control::Count;
		const auto& b  = ok ? m_config.Get(c) : Binding {};
		m_row_reverse->setChecked(b.reverse);
		m_row_dead_zone->setValue(b.dead_zone * 100.0);
		m_row_curve->setValue(b.curve);
		const bool axis = ok && Config::Wheel::IsAxisControl(c);
		m_row_dead_zone->setEnabled(axis);
		m_row_curve->setEnabled(axis);
		m_map_button->setEnabled(ok);
		m_clear_button->setEnabled(ok);
		m_loading = false;
	});
	auto row_changed = [this] {
		if (m_loading) {
			return;
		}
		const auto c = SelectedControl();
		if (c == Control::Count) {
			return;
		}
		auto& b     = m_config.Get(c);
		b.reverse   = m_row_reverse->isChecked();
		b.dead_zone = static_cast<float>(m_row_dead_zone->value() / 100.0);
		b.curve     = static_cast<float>(m_row_curve->value());
	};
	connect(m_row_reverse, &QCheckBox::toggled, this, row_changed);
	connect(m_row_dead_zone, &QDoubleSpinBox::valueChanged, this, row_changed);
	connect(m_row_curve, &QDoubleSpinBox::valueChanged, this, row_changed);
	// Another force-feedback device has its own sense.
	connect(m_ffb_device, &QComboBox::activated, this, [this] { ForgetForceSense(); });
	connect(m_gain, &QSlider::valueChanged, this,
	        [this](int v) { m_gain_label->setText(QStringLiteral("%1 %").arg(v)); });
	m_table->setCurrentItem(m_table->topLevelItem(0));
}

void WheelMappingDialog::LoadToUi() {
	m_gain->setValue(static_cast<int>(m_config.ffb_gain));
	m_gain_label->setText(QStringLiteral("%1 %").arg(m_config.ffb_gain));
	m_direction->setCurrentIndex(m_direction->findData(static_cast<int>(m_config.ffb_direction)));
	m_reverse->setChecked(m_config.reverse_effects);
	m_rotation->setValue(static_cast<int>(m_config.physical_rotation_deg));
	m_soft_lock->setChecked(m_config.soft_lock);
	m_background->setChecked(m_config.background_input);
	m_software_conditions->setChecked(m_config.software_conditions);
	UpdateSenseLabel();
	emit m_table->currentItemChanged(m_table->currentItem(), nullptr);
}

void WheelMappingDialog::UpdateSenseLabel() {
	switch (m_config.force_sense) {
		case ForceSense::Unknown:
			m_sense_label->setText(tr("not measured: run Test force feedback"));
			break;
		case ForceSense::Normal: m_sense_label->setText(tr("measured")); break;
		case ForceSense::Reversed: m_sense_label->setText(tr("measured (reversed)")); break;
	}
}

void WheelMappingDialog::ForgetForceSense() {
	m_config.force_sense = ForceSense::Unknown;
	UpdateSenseLabel();
}

void WheelMappingDialog::SaveFromUi() {
	if (const auto model = m_emulated->currentData().toString(); !model.isEmpty()) {
		m_config.model = model.toStdString();
	}
	m_config.ffb_gain              = static_cast<uint32_t>(m_gain->value());
	m_config.ffb_direction         = static_cast<FfbDirection>(m_direction->currentData().toInt());
	m_config.reverse_effects       = m_reverse->isChecked();
	m_config.physical_rotation_deg = static_cast<uint32_t>(m_rotation->value());
	m_config.soft_lock             = m_soft_lock->isChecked();
	m_config.background_input      = m_background->isChecked();
	m_config.software_conditions   = m_software_conditions->isChecked();
	if (m_ffb_device->count() > 0) {
		m_config.ffb_device = DeviceId::FromU64(m_ffb_device->currentData().toULongLong());
	}
}

bool WheelMappingDialog::Save() {
	SaveFromUi();
	std::string error;
	if (!Config::Wheel::Save(m_path.toStdWString(), m_config, &error)) {
		QMessageBox::warning(this, tr("Racing wheel"),
		                     tr("Cannot save %1: %2").arg(m_path, QString::fromStdString(error)));
		return false;
	}
	return true;
}

void WheelMappingDialog::RefreshDevices() {
	if (!m_sdl_ready) {
		return;
	}
	SDL_UpdateJoysticks();
	int                 count = 0;
	SDL_JoystickID*     ids   = SDL_GetJoysticks(&count);
	std::vector<Device> keep;
	bool                added = false;
	for (int i = 0; i < count; i++) {
		auto it = std::find_if(m_devices.begin(), m_devices.end(),
		                       [&](const Device& d) { return d.id == ids[i]; });
		if (it != m_devices.end()) {
			keep.push_back(*it);
			m_devices.erase(it);
			continue;
		}
		SDL_Joystick* j = SDL_OpenJoystick(ids[i]);
		if (j == nullptr) {
			continue;
		}
		Device d;
		d.id                 = ids[i];
		d.joystick           = j;
		d.device.vendor_id   = SDL_GetJoystickVendor(j);
		d.device.product_id  = SDL_GetJoystickProduct(j);
		d.device.num_axes    = static_cast<uint16_t>(std::max(SDL_GetNumJoystickAxes(j), 0));
		d.device.num_hats    = static_cast<uint16_t>(std::max(SDL_GetNumJoystickHats(j), 0));
		d.device.num_buttons = static_cast<uint16_t>(std::max(SDL_GetNumJoystickButtons(j), 0));
		d.name               = QString::fromUtf8(SDL_GetJoystickName(j));
		for (int a = 0; a < d.device.num_axes; a++) {
			d.axis_min.push_back(ReadAxis(j, a));
			d.axis_max.push_back(d.axis_min.back());
		}
		keep.push_back(d);
		added = true;
	}
	SDL_free(ids);
	const bool removed = !m_devices.empty();
	for (auto& d: m_devices) {
		SDL_CloseJoystick(d.joystick);
	}
	m_devices = std::move(keep);
	// The picker and the status line change with the devices only, so that an open drop-down and
	// the last result shown are left alone. An empty picker is filled from the config.
	if (!added && !removed && m_ffb_device->count() > 0) {
		return;
	}

	// The force-feedback picker: connected haptic devices, plus whatever the config names.
	const auto wanted = m_ffb_device->count() > 0 ? m_ffb_device->currentData().toULongLong()
	                                              : m_config.ffb_device.AsU64();
	m_ffb_device->blockSignals(true);
	m_ffb_device->clear();
	m_ffb_device->addItem(tr("The steering device"), QVariant::fromValue<qulonglong>(0));
	for (const auto& d: m_devices) {
		if (SDL_IsJoystickHaptic(d.joystick)) {
			m_ffb_device->addItem(QStringLiteral("%1 (%2:%3)")
			                          .arg(d.name)
			                          .arg(d.device.vendor_id, 4, 16, QLatin1Char('0'))
			                          .arg(d.device.product_id, 4, 16, QLatin1Char('0')),
			                      QVariant::fromValue<qulonglong>(d.device.AsU64()));
		}
	}
	int index = m_ffb_device->findData(QVariant::fromValue<qulonglong>(wanted));
	if (index < 0 && wanted != 0) {
		m_ffb_device->addItem(tr("Not connected (0x%1)").arg(wanted, 16, 16, QLatin1Char('0')),
		                      QVariant::fromValue<qulonglong>(wanted));
		index = m_ffb_device->count() - 1;
	}
	m_ffb_device->setCurrentIndex(std::max(index, 0));
	m_ffb_device->blockSignals(false);

	if (m_capture_control < 0) {
		m_status->setText(m_devices.empty()
		                      ? tr("No joystick found. Connect and power the wheel.")
		                      : tr("%n device(s) connected. Select a control and press Map.", "",
		                           static_cast<int>(m_devices.size())));
	}
}

const WheelMappingDialog::Device* WheelMappingDialog::FindDevice(const DeviceId& id, int axis,
                                                                 bool haptic) const {
	const Device* first = nullptr;
	for (const auto& d: m_devices) {
		if (d.device != id) {
			continue;
		}
		const bool live = axis >= 0 && axis < static_cast<int>(d.axis_min.size()) &&
		                  d.axis_max[axis] - d.axis_min[axis] > AxisActivity;
		if (haptic ? SDL_IsJoystickHaptic(d.joystick) : axis < 0 || live) {
			return &d;
		}
		first = first != nullptr ? first : &d;
	}
	if (first != nullptr) {
		return first;
	}
	const Device* only = nullptr;
	for (const auto& d: m_devices) {
		if (d.device.vendor_id == id.vendor_id && d.device.product_id == id.product_id) {
			if (only != nullptr) {
				return nullptr;
			}
			only = &d;
		}
	}
	return only;
}

QString WheelMappingDialog::BindingText(const Binding& b) const {
	if (!b.IsBound()) {
		return tr("(not bound)");
	}
	const auto* d    = FindDevice(b.device);
	QString     name = d != nullptr ? d->name
	                                : tr("absent %1:%2")
	                                      .arg(b.device.vendor_id, 4, 16, QLatin1Char('0'))
	                                      .arg(b.device.product_id, 4, 16, QLatin1Char('0'));
	switch (b.type) {
		case BindingType::Axis: return tr("%1: axis %2").arg(name).arg(b.index);
		case BindingType::Button: return tr("%1: button %2").arg(name).arg(b.index);
		case BindingType::Hat:
			return tr("%1: hat %2 %3")
			    .arg(name)
			    .arg(b.index)
			    .arg(QString::fromLatin1(Config::Wheel::HatDirectionName(b.hat).data()));
		case BindingType::None: break;
	}
	return {};
}

QString WheelMappingDialog::ValueText(Control c) const {
	const auto& b = m_config.Get(c);
	if (!b.IsBound()) {
		return {};
	}
	const auto* d = FindDevice(b.device, b.type == BindingType::Axis ? b.index : -1);
	if (d == nullptr) {
		return tr("absent");
	}
	if (Config::Wheel::IsAxisControl(c)) {
		int16_t raw = 0;
		if (b.type == BindingType::Axis) {
			raw = ReadAxis(d->joystick, b.index);
		} else {
			const bool pressed = b.type == BindingType::Button
			                         ? SDL_GetJoystickButton(d->joystick, b.index)
			                         : HatHas(SDL_GetJoystickHat(d->joystick, b.index), b.hat);
			raw                = pressed != b.reverse ? (b.positive ? 32767 : -32768)
			                                          : (c == Control::Steering ? 0 : 32767);
		}
		if (b.type == BindingType::Axis && b.reverse) {
			raw = raw == -32768 ? int16_t {32767} : static_cast<int16_t>(-raw);
		}
		if (c == Control::Steering) {
			return QStringLiteral("%1").arg(raw / 32767.0, 0, 'f', 3);
		}
		const double pressed = (32767.0 - raw) / 65535.0;
		return QStringLiteral("%1 %").arg(std::lround(pressed * 100.0));
	}
	bool pressed = false;
	switch (b.type) {
		case BindingType::Axis: {
			const int v = ReadAxis(d->joystick, b.index) * (b.reverse ? -1 : 1);
			pressed     = b.positive ? v > 16383 : v < -16384;
			break;
		}
		case BindingType::Button:
			pressed = SDL_GetJoystickButton(d->joystick, b.index) != b.reverse;
			break;
		case BindingType::Hat:
			pressed = HatHas(SDL_GetJoystickHat(d->joystick, b.index), b.hat) != b.reverse;
			break;
		case BindingType::None: break;
	}
	return pressed ? tr("pressed") : QString();
}

void WheelMappingDialog::UpdateRow(QTreeWidgetItem* item) {
	const auto c = static_cast<Control>(item->data(0, Qt::UserRole).toInt());
	item->setText(1, BindingText(m_config.Get(c)));
	item->setText(2, ValueText(c));
}

Control WheelMappingDialog::SelectedControl() const {
	const auto* item = m_table->currentItem();
	return item != nullptr ? static_cast<Control>(item->data(0, Qt::UserRole).toInt())
	                       : Control::Count;
}

std::map<uint32_t, WheelMappingDialog::Snapshot> WheelMappingDialog::TakeSnapshot() const {
	std::map<uint32_t, Snapshot> snap;
	for (const auto& d: m_devices) {
		auto& s = snap[d.id];
		for (int a = 0; a < d.device.num_axes; a++) {
			s.axes.push_back(ReadAxis(d.joystick, a));
		}
		for (int b = 0; b < d.device.num_buttons; b++) {
			s.buttons.push_back(SDL_GetJoystickButton(d.joystick, b));
		}
		for (int h = 0; h < d.device.num_hats; h++) {
			s.hats.push_back(SDL_GetJoystickHat(d.joystick, h));
		}
	}
	return snap;
}

void WheelMappingDialog::StartCapture() {
	const auto c = SelectedControl();
	if (c == Control::Count || m_devices.empty()) {
		return;
	}
	m_capture_control  = static_cast<int>(c);
	m_capture_snapshot = TakeSnapshot();
	m_capture_clock.start();
	m_map_button->setEnabled(false);
	m_table->setEnabled(false);
}

void WheelMappingDialog::FinishCapture(const Binding& binding, int16_t rest) {
	const auto c      = static_cast<Control>(m_capture_control);
	m_capture_control = -1;
	m_map_button->setEnabled(true);
	m_table->setEnabled(true);
	if (binding.IsBound()) {
		// A new binding starts clean: reverse, dead zone and curve of the old one do not carry
		// over.
		auto& b = m_config.Get(c);
		b       = binding;
		if (c == Control::Steering) {
			ForgetForceSense();
		}
		// A pedal resting at the negative end is reversed so that rest reads released (as RPCS3's
		// G27 pedals).
		if (Config::Wheel::IsAxisControl(c) && c != Control::Steering &&
		    b.type == BindingType::Axis) {
			b.reverse = rest < 0;
		}
		m_status->setText(tr("%1 bound to %2.").arg(ControlLabel(c), BindingText(b)));
		emit m_table->currentItemChanged(m_table->currentItem(), nullptr);
	} else {
		m_status->setText(tr("Nothing was pressed or moved; %1 unchanged.").arg(ControlLabel(c)));
	}
}

void WheelMappingDialog::ClearBinding() {
	const auto c = SelectedControl();
	if (c != Control::Count) {
		m_config.Get(c) = Binding {};
		if (c == Control::Steering) {
			ForgetForceSense();
		}
		emit m_table->currentItemChanged(m_table->currentItem(), nullptr);
	}
}

void WheelMappingDialog::Tick() {
	if (m_refresh_clock.elapsed() > 1000) {
		RefreshDevices();
		m_refresh_clock.restart();
	} else {
		SDL_UpdateJoysticks();
	}
	// Joystick events are not used here; do not let them pile up in SDL's queue.
	SDL_FlushEvents(SDL_EVENT_JOYSTICK_AXIS_MOTION, SDL_EVENT_JOYSTICK_UPDATE_COMPLETE);

	if (m_capture_control >= 0) {
		const auto c           = static_cast<Control>(m_capture_control);
		const bool axis_wanted = Config::Wheel::IsAxisControl(c);
		const int  left_ms     = CaptureTimeoutMs - static_cast<int>(m_capture_clock.elapsed());
		if (left_ms <= 0) {
			FinishCapture({});
			return;
		}
		m_status->setText(tr("%1: %2 now (%3 s left)")
		                      .arg(ControlLabel(c), axis_wanted
		                                                ? tr("move the axis or press a button")
		                                                : tr("press the button"))
		                      .arg((left_ms + 999) / 1000));
		for (const auto& d: m_devices) {
			const auto it = m_capture_snapshot.find(d.id);
			if (it == m_capture_snapshot.end()) {
				continue;
			}
			const auto& s = it->second;
			for (int a = 0; a < static_cast<int>(s.axes.size()); a++) {
				const int v = ReadAxis(d.joystick, a);
				if (std::abs(v - s.axes[a]) > AxisThreshold) {
					Binding b;
					b.device = d.device;
					b.type   = BindingType::Axis;
					b.index  = a;
					// A button control bound to an axis fires at the end it was moved towards.
					b.positive = !axis_wanted && v > s.axes[a];
					FinishCapture(b, s.axes[a]);
					return;
				}
			}
			for (int i = 0; i < static_cast<int>(s.buttons.size()); i++) {
				if (SDL_GetJoystickButton(d.joystick, i) && !s.buttons[i]) {
					Binding b;
					b.device = d.device;
					b.type   = BindingType::Button;
					b.index  = i;
					FinishCapture(b);
					return;
				}
			}
			for (int h = 0; h < static_cast<int>(s.hats.size()); h++) {
				const uint8_t hat = SDL_GetJoystickHat(d.joystick, h);
				if (hat != s.hats[h] && hat != SDL_HAT_CENTERED) {
					Binding b;
					b.device = d.device;
					b.type   = BindingType::Hat;
					b.index  = h;
					b.hat    = (hat & SDL_HAT_UP)     ? HatDirection::Up
					           : (hat & SDL_HAT_DOWN) ? HatDirection::Down
					           : (hat & SDL_HAT_LEFT) ? HatDirection::Left
					                                  : HatDirection::Right;
					FinishCapture(b);
					return;
				}
			}
		}
	}
	for (auto& d: m_devices) {
		for (size_t a = 0; a < d.axis_min.size(); a++) {
			const int16_t v = ReadAxis(d.joystick, static_cast<int>(a));
			d.axis_min[a]   = std::min(d.axis_min[a], v);
			d.axis_max[a]   = std::max(d.axis_max[a], v);
		}
	}
	for (int i = 0; i < m_table->topLevelItemCount(); i++) {
		UpdateRow(m_table->topLevelItem(i));
	}
}

void WheelMappingDialog::ImportRpcs3() {
	const auto path =
	    QFileDialog::getOpenFileName(this, tr("RPCS3 LogitechG27.yml"), QString(),
	                                 tr("RPCS3 G27 mapping (LogitechG27.yml);;YAML (*.yml)"));
	if (path.isEmpty()) {
		return;
	}
	QFile file(path);
	if (!file.open(QIODevice::ReadOnly)) {
		QMessageBox::warning(this, tr("Racing wheel"), file.errorString());
		return;
	}
	std::string error;
	auto        imported = Config::Wheel::ImportRpcs3G27(file.readAll().toStdString(), &error);
	if (!imported) {
		QMessageBox::warning(this, tr("Racing wheel"),
		                     tr("Not imported: %1").arg(QString::fromStdString(error)));
		return;
	}
	// Bindings, the force-feedback device, its direction encoding and reverse come from RPCS3;
	// strength and rotation stay. The force direction is measured again.
	SaveFromUi();
	m_config.bindings        = imported->bindings;
	m_config.ffb_device      = imported->ffb_device;
	m_config.ffb_direction   = imported->ffb_direction;
	m_config.reverse_effects = imported->reverse_effects;
	m_config.force_sense     = ForceSense::Unknown;
	LoadToUi();
	m_ffb_device->clear();
	RefreshDevices();
	m_status->setText(tr("Imported %1.").arg(path));
}

void WheelMappingDialog::TestForce() {
	SaveFromUi();
	const auto& steer  = m_config.Get(Control::Steering);
	const auto  target = m_config.ffb_device.IsNone() ? steer.device : m_config.ffb_device;
	const auto* d      = FindDevice(target, -1, true);
	if (d == nullptr || !SDL_IsJoystickHaptic(d->joystick)) {
		QMessageBox::information(this, tr("Racing wheel"),
		                         tr("No force-feedback device is connected."));
		return;
	}
	if (QMessageBox::question(this, tr("Test force feedback"),
	                          tr("The wheel will turn briefly one way and then the other, at a "
	                             "low force. Keep your hands near the rim, not on it.")) !=
	    QMessageBox::Yes) {
		return;
	}
	SDL_Haptic* h = SDL_OpenHapticFromJoystick(d->joystick);
	if (h == nullptr) {
		QMessageBox::warning(this, tr("Racing wheel"),
		                     tr("Cannot open force feedback: %1").arg(SDL_GetError()));
		return;
	}
	// As the emulator plays forces: without the device's own centring spring, at full gain.
	const uint32_t features = SDL_GetHapticFeatures(h);
	if ((features & SDL_HAPTIC_AUTOCENTER) != 0) {
		SDL_SetHapticAutocenter(h, 0);
	}
	if ((features & SDL_HAPTIC_GAIN) != 0) {
		SDL_SetHapticGain(h, 100);
	}
	const auto* sd =
	    steer.type == BindingType::Axis ? FindDevice(steer.device, steer.index) : nullptr;
	// The device's own axis, before the binding's reverse: the sense is stored against it.
	auto steering = [&] {
		SDL_UpdateJoysticks();
		return sd != nullptr ? ReadAxis(sd->joystick, steer.index) / 32767.0 : 0.0;
	};
	// 10 % of full force, 300 ms each way, sent as is: the movement measures the wheel's own force
	// sense, which the forces computed by the emulator depend on.
	const double level = 0.10;
	QStringList  report;
	double       moved_by[2] = {0.0, 0.0};
	for (const double sign: {1.0, -1.0}) {
		SDL_HapticEffect e {};
		e.constant.type             = SDL_HAPTIC_CONSTANT;
		e.constant.length           = 300;
		e.constant.level            = static_cast<int16_t>(std::lround(sign * level * 32767.0));
		const auto dir              = static_cast<FfbDirection>(m_direction->currentData().toInt());
		e.constant.direction.type   = dir == FfbDirection::SteeringAxis ? SDL_HAPTIC_STEERING_AXIS
		                              : dir == FfbDirection::Cartesian  ? SDL_HAPTIC_CARTESIAN
		                                                                : SDL_HAPTIC_POLAR;
		e.constant.direction.dir[0] = dir == FfbDirection::Cartesian ? 1 : 0;
		const double before         = steering();
		const int    id             = SDL_CreateHapticEffect(h, &e);
		if (id < 0) {
			report << tr("the device refused the effect: %1").arg(SDL_GetError());
			break;
		}
		SDL_RunHapticEffect(h, id, 1);
		SDL_Delay(320);
		SDL_DestroyHapticEffect(h, id);
		const double moved         = steering() - before;
		moved_by[sign > 0 ? 0 : 1] = moved;
		// As the game sees it, with the binding's reverse.
		const double seen = steer.reverse ? -moved : moved;
		report << tr("%1 force: %2")
		              .arg(sign > 0 ? tr("positive") : tr("negative"))
		              .arg(std::abs(moved) < 0.005 ? tr("no movement")
		                   : seen > 0              ? tr("turned right")
		                                           : tr("turned left"));
		SDL_Delay(600);
	}
	SDL_StopHapticEffects(h);
	SDL_CloseHaptic(h);
	// Both pulses moved the rim, in opposite directions: the wheel's force sense is known.
	constexpr double Clear = 0.005;
	if (moved_by[0] > Clear && moved_by[1] < -Clear) {
		m_config.force_sense = ForceSense::Normal;
		report << tr("force direction measured");
	} else if (moved_by[0] < -Clear && moved_by[1] > Clear) {
		m_config.force_sense = ForceSense::Reversed;
		report << tr("force direction measured (reversed)");
	} else {
		report << tr("unclear, not changed (hands off the rim, then test again)");
	}
	UpdateSenseLabel();
	m_status->setText(report.join(QStringLiteral("; ")));
}
