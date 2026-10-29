#ifndef LAUNCHER_INCLUDE_WHEEL_MAPPING_DIALOG_H_
#define LAUNCHER_INCLUDE_WHEEL_MAPPING_DIALOG_H_

#include "common/wheelConfig.h"

#include <QDialog>
#include <QElapsedTimer>
#include <QString>
#include <QTimer>

#include <map>
#include <vector>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSlider;
class QSpinBox;
class QTreeWidget;
class QTreeWidgetItem;
struct SDL_Joystick;

// The racing wheel page: which host device drives each control of the emulated wheel (press or
// move to map, with live values), force-feedback settings, and a short test force. Saves
// wheel.json, which a running emulator picks up within a second.
class WheelMappingDialog final: public QDialog {
public:
	WheelMappingDialog(const QString& config_path, bool enabled, QWidget* parent = nullptr);
	~WheelMappingDialog() override;

	[[nodiscard]] bool Enabled() const;

private:
	struct Device {
		uint32_t                id       = 0;
		SDL_Joystick*           joystick = nullptr;
		Config::Wheel::DeviceId device;
		QString                 name;
		// The range each axis has covered, to tell a live device from a dead identical twin.
		std::vector<int16_t> axis_min;
		std::vector<int16_t> axis_max;
	};
	struct Snapshot {
		std::vector<int16_t> axes;
		std::vector<bool>    buttons;
		std::vector<uint8_t> hats;
	};

	void BuildUi(bool enabled);
	void LoadToUi();
	void SaveFromUi();
	void RefreshDevices();
	void Tick();
	void UpdateRow(QTreeWidgetItem* item);
	void StartCapture();
	void FinishCapture(const Config::Wheel::Binding& binding, int16_t rest = 0);
	void ClearBinding();
	void ImportRpcs3();
	void TestForce();
	void ForgetForceSense();
	void UpdateSenseLabel();
	bool Save();

	// Of identical twins, the one whose axis has moved, or the one with force feedback.
	[[nodiscard]] const Device* FindDevice(const Config::Wheel::DeviceId& id, int axis = -1,
	                                       bool haptic = false) const;
	[[nodiscard]] QString       BindingText(const Config::Wheel::Binding& b) const;
	[[nodiscard]] QString       ValueText(Config::Wheel::Control c) const;
	[[nodiscard]] Config::Wheel::Control       SelectedControl() const;
	[[nodiscard]] std::map<uint32_t, Snapshot> TakeSnapshot() const;

	QString                    m_path;
	Config::Wheel::WheelConfig m_config;
	std::vector<Device>        m_devices;
	bool                       m_sdl_ready = false;

	QTreeWidget*    m_table               = nullptr;
	QComboBox*      m_emulated            = nullptr;
	QSlider*        m_gain                = nullptr;
	QLabel*         m_gain_label          = nullptr;
	QComboBox*      m_direction           = nullptr;
	QCheckBox*      m_reverse             = nullptr;
	QSpinBox*       m_rotation            = nullptr;
	QCheckBox*      m_soft_lock           = nullptr;
	QCheckBox*      m_background          = nullptr;
	QCheckBox*      m_software_conditions = nullptr;
	QComboBox*      m_ffb_device          = nullptr;
	QCheckBox*      m_row_reverse         = nullptr;
	QDoubleSpinBox* m_row_dead_zone       = nullptr;
	QDoubleSpinBox* m_row_curve           = nullptr;
	QPushButton*    m_map_button          = nullptr;
	QPushButton*    m_clear_button        = nullptr;
	QLabel*         m_status              = nullptr;
	QLabel*         m_sense_label         = nullptr;

	QTimer                       m_timer;
	QElapsedTimer                m_refresh_clock;
	int                          m_capture_control = -1;
	QElapsedTimer                m_capture_clock;
	std::map<uint32_t, Snapshot> m_capture_snapshot;
	bool                         m_loading = false;
};

#endif /* LAUNCHER_INCLUDE_WHEEL_MAPPING_DIALOG_H_ */
