@echo off
rem As test 1, with the CP frame fence off (KYTY_FRAME_FENCE=0). Water may look white or opaque in this test; that is expected.
copy /y "%~dp0test-presets\7-diagnostics-frame-fence-off.json" "%~dp0u59-preset.json" >nul
start "" "%~dp0launcher.exe"
