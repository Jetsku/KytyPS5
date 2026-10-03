@echo off
rem As test 1, with the Vulkan overlay layers (RTSS, Steam, Rockstar, Nsight and others) off for this launch.
set VK_LOADER_LAYERS_DISABLE=~implicit~
copy /y "%~dp0test-presets\2-diagnostics.json" "%~dp0u59-preset.json" >nul
start "" "%~dp0launcher.exe"
