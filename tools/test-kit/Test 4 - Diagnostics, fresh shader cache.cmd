@echo off
rem As test 1, after moving the shader cache aside (nothing is deleted; the old one is kept as _PipelineCache-old-<number>).
rem The first start compiles every shader again and is slower.
if exist "%~dp0_PipelineCache" ren "%~dp0_PipelineCache" "_PipelineCache-old-%RANDOM%"
copy /y "%~dp0test-presets\2-diagnostics.json" "%~dp0u59-preset.json" >nul
start "" "%~dp0launcher.exe"
