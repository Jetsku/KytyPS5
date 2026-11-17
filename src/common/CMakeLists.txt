file(GLOB common_src CONFIGURE_DEPENDS
	"*.cpp"
	"logging/*.cpp"
	"platform/*.cpp"
)
# The start-up CPU check is its own baseline-compiled object (kyty_cpu_check, top-level CMakeLists.txt).
list(FILTER common_src EXCLUDE REGEX "/cpuCheck\\.cpp$")

file(GLOB_RECURSE common_headers CONFIGURE_DEPENDS "*.h")

if(KYTY_CLANG_CL)
	set_source_files_properties(${common_src} PROPERTIES COMPILE_FLAGS "-Wno-pragma-pack")
endif()

add_library(common STATIC ${common_src} ${common_headers})

# Loading/watchdog diagnostics handle filesystem and thread-start exceptions locally.
# Other sources retain the project's no-exceptions compilation policy.
if(LINUX)
	set_property(SOURCE profiler.cpp hangWatchdog.cpp APPEND PROPERTY COMPILE_OPTIONS -fexceptions)
endif()
set(localization_dir "${PROJECT_SOURCE_DIR}/assets/localization")
file(GLOB localization_catalogs CONFIGURE_DEPENDS "${localization_dir}/*.json")
set(localization_data "${CMAKE_CURRENT_BINARY_DIR}/localizationData.h")
add_custom_command(
	OUTPUT "${localization_data}"
	COMMAND ${CMAKE_COMMAND} "-DINPUTS=${localization_catalogs}" "-DOUTPUT=${localization_data}"
		-P "${KYTY_SOURCE_DIR}/embed_localization.cmake"
	DEPENDS ${localization_catalogs} "${KYTY_SOURCE_DIR}/embed_localization.cmake"
	COMMENT "Embed notification localization catalogs"
	VERBATIM
)
target_sources(common PRIVATE "${localization_data}")

target_include_directories(common
	PUBLIC ${KYTY_SOURCE_DIR}
	PRIVATE
		${CMAKE_CURRENT_BINARY_DIR}
		${KYTY_THIRD_PARTY_DIR}/cpuinfo/include
)

target_link_libraries(common
	PUBLIC
		fmt::fmt
		Tracy::TracyClient
	PRIVATE
		cpuinfo
		kyty_git_version
		spdlog::spdlog
		ZArchive::zarchive
		nlohmann_json::nlohmann_json
)
if(APPLE)
	# hostStats.cpp reads GPU utilization from the IOAccelerator registry entries.
	target_link_libraries(common PRIVATE "-framework CoreFoundation" "-framework IOKit")
endif()

get_property(inc_headers TARGET common PROPERTY INCLUDE_DIRECTORIES)
set(check_headers ${CMAKE_CURRENT_SOURCE_DIR})

clang_tidy_check(common "" "${check_headers}" "${inc_headers}")
include_what_you_use(common "${inc_headers}")
