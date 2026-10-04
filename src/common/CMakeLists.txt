file(GLOB common_src CONFIGURE_DEPENDS
	"*.cpp"
	"logging/*.cpp"
	"platform/*.cpp"
)

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

target_include_directories(common
	PUBLIC ${KYTY_SOURCE_DIR}
	PRIVATE
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
)

get_property(inc_headers TARGET common PROPERTY INCLUDE_DIRECTORIES)
set(check_headers ${CMAKE_CURRENT_SOURCE_DIR})

clang_tidy_check(common "" "${check_headers}" "${inc_headers}")
include_what_you_use(common "${inc_headers}")
