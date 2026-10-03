cmake_minimum_required(VERSION 3.18...4.3)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0)
	set(CMAKE_POLICY_VERSION_MINIMUM 3.10 CACHE STRING "Minimum policy version for bundled dependencies")
endif()

get_property(KYTY_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT KYTY_GENERATOR_IS_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

project(Kyty)

find_program(SCCACHE_PROGRAM sccache)
find_program(CCACHE_PROGRAM ccache)

foreach(lang IN ITEMS C CXX)
    # An explicitly empty launcher disables caching for this language.
    if(NOT DEFINED CMAKE_${lang}_COMPILER_LAUNCHER)
        if(SCCACHE_PROGRAM)
            set(CMAKE_${lang}_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
            message(STATUS "sccache enabled for ${lang}: ${SCCACHE_PROGRAM}")
        elseif(CCACHE_PROGRAM AND (CMAKE_${lang}_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC" OR
                                   CMAKE_${lang}_COMPILER_ID STREQUAL "MSVC"))
            # ccache with clang-cl (or cl) and Ninja hands Ninja no header dependencies
            # (`ninja -t deps` shows "#deps 0"): editing a header then rebuilds nothing that
            # includes it, and a changed class layout links old and new objects together.
            message(STATUS "ccache not used for ${lang}: it breaks header dependencies with an "
                           "MSVC-style compiler (set CMAKE_${lang}_COMPILER_LAUNCHER to force it)")
        elseif(CCACHE_PROGRAM)
            set(CMAKE_${lang}_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
            message(STATUS "ccache enabled for ${lang}: ${CCACHE_PROGRAM}")
        endif()
    endif()
endforeach()

set(KYTY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(KYTY_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests")
set(KYTY_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")

if(CMAKE_SYSTEM_NAME MATCHES ".*Linux")
	set(LINUX TRUE)
endif()

if (NOT (WIN32 OR LINUX OR APPLE))
    message(FATAL_ERROR "only Windows, Linux, and macOS builds are supported")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

include("${KYTY_SOURCE_DIR}/utils.cmake")
include(CTest)

option(KYTY_ENABLE_CLANG_TIDY "Run clang-tidy checks during builds" OFF)

set(KYTY_BUILD_ORIGIN "Source" CACHE STRING "Build provenance shown in diagnostics")
set_property(CACHE KYTY_BUILD_ORIGIN PROPERTY STRINGS Source Official Fork)
if(NOT KYTY_BUILD_ORIGIN STREQUAL "Source" AND NOT KYTY_BUILD_ORIGIN STREQUAL "Official" AND
   NOT KYTY_BUILD_ORIGIN STREQUAL "Fork")
	message(FATAL_ERROR "KYTY_BUILD_ORIGIN must be Source, Official, or Fork")
endif()

set(KYTY_RELEASE_TAG "" CACHE STRING "Release tag embedded in official builds")
set(KYTY_BUILD_REPOSITORY "" CACHE STRING "GitHub repository embedded in fork builds")
set(KYTY_OFFICIAL_BUILD OFF)
set(KYTY_FORK_BUILD OFF)
if(KYTY_BUILD_ORIGIN STREQUAL "Official")
	if(KYTY_RELEASE_TAG STREQUAL "")
		message(FATAL_ERROR "Official builds require KYTY_RELEASE_TAG")
	endif()
	set(KYTY_OFFICIAL_BUILD ON)
elseif(KYTY_BUILD_ORIGIN STREQUAL "Fork")
	if(KYTY_BUILD_REPOSITORY STREQUAL "")
		message(FATAL_ERROR "Fork builds require KYTY_BUILD_REPOSITORY")
	endif()
	set(KYTY_FORK_BUILD ON)
endif()

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

include(TestBigEndian)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
	message(FATAL_ERROR "only 64-bit builds are supported")
endif()

test_big_endian(BIG_ENDIAN)
if(BIG_ENDIAN)
	set(KYTY_ENDIAN KYTY_ENDIAN_BIG)
else()
	set(KYTY_ENDIAN KYTY_ENDIAN_LITTLE)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
	set(KYTY_BUILD KYTY_BUILD_DEBUG)
else()
	set(KYTY_BUILD KYTY_BUILD_RELEASE)
endif()

if(LINUX OR APPLE)
	# macOS rides the POSIX/Linux code paths until it gets a dedicated platform
	set(KYTY_PLATFORM KYTY_PLATFORM_LINUX)
else()
	set(KYTY_PLATFORM KYTY_PLATFORM_WINDOWS)
endif()

set(CLANG FALSE)
set(GCC FALSE)
set(KYTY_CLANG_CL FALSE)
get_filename_component(KYTY_CXX_COMPILER_NAME "${CMAKE_CXX_COMPILER}" NAME_WE)
string(TOLOWER "${KYTY_CXX_COMPILER_NAME}" KYTY_CXX_COMPILER_NAME)

if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
	set(CLANG TRUE)
	set(KYTY_COMPILER CLANG)
	if(WIN32 AND KYTY_CXX_COMPILER_NAME STREQUAL "clang-cl")
		set(KYTY_CLANG_CL TRUE)
		set(KYTY_LINKER LLD_LINK)
	elseif(APPLE)
		set(KYTY_LINKER LD64) # Apple's default linker; lld flags don't apply
	else()
		set(KYTY_LINKER LLD)
	endif()
	if ((CMAKE_AR MATCHES "llvm") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12.0.0) AND (CMAKE_CXX_COMPILER MATCHES "winlibs"))
		string(REPLACE "llvm-ar.exe" "ar.exe" CMAKE_AR ${CMAKE_AR}) # llvm-ar.exe 11.0.0 from winlibs is broken
	endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	set(GCC TRUE)
	set(KYTY_COMPILER GCC)
	set(KYTY_LINKER LD)
else()
	message(FATAL_ERROR "only Clang and GCC are supported; found ${CMAKE_CXX_COMPILER_ID}")
endif()

set(KYTY_PROJECT_NAME "Emulator" CACHE STRING "Project name suffix")

string(TOLOWER ${KYTY_COMPILER} KYTY_COMPILER_ID)
string(TOLOWER ${KYTY_LINKER} KYTY_LINKER_ID)

if (CLANG AND (KYTY_LINKER STREQUAL LLD))
	set(KYTY_LD_OPTIONS "-fuse-ld=lld")
endif()

if (KYTY_LINKER STREQUAL LD AND NOT LINUX)
	set(KYTY_LD_OPTIONS "-Wl,--image-base=0x100000000000")
endif()

project(Kyty${KYTY_PROJECT_NAME}${CMAKE_BUILD_TYPE}${KYTY_COMPILER} VERSION 0.3.0)

include_directories(
	${KYTY_THIRD_PARTY_DIR}/gtest/include
	${KYTY_THIRD_PARTY_DIR}/gtest
	${KYTY_THIRD_PARTY_DIR}/fmt/include
	${KYTY_THIRD_PARTY_DIR}/magic_enum/include/magic_enum
	${PROJECT_BINARY_DIR}
	${KYTY_SOURCE_DIR}
)

set(KYTY_VERSION "${PROJECT_VERSION}")

configure_file(
  ${KYTY_SOURCE_DIR}/cmake_config.h.in
  ${PROJECT_BINARY_DIR}/cmake_config.h
  )

find_package(Git)

add_custom_target(KytyGitVersion
	COMMAND ${CMAKE_COMMAND}
		-D INPUT_FILE=${KYTY_SOURCE_DIR}/kytyGitVersion.h.in
		-D OUTPUT_FILE=${PROJECT_BINARY_DIR}/kytyGitVersion.h
		-D GIT_EXECUTABLE=${GIT_EXECUTABLE}
		-D GIT_WORKING_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}
		-P ${KYTY_SOURCE_DIR}/generate_version.cmake
	BYPRODUCTS ${PROJECT_BINARY_DIR}/kytyGitVersion.h
	COMMENT "Generate kytyGitVersion.h"
)

add_library(kyty_git_version INTERFACE)
add_dependencies(kyty_git_version KytyGitVersion)

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0)
	list(APPEND KYTY_IWYU
		kyty_emulator
		common
		#launcher
	)
	list(APPEND KYTY_CLANG_TIDY
		kyty_emulator
		#common
		launcher
	)
endif()

option(KYTY_BUILD_LAUNCHER "Build Qt launcher" ON)

config_compiler_and_linker()

# Host CPU baseline for every target (clang, clang-cl, gcc). Empty: the compiler's default
# (x86-64, SSE2: no POPCNT/BMI/AVX2). x86-64-v3 (AVX2, FMA, BMI1/2, LZCNT, MOVBE) suits releases:
# every CPU that can run PS5 guest code natively (Zen 2 instructions) has it. native or znver4 is
# for a build that only runs on the PC that built it. Floating-point contraction stays off, so
# host float math (constant folding in the recompiler, format conversions) gives the same results
# as the default build. A compile option, not CMAKE_CXX_FLAGS: the codegen version stays.
set(KYTY_MARCH "" CACHE STRING "Host -march (empty: compiler default; x86-64-v3 for releases; native for this PC only)")
if(KYTY_MARCH)
	if(KYTY_CLANG_CL)
		add_compile_options("/clang:-march=${KYTY_MARCH}" "/clang:-ffp-contract=off")
	else()
		add_compile_options("-march=${KYTY_MARCH}" "-ffp-contract=off")
	endif()
	message(STATUS "Host code for -march=${KYTY_MARCH}")
endif()

# zstd enables ASM. The first configure detects clang-cl's ASM as MSVC-like, but CMake does not save that
# (CMakeASMCompiler.cmake.in). On every later CMake run in the same directory the Ninja generator then
# takes the compiler for a GNU one and writes all paths with forward slashes: a full rebuild, and new
# anonymous-namespace names (they hash the source path, see PGO below). Keep the first run's answer.
if(KYTY_CLANG_CL)
	set(CMAKE_ASM_SIMULATE_ID MSVC)
	set(CMAKE_ASM_COMPILER_FRONTEND_VARIANT MSVC)
endif()

add_subdirectory("${KYTY_THIRD_PARTY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/3rdparty")

# Profile-guided optimization (clang and clang-cl). Use a separate build directory for each step:
# 1. KYTY_PGO_GENERATE=ON builds an instrumented emulator. Run it with KYTY_PGO_FILE=<x.profraw>:
#    it writes its counters every KYTY_PGO_DUMP_SECONDS (default 20) to that file (main.cpp), so a
#    run stopped with TerminateProcess keeps its last complete dump.
# 2. llvm-profdata merge -o <x.profdata> <dumps>, with the llvm-profdata next to the compiler.
# 3. KYTY_PGO_USE=<x.profdata> builds the optimized emulator. Use a new file name for a new
#    profile: a compiler cache (sccache) does not see the profile's contents.
# Steps 1 and 3 need the same compile flags: with /Z7 (debug info) clang-cl names and hashes some
# lambdas differently, and their profile data then does not match. Releases build without /Z7.
# The profile finds functions by name, and clang-cl names a function in an anonymous namespace after
# a hash of the compiled .cpp file's path as written on the command line (?A0x<hash>). A profile
# fully matches only a build from the same path, spelled the same way. So both steps configure from
# KYTY_PGO_SOURCE_DIR, a directory junction to the checkout (mklink /J C:\kyty-src <checkout>;
# cmake -S C:\kyty-src): CMake and clang-cl keep the junction's path. The release workflow does the same.
# Only Kyty's own targets (declared below: common, emulator, tests, launcher) get the flags.
# Third-party code that a training run does not exercise (the SPIR-V optimizer, ...) keeps its
# normal optimization instead of being treated as cold. The flags are compile options, not
# CMAKE_CXX_FLAGS, so the codegen version (program cache key, src/codegen_version.cmake) stays.
option(KYTY_PGO_GENERATE "Instrumented build for profile-guided optimization (clang)" OFF)
set(KYTY_PGO_USE "" CACHE FILEPATH "llvm-profdata output to optimize with (clang; empty: none)")
if(KYTY_PGO_GENERATE OR KYTY_PGO_USE)
	if(NOT CLANG)
		message(FATAL_ERROR "KYTY_PGO_GENERATE/KYTY_PGO_USE need clang or clang-cl")
	endif()
	if(KYTY_PGO_GENERATE AND KYTY_PGO_USE)
		message(FATAL_ERROR "KYTY_PGO_GENERATE and KYTY_PGO_USE are separate builds")
	endif()
endif()
set(KYTY_PGO_SOURCE_DIR "C:/kyty-src" CACHE PATH "Source path that PGO profiles are recorded and used from (clang-cl)")
if((KYTY_PGO_GENERATE OR KYTY_PGO_USE) AND KYTY_CLANG_CL AND NOT CMAKE_SOURCE_DIR STREQUAL KYTY_PGO_SOURCE_DIR)
	message(WARNING "PGO: configured from ${CMAKE_SOURCE_DIR}, not from KYTY_PGO_SOURCE_DIR "
	                "(${KYTY_PGO_SOURCE_DIR}): functions in anonymous namespaces get other names than in "
	                "the profile and lose it. Configure with -S ${KYTY_PGO_SOURCE_DIR}, a junction to this checkout.")
endif()
if(KYTY_PGO_GENERATE)
	add_compile_options(-fprofile-instr-generate)
	add_compile_definitions(KYTY_PGO_GENERATE=1)
	if(KYTY_CLANG_CL)
		# lld-link is called directly, so the profile runtime has to be named.
		cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH kyty_llvm_bin)
		file(GLOB kyty_profile_runtime "${kyty_llvm_bin}/../lib/clang/*/lib/windows/clang_rt.profile-x86_64.lib")
		if(NOT kyty_profile_runtime)
			message(FATAL_ERROR "KYTY_PGO_GENERATE: clang_rt.profile-x86_64.lib not found next to ${CMAKE_CXX_COMPILER}")
		endif()
		list(GET kyty_profile_runtime 0 kyty_profile_runtime)
		link_libraries("${kyty_profile_runtime}")
	else()
		add_link_options(-fprofile-instr-generate)
	endif()
	message(STATUS "PGO: instrumented build (run with KYTY_PGO_FILE=<file.profraw>)")
elseif(KYTY_PGO_USE)
	if(NOT EXISTS "${KYTY_PGO_USE}")
		message(FATAL_ERROR "KYTY_PGO_USE: ${KYTY_PGO_USE} does not exist")
	endif()
	# No warnings for sources without profile data (launcher, tests) or per function.
	add_compile_options("-fprofile-instr-use=${KYTY_PGO_USE}" -Wno-profile-instr-unprofiled -Wno-backend-plugin)
	message(STATUS "PGO: optimizing with ${KYTY_PGO_USE}")
endif()

add_subdirectory("${KYTY_SOURCE_DIR}/common" "${CMAKE_CURRENT_BINARY_DIR}/common")

file(GLOB kyty_emulator_src CONFIGURE_DEPENDS
	"${KYTY_SOURCE_DIR}/libs/*.cpp"
	"${KYTY_SOURCE_DIR}/libs/*.h"
	"${KYTY_SOURCE_DIR}/graphics/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/*.h"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/*.h"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/command_processor/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/command_processor/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/cache/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/cache/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/drawPrep/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/drawPrep/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/image/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/image/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/cfg/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/cfg/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/decode/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/decode/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/translate/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/translate/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/backend/spirv/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/backend/spirv/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/opcodes/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/opcodes/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/opcodes/*.inc"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/*.h"
	"${KYTY_SOURCE_DIR}/graphics/presentation/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/presentation/*.h"
	"${KYTY_SOURCE_DIR}/graphics/presentation/window/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/presentation/window/*.h"
	"${KYTY_SOURCE_DIR}/kernel/*.cpp"
	"${KYTY_SOURCE_DIR}/kernel/*.h"
	"${KYTY_SOURCE_DIR}/loader/*.cpp"
	"${KYTY_SOURCE_DIR}/loader/*.h"
)

find_program(KYTY_GLSLANG_VALIDATOR glslangValidator REQUIRED)
set(gpu_tiler_shader_dir "${KYTY_SOURCE_DIR}/graphics/host_gpu/shaders")
set(gpu_tiler_generated_dir "${PROJECT_BINARY_DIR}/gpu_tiler_shaders")
set(gpu_tiler_shader_names
	standard256
	standard4
	standard4_3d
	standard64
	standard64_3d
	prt
	prt_3d
	render_target
	depth
	promote_d16
	demote_d16
	swap_bgra16
)
file(GLOB gpu_tiler_shader_includes CONFIGURE_DEPENDS "${gpu_tiler_shader_dir}/gpu_tiler_*.inc")
foreach(shader_name IN LISTS gpu_tiler_shader_names)
	set(shader_source "${gpu_tiler_shader_dir}/gpu_tiler_${shader_name}.comp")
	set(shader_spv "${gpu_tiler_generated_dir}/gpu_tiler_${shader_name}.spv")
	set(shader_header "${gpu_tiler_generated_dir}/gpu_tiler_${shader_name}_spv.h")
	string(TOUPPER "GPU_TILER_${shader_name}_SPV" shader_symbol)
	add_custom_command(
		OUTPUT "${shader_header}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
		COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
		        "-I${gpu_tiler_shader_dir}" -o "${shader_spv}" "${shader_source}"
		COMMAND ${CMAKE_COMMAND} -DINPUT=${shader_spv} -DOUTPUT=${shader_header}
		        -DSYMBOL=${shader_symbol} -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
		DEPENDS "${shader_source}" ${gpu_tiler_shader_includes}
		VERBATIM
	)
	list(APPEND gpu_tiler_shader_headers "${shader_header}")
endforeach()
# Image variants of the 2D tile families (TileManager::DetileToImage / TileFromImage): the same
# sources with the linear side in a native image instead of a buffer.
set(gpu_tiler_image_family_names
	standard256
	standard4
	standard64
	prt
	render_target
	depth
)
foreach(shader_name IN LISTS gpu_tiler_image_family_names)
	foreach(direction IN ITEMS store load)
		string(TOUPPER "${direction}" direction_upper)
		set(shader_source "${gpu_tiler_shader_dir}/gpu_tiler_${shader_name}.comp")
		set(shader_spv "${gpu_tiler_generated_dir}/gpu_tiler_image_${direction}_${shader_name}.spv")
		set(shader_header
		    "${gpu_tiler_generated_dir}/gpu_tiler_image_${direction}_${shader_name}_spv.h")
		string(TOUPPER "GPU_TILER_IMAGE_${direction}_${shader_name}_SPV" shader_symbol)
		add_custom_command(
			OUTPUT "${shader_header}"
			COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
			COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
			        "-DTILER_IMAGE_${direction_upper}" "-I${gpu_tiler_shader_dir}"
			        -o "${shader_spv}" "${shader_source}"
			COMMAND ${CMAKE_COMMAND} -DINPUT=${shader_spv} -DOUTPUT=${shader_header}
			        -DSYMBOL=${shader_symbol} -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
			DEPENDS "${shader_source}" ${gpu_tiler_shader_includes}
			VERBATIM
		)
		list(APPEND gpu_tiler_shader_headers "${shader_header}")
	endforeach()
endforeach()

set(fault_buffer_shader_source "${gpu_tiler_shader_dir}/fault_buffer_process.comp")
set(fault_buffer_shader_spv "${gpu_tiler_generated_dir}/fault_buffer_process.spv")
set(fault_buffer_shader_header "${gpu_tiler_generated_dir}/fault_buffer_process_spv.h")
add_custom_command(
	OUTPUT "${fault_buffer_shader_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -DCACHING_PAGEBITS=14 -DMAX_PAGE_FAULTS=1024
	        -o "${fault_buffer_shader_spv}" "${fault_buffer_shader_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${fault_buffer_shader_spv} -DOUTPUT=${fault_buffer_shader_header}
	        -DSYMBOL=FAULT_BUFFER_PROCESS_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${fault_buffer_shader_source}"
	VERBATIM
)
list(APPEND kyty_emulator_src "${fault_buffer_shader_header}")
list(APPEND kyty_emulator_src ${gpu_tiler_shader_headers})

set(gpu_blit_generated_dir "${PROJECT_BINARY_DIR}/gpu_blit_shaders")
set(gpu_blit_shader_sources
	"${gpu_tiler_shader_dir}/gpu_blit_fs_triangle.vert"
	"${gpu_tiler_shader_dir}/gpu_blit_color_to_ms_depth.frag"
	"${gpu_tiler_shader_dir}/gpu_blit_depth_to_color32.comp"
	"${gpu_tiler_shader_dir}/gpu_blit_color32_to_depth.frag"
	"${gpu_tiler_shader_dir}/gpu_video_out_overlay.frag"
	"${gpu_tiler_shader_dir}/gpu_video_out_downscale.frag"
)
foreach(shader_source IN LISTS gpu_blit_shader_sources)
	get_filename_component(shader_name "${shader_source}" NAME_WE)
	set(shader_spv "${gpu_blit_generated_dir}/${shader_name}.spv")
	set(shader_header "${gpu_blit_generated_dir}/${shader_name}_spv.h")
	string(TOUPPER "${shader_name}_SPV" shader_symbol)
	add_custom_command(
		OUTPUT "${shader_header}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_blit_generated_dir}"
		COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
		        "-I${gpu_tiler_shader_dir}" -o "${shader_spv}" "${shader_source}"
		COMMAND ${CMAKE_COMMAND} -DINPUT=${shader_spv} -DOUTPUT=${shader_header}
		        -DSYMBOL=${shader_symbol} -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
		DEPENDS "${shader_source}"
		VERBATIM
	)
	list(APPEND gpu_blit_shader_headers "${shader_header}")
endforeach()
list(APPEND kyty_emulator_src ${gpu_blit_shader_headers})

set(gpu_dcc_generated_dir "${PROJECT_BINARY_DIR}/gpu_dcc_shaders")
set(gpu_dcc_shader_names validate clear occlusion)
foreach(shader_name IN LISTS gpu_dcc_shader_names)
	set(shader_source "${gpu_tiler_shader_dir}/gpu_dcc_${shader_name}.comp")
	set(shader_spv "${gpu_dcc_generated_dir}/gpu_dcc_${shader_name}.spv")
	set(shader_header "${gpu_dcc_generated_dir}/gpu_dcc_${shader_name}_spv.h")
	string(TOUPPER "GPU_DCC_${shader_name}_SPV" shader_symbol)
	add_custom_command(
		OUTPUT "${shader_header}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_dcc_generated_dir}"
		COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
		        -o "${shader_spv}" "${shader_source}"
		COMMAND ${CMAKE_COMMAND} -DINPUT=${shader_spv} -DOUTPUT=${shader_header}
		        -DSYMBOL=${shader_symbol} -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
		DEPENDS "${shader_source}"
		VERBATIM
	)
	list(APPEND kyty_emulator_src "${shader_header}")
endforeach()

# Native indirect mesh draws (renderer/meshIndirect.h): guest argument record -> mesh dispatches.
set(gpu_mesh_generated_dir "${PROJECT_BINARY_DIR}/gpu_mesh_shaders")
set(gpu_mesh_indirect_source "${gpu_tiler_shader_dir}/gpu_mesh_indirect.comp")
set(gpu_mesh_indirect_spv "${gpu_mesh_generated_dir}/gpu_mesh_indirect.spv")
set(gpu_mesh_indirect_header "${gpu_mesh_generated_dir}/gpu_mesh_indirect_spv.h")
add_custom_command(
	OUTPUT "${gpu_mesh_indirect_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_mesh_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${gpu_mesh_indirect_spv}" "${gpu_mesh_indirect_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${gpu_mesh_indirect_spv} -DOUTPUT=${gpu_mesh_indirect_header}
	        -DSYMBOL=GPU_MESH_INDIRECT_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${gpu_mesh_indirect_source}"
	VERBATIM
)
list(APPEND kyty_emulator_src "${gpu_mesh_indirect_header}")

# Guest SET_PREDICATION on the GPU (renderer/gpuPredication.h): 64-bit guest predicate -> the
# 32-bit word VK_EXT_conditional_rendering reads.
set(gpu_predicate_source "${gpu_tiler_shader_dir}/gpu_predicate.comp")
set(gpu_predicate_spv "${gpu_mesh_generated_dir}/gpu_predicate.spv")
set(gpu_predicate_header "${gpu_mesh_generated_dir}/gpu_predicate_spv.h")
add_custom_command(
	OUTPUT "${gpu_predicate_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_mesh_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${gpu_predicate_spv}" "${gpu_predicate_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${gpu_predicate_spv} -DOUTPUT=${gpu_predicate_header}
	        -DSYMBOL=GPU_PREDICATE_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${gpu_predicate_source}"
	VERBATIM
)
list(APPEND kyty_emulator_src "${gpu_predicate_header}")

# Persistent program cache (KYTY_PROGRAM_CACHE, pipeline/programDiskCache.h): the codegen version
# (recompiler/CodegenFingerprint.h) hashes these files, the code that determines what translating
# a program produces. src/codegen_version.cmake fails the build when one of them includes a
# project header outside this set (unless the script exempts it).
file(GLOB_RECURSE kyty_codegen_sources CONFIGURE_DEPENDS
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/*.inc"
)
list(APPEND kyty_codegen_sources
	"${KYTY_SOURCE_DIR}/graphics/shader/shader.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderBindings.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderBindings.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderCompiler.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderPixelParameter.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderStaticKey.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_defs.h"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.h"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/programDiskCache.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/programDiskCache.cpp"
)
list(SORT kyty_codegen_sources)
string(REPLACE ";" "\n" kyty_codegen_source_lines "${kyty_codegen_sources}")
file(CONFIGURE OUTPUT "${PROJECT_BINARY_DIR}/kytyCodegenSources.txt"
	CONTENT "${kyty_codegen_source_lines}\n")
set(kyty_codegen_header "${PROJECT_BINARY_DIR}/kytyCodegenVersion.h")
add_custom_command(
	OUTPUT "${kyty_codegen_header}"
	COMMAND ${CMAKE_COMMAND}
	        "-DSOURCE_ROOT=${KYTY_SOURCE_DIR}"
	        "-DFILE_LIST=${PROJECT_BINARY_DIR}/kytyCodegenSources.txt"
	        "-DOUTPUT_FILE=${kyty_codegen_header}"
	        "-DCOMPILER=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} ${CMAKE_BUILD_TYPE} ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}"
	        -P "${KYTY_SOURCE_DIR}/codegen_version.cmake"
	DEPENDS ${kyty_codegen_sources} "${PROJECT_BINARY_DIR}/kytyCodegenSources.txt"
	        "${KYTY_SOURCE_DIR}/codegen_version.cmake"
	COMMENT "Generate kytyCodegenVersion.h"
	VERBATIM
)
list(APPEND kyty_emulator_src "${kyty_codegen_header}")

list(APPEND kyty_emulator_src
	"${KYTY_SOURCE_DIR}/emulator.h"
	"${KYTY_SOURCE_DIR}/emulator.cpp"
)
list(REMOVE_DUPLICATES kyty_emulator_src)

if (KYTY_CLANG_CL)
	set_source_files_properties(${kyty_emulator_src} PROPERTIES COMPILE_FLAGS "-Wno-pragma-pack -Wno-deprecated-declarations -D_TIMESPEC_DEFINED")
endif()

set(kyty_emulator_link_libraries common Vulkan::Headers spirv-tools-opt spirv-tools SDL3::SDL3-static imgui xxhash FFmpeg::ffmpeg fmt::fmt nlohmann_json::nlohmann_json LibAtrac9 kyty_git_version Zydis::Zydis xbyak::xbyak opus)

# Linux system libraries.
if(LINUX)
	find_package(Threads REQUIRED)
	list(APPEND kyty_emulator_link_libraries m ${CMAKE_DL_LIBS} Threads::Threads)
endif()

set(inc_headers
	${KYTY_SOURCE_DIR}
	${KYTY_THIRD_PARTY_DIR}/VulkanMemoryAllocator/include
	${KYTY_THIRD_PARTY_DIR}/SPIRV-Tools/include
	${KYTY_THIRD_PARTY_DIR}/SPIRV-Headers/include
	${KYTY_THIRD_PARTY_DIR}/fmt/include
	${KYTY_THIRD_PARTY_DIR}/xxhash
	${KYTY_THIRD_PARTY_DIR}/cpuinfo/include
	${KYTY_THIRD_PARTY_DIR}/renderdoc
	${KYTY_THIRD_PARTY_DIR}/stb
)

if (KYTY_CLANG_CL)
	list(APPEND kyty_emulator_link_libraries winpthread)
	list(APPEND inc_headers
		${KYTY_THIRD_PARTY_DIR}/winpthread/include
	)
endif()

list(APPEND check_headers
	${KYTY_SOURCE_DIR}
)

function(add_kyty_full_emulator_test target source)
	add_executable(${target} EXCLUDE_FROM_ALL ${source} ${kyty_emulator_src})
	target_link_libraries(${target} ${kyty_emulator_link_libraries})
	target_include_directories(${target} PRIVATE ${inc_headers})
	if (WIN32)
		target_link_libraries(${target} iphlpapi)
	endif()
	if (KYTY_CLANG_CL)
		target_link_libraries(${target} onecore)
		add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${KYTY_THIRD_PARTY_DIR}/winpthread/bin/libwinpthread-1.dll" $<TARGET_FILE_DIR:${target}>/libwinpthread-1.dll)
	endif()
	# The macOS x86_64 guest address space needs its .zerofill segments anchored
	# by linker flags, or the kernel kills the binary on load (posix_spawn EIO).
	configure_macos_guest_address_space(${target})
endfunction()

function(configure_macos_guest_address_space target)
	if(APPLE AND (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR
			(NOT CMAKE_OSX_ARCHITECTURES AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")))
		target_sources(${target} PRIVATE "${KYTY_SOURCE_DIR}/kernel/macosGuestAddressSpace.cpp")
		target_compile_definitions(${target} PRIVATE KYTY_LINKED_GUEST_ADDRESS_SPACE=1)
		target_link_options(${target} PRIVATE
			-Wl,-ld_classic,-no_pie,-no_fixup_chains,-no_huge,-pagezero_size,0x40000,-segaddr,SYSTEM_MANAGED,0x40000,-segaddr,SYSTEM_RESERVED,0x7ffffc000,-segaddr,USER_AREA,0x7000000000,-image_base,0x700000000000)
	endif()
endfunction()

add_kyty_full_emulator_test(shader_cfg_tests "${KYTY_TESTS_DIR}/shaderCfgTests.cpp")

add_executable(scalar_provenance_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ScalarProvenanceTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/CodegenOptions.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/hostMemory.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ReadLaneElimination.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
)
target_link_libraries(scalar_provenance_tests common fmt::fmt)
target_include_directories(scalar_provenance_tests PRIVATE ${inc_headers})

add_executable(page_manager_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PageManagerTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/pageManager.cpp"
)
target_link_libraries(page_manager_tests common)
target_include_directories(page_manager_tests PRIVATE ${inc_headers})

add_executable(archive_file_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ArchiveFileTests.cpp"
)
target_link_libraries(archive_file_tests common ZArchive::zarchive)
target_include_directories(archive_file_tests PRIVATE ${inc_headers})

add_executable(avplayer_file_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/AvPlayerFileTests.cpp")
target_link_libraries(avplayer_file_tests common FFmpeg::ffmpeg ZArchive::zarchive)
target_include_directories(avplayer_file_tests PRIVATE ${inc_headers})

add_executable(bit_array_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/BitArrayTests.cpp"
)
target_include_directories(bit_array_tests PRIVATE ${inc_headers})

# Per-thread profiler counters (common/profiler.h); also prints a timing of the counting paths.
add_executable(profiler_counter_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ProfilerCounterTests.cpp"
)
target_link_libraries(profiler_counter_tests common)
target_include_directories(profiler_counter_tests PRIVATE ${inc_headers})
if(LINUX)
	target_link_libraries(profiler_counter_tests Threads::Threads)
endif()

# KYTY_CPU_RESERVE (common/cpuPlacement.h): layout choice, live placement on this machine (Windows),
# and --bench (off against cp in child processes).
add_executable(cpu_placement_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/CpuPlacementTests.cpp"
)
target_link_libraries(cpu_placement_tests common)
target_include_directories(cpu_placement_tests PRIVATE ${inc_headers})
if(LINUX)
	target_link_libraries(cpu_placement_tests Threads::Threads)
endif()

# Live switches (common/liveSwitch.h, KYTY_LIVE_FILE): parsing, application at the flip, the
# synthetic CP spin and the live-control thread.
add_executable(live_switch_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/LiveSwitchTests.cpp"
)
target_link_libraries(live_switch_tests common)
target_include_directories(live_switch_tests PRIVATE ${inc_headers})
if(LINUX)
	target_link_libraries(live_switch_tests Threads::Threads)
endif()

add_executable(lod_stats_report_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/LodStatsReportTests.cpp"
)
target_include_directories(lod_stats_report_tests PRIVATE ${inc_headers})

add_executable(game_patch_filter_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/GamePatchFilterTests.cpp"
)
target_include_directories(game_patch_filter_tests PRIVATE ${inc_headers})
add_executable(lru_cache_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/LruCacheTests.cpp"
)
target_include_directories(lru_cache_tests PRIVATE ${inc_headers})

# KYTY_FUNCTION_ARRAY_SHRINK: the SPIR-V pass that shrinks Function-storage arrays to their proven
# index bound (graphics/host_gpu/spirvLocalArrays.h).
add_executable(spirv_local_arrays_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/SpirvLocalArraysTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/spirvLocalArrays.cpp"
)
target_link_libraries(spirv_local_arrays_tests spirv-tools)
target_include_directories(spirv_local_arrays_tests PRIVATE ${inc_headers})

# KYTY_VRAM_GC_BUDGET: the budget-relative collection marks (graphics/host_gpu/vramBudget.h).
add_executable(vram_budget_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/VramBudgetTests.cpp")
target_include_directories(vram_budget_tests PRIVATE "${KYTY_SOURCE_DIR}")

add_executable(bda_hot_ranges_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/BdaHotRangesTests.cpp")
target_link_libraries(bda_hot_ranges_tests common)
target_include_directories(bda_hot_ranges_tests PRIVATE ${inc_headers})

add_executable(ime_dialog_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ImeDialogTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/imeCommon.cpp"
	"${KYTY_SOURCE_DIR}/libs/imeDialog.cpp"
)
target_include_directories(ime_dialog_tests PRIVATE ${inc_headers})

add_executable(memory_tracker_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/MemoryTrackerTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/pageManager.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/memoryTracker.cpp"
)
target_link_libraries(memory_tracker_tests fmt::fmt common)
target_include_directories(memory_tracker_tests PRIVATE ${inc_headers})

add_executable(shader_vertex_metadata_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ShaderVertexMetadataTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/hostMemory.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderVertexMetadata.cpp"
)
target_include_directories(shader_vertex_metadata_tests PRIVATE ${inc_headers})

add_executable(resource_materialization_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ResourceMaterializationTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/hostMemory.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
)
target_link_libraries(resource_materialization_tests common fmt::fmt xxhash)
if(LINUX)
	# The shared-plan evaluation case runs std::thread workers.
	target_link_libraries(resource_materialization_tests Threads::Threads)
endif()
target_include_directories(resource_materialization_tests PRIVATE ${inc_headers})

add_executable(resource_tracking_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ResourceTrackingTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/hostMemory.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderPixelParameter.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceTracking.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ShaderInfoCollection.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/BindingLayout.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/WriteRangeAnalysis.cpp"
	# BindingLayout asks LoopGuardApplies (KYTY_LOOP_GUARD) whether a program binds GDS.
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/CodegenOptions.cpp"
)
target_link_libraries(resource_tracking_tests common fmt::fmt)
target_include_directories(resource_tracking_tests PRIVATE ${inc_headers})

add_executable(audio_out2_port_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/AudioOut2PortTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/libAudio2.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(audio_out2_port_tests common fmt::fmt)
target_include_directories(audio_out2_port_tests PRIVATE ${inc_headers})

add_executable(audio_out_timing_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/AudioOutTimingTests.cpp")
target_link_libraries(audio_out_timing_tests common fmt::fmt SDL3::SDL3-static)
target_include_directories(audio_out_timing_tests PRIVATE ${inc_headers})

add_executable(ngs2_sampler_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/Ngs2SamplerTests.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(ngs2_sampler_tests common fmt::fmt LibAtrac9)
target_include_directories(ngs2_sampler_tests PRIVATE ${inc_headers})

add_executable(pad_haptics_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PadHapticsTests.cpp"
)
target_link_libraries(pad_haptics_tests common fmt::fmt SDL3::SDL3-static opus)
target_include_directories(pad_haptics_tests PRIVATE ${inc_headers})

add_executable(controller_settings_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ControllerSettingsTests.cpp"
)
target_link_libraries(controller_settings_tests common fmt::fmt SDL3::SDL3-static)
target_include_directories(controller_settings_tests PRIVATE ${inc_headers})

add_executable(save_data_memory_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/SaveDataMemoryTests.cpp"
)
target_link_libraries(save_data_memory_tests common fmt::fmt)
target_include_directories(save_data_memory_tests PRIVATE ${inc_headers})

add_executable(png_enc_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/PngEncTests.cpp")
target_link_libraries(png_enc_tests common fmt::fmt)
target_include_directories(png_enc_tests PRIVATE ${inc_headers})

add_executable(ces_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/CesTests.cpp")
target_link_libraries(ces_tests common fmt::fmt SDL3::SDL3-static)
target_include_directories(ces_tests PRIVATE ${inc_headers})

add_executable(save_data_dialog_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/SaveDataDialogTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/dialog.cpp"
)
target_link_libraries(save_data_dialog_tests common fmt::fmt)
target_include_directories(save_data_dialog_tests PRIVATE ${inc_headers})

add_kyty_full_emulator_test(http_uri_parse_tests
	"${KYTY_TESTS_DIR}/HttpUriParseTests.cpp")

add_executable(event_queue_lifetime_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/EventQueueLifetimeTests.cpp"
	"${KYTY_SOURCE_DIR}/kernel/eventQueue.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(event_queue_lifetime_tests common fmt::fmt)
target_include_directories(event_queue_lifetime_tests PRIVATE ${inc_headers})

# Kernel event delivery semantics: kqueue-style coalescing, EV_CLEAR/EV_ONESHOT, delivery order.
add_executable(event_queue_semantics_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/EventQueueSemanticsTests.cpp"
	"${KYTY_SOURCE_DIR}/kernel/eventQueue.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(event_queue_semantics_tests common fmt::fmt)
target_include_directories(event_queue_semantics_tests PRIVATE ${inc_headers})

# Service-thread priority, sched_yield and the pending-signal fast path.
add_executable(thread_service_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ThreadServiceTests.cpp"
)
target_link_libraries(thread_service_tests common)
target_include_directories(thread_service_tests PRIVATE ${inc_headers})

add_kyty_full_emulator_test(kernel_file_system_tests
	"${KYTY_TESTS_DIR}/KernelFileSystemTests.cpp")
target_link_libraries(kernel_file_system_tests ZArchive::zarchive)

add_executable(sync_on_address_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/SyncOnAddressTests.cpp"
	"${KYTY_SOURCE_DIR}/kernel/syncOnAddress.cpp"
)
target_link_libraries(sync_on_address_tests common)
target_include_directories(sync_on_address_tests PRIVATE ${inc_headers})

add_executable(image_page_table_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ImagePageTableTests.cpp"
)
target_link_libraries(image_page_table_tests common fmt::fmt)
target_include_directories(image_page_table_tests PRIVATE ${inc_headers})

add_executable(image_cache_gc_policy_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ImageCacheGcPolicyTests.cpp"
)
target_include_directories(image_cache_gc_policy_tests PRIVATE "${KYTY_SOURCE_DIR}")

add_executable(host_input_pulse_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/HostInputPulseTests.cpp")
target_include_directories(host_input_pulse_tests PRIVATE "${KYTY_SOURCE_DIR}")

add_executable(eop_timestamp_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/EopTimestampTests.cpp"
)
target_include_directories(eop_timestamp_tests PRIVATE "${KYTY_SOURCE_DIR}")
if(LINUX)
	target_link_libraries(eop_timestamp_tests Threads::Threads)
endif()

add_executable(draw_prep_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/DrawPrepTests.cpp"
)
target_include_directories(draw_prep_tests PRIVATE ${inc_headers})
# ReadSet digest certificates hash with XXH3.
target_link_libraries(draw_prep_tests xxhash)
if(LINUX)
	target_link_libraries(draw_prep_tests Threads::Threads)
endif()

add_executable(ampr_counter_bank_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/AmprCounterBankTests.cpp"
)
target_include_directories(ampr_counter_bank_tests PRIVATE "${KYTY_SOURCE_DIR}")

# Pure helpers of the texture/descriptor binding path; no Vulkan device is created.
add_executable(binding_path_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/BindingPathTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/descriptorSetReuse.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/pipelineLayoutCache.cpp"
)
target_link_libraries(binding_path_tests common fmt::fmt xxhash Vulkan::Headers)
target_include_directories(binding_path_tests PRIVATE ${inc_headers})

# Tile addressing (guest_gpu/tile.cpp) against maps and mip fetches measured on PS5 hardware.
add_executable(tile_hardware_reference_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/TileHardwareReferenceTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/tile.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
)
target_link_libraries(tile_hardware_reference_tests common fmt::fmt)
target_include_directories(tile_hardware_reference_tests PRIVATE ${inc_headers})

# CP recorder command stream (renderer/commandStream.h): ring, encoder, replay and verify mode,
# without a Vulkan device (KYTY_CP_RECORDER).
add_executable(cp_recorder_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/CpRecorderTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/commandStream.cpp"
)
target_link_libraries(cp_recorder_tests common fmt::fmt Vulkan::Headers)
target_include_directories(cp_recorder_tests PRIVATE ${inc_headers})
if(LINUX)
	target_link_libraries(cp_recorder_tests Threads::Threads)
endif()

# P3 op stream (command_processor/cpOps.h): every op kind through the op ring, wrap-around and a
# producer/consumer thread pair, without a Vulkan device (KYTY_CP_SEQ).
add_executable(cp_sequencer_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/CpSequencerTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/command_processor/cpOps.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/commandStream.cpp"
)
target_link_libraries(cp_sequencer_tests common fmt::fmt Vulkan::Headers xxhash)
target_include_directories(cp_sequencer_tests PRIVATE ${inc_headers})
if(LINUX)
	target_link_libraries(cp_sequencer_tests Threads::Threads)
endif()

add_kyty_full_emulator_test(shader_recompiler_compute_tests "${KYTY_TESTS_DIR}/ShaderRecompilerComputeTests.cpp")

set(gpu_test_generated_dir "${PROJECT_BINARY_DIR}/gpu_test_shaders")
set(gpu_test_ms_depth_source
	"${gpu_tiler_shader_dir}/gpu_test_ms_depth.comp")
set(gpu_test_ms_depth_spv
	"${gpu_test_generated_dir}/gpu_test_ms_depth.spv")
set(gpu_test_ms_depth_header
	"${gpu_test_generated_dir}/gpu_test_ms_depth_spv.h")
add_custom_command(
	OUTPUT "${gpu_test_ms_depth_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_test_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${gpu_test_ms_depth_spv}" "${gpu_test_ms_depth_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${gpu_test_ms_depth_spv}
	        -DOUTPUT=${gpu_test_ms_depth_header} -DSYMBOL=GPU_TEST_MS_DEPTH_SPV
	        -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${gpu_test_ms_depth_source}"
	VERBATIM
)
target_sources(shader_recompiler_compute_tests PRIVATE
	"${gpu_test_ms_depth_header}")

add_kyty_full_emulator_test(virtual_memory_allocation_tests "${KYTY_TESTS_DIR}/VirtualMemoryAllocationTests.cpp")
add_kyty_full_emulator_test(repeat_trace_tests "${KYTY_TESTS_DIR}/RepeatTraceTests.cpp")
add_kyty_full_emulator_test(http_wait_request_tests "${KYTY_TESTS_DIR}/HttpWaitRequestTests.cpp")
add_executable(av_player_sync_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/AvPlayerSyncTests.cpp")
target_include_directories(av_player_sync_tests PRIVATE "${KYTY_SOURCE_DIR}")
# APR host file handle pool (libs/aprHostFilePool.h): bookkeeping, content under concurrency, and
# (with --bench) the cost of an open per read against a pooled handle.
add_executable(apr_host_file_pool_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/AprHostFilePoolTests.cpp")
target_link_libraries(apr_host_file_pool_tests common)
target_include_directories(apr_host_file_pool_tests PRIVATE ${inc_headers})
if(LINUX)
	target_link_libraries(apr_host_file_pool_tests Threads::Threads)
endif()
# Once-only GPU loss report (graphics/host_gpu/deviceLostReport.h): header-only, no Vulkan.
add_executable(device_lost_report_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/DeviceLostReportTests.cpp")
target_include_directories(device_lost_report_tests PRIVATE "${KYTY_SOURCE_DIR}")
if(LINUX)
	target_link_libraries(device_lost_report_tests Threads::Threads)
endif()
# Search for a faulting GPU address in memory (kernel/pointerScan.h): header-only, no OS types.
add_executable(pointer_scan_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/PointerScanTests.cpp")
target_include_directories(pointer_scan_tests PRIVATE "${KYTY_SOURCE_DIR}")

# Bounded, once-only emergency cache save; no Vulkan device or game needed.
add_executable(emergency_save_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/EmergencySaveTests.cpp"
	"${KYTY_SOURCE_DIR}/common/subsystems.cpp")
target_include_directories(emergency_save_tests PRIVATE "${KYTY_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")
if(LINUX)
	target_link_libraries(emergency_save_tests Threads::Threads)
endif()

add_executable(program_cache_emergency_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ProgramCacheEmergencyTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/programDiskCache.cpp")
target_link_libraries(program_cache_emergency_tests common xxhash)
target_include_directories(program_cache_emergency_tests PRIVATE ${inc_headers})
if(LINUX)
	target_link_libraries(program_cache_emergency_tests Threads::Threads)
endif()

add_executable(program_cache_preload_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ProgramCachePreloadTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/programDiskCache.cpp")
target_link_libraries(program_cache_preload_tests common xxhash)
target_include_directories(program_cache_preload_tests PRIVATE ${inc_headers})
if(LINUX)
	target_link_libraries(program_cache_preload_tests Threads::Threads)
endif()
target_compile_definitions(virtual_memory_allocation_tests PRIVATE
	KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS=1)

# These tests use exceptions.
if(NOT KYTY_CLANG_CL)
	foreach(kyty_exception_test scalar_provenance_tests resource_tracking_tests
			virtual_memory_allocation_tests)
		target_compile_options(${kyty_exception_test} PRIVATE -fexceptions)
	endforeach()
endif()

if(BUILD_TESTING)
	add_test(NAME http_wait_request COMMAND $<TARGET_FILE:http_wait_request_tests>)
	add_test(NAME av_player_sync COMMAND $<TARGET_FILE:av_player_sync_tests>)
	add_test(NAME apr_host_file_pool COMMAND $<TARGET_FILE:apr_host_file_pool_tests>)
	add_test(NAME device_lost_report COMMAND $<TARGET_FILE:device_lost_report_tests>)
	add_test(NAME pointer_scan COMMAND $<TARGET_FILE:pointer_scan_tests>)
	add_test(NAME emergency_save COMMAND $<TARGET_FILE:emergency_save_tests>)
	set_tests_properties(emergency_save PROPERTIES TIMEOUT 10)
	add_test(NAME program_cache_emergency COMMAND $<TARGET_FILE:program_cache_emergency_tests>)
	set_tests_properties(program_cache_emergency PROPERTIES TIMEOUT 15)
	add_test(NAME program_cache_preload COMMAND $<TARGET_FILE:program_cache_preload_tests>)
	set_tests_properties(program_cache_preload PROPERTIES TIMEOUT 30)
	set_tests_properties(http_wait_request PROPERTIES TIMEOUT 10)
	add_test(NAME emulator_user_name_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --user-name 1234567890123456)
	add_test(NAME emulator_user_name_too_long
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --user-name 12345678901234567)
	set_tests_properties(emulator_user_name_too_long PROPERTIES WILL_FAIL TRUE)
	add_test(NAME emulator_present_mode_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --present-mode Immediate)
	add_test(NAME emulator_present_mode_invalid
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --present-mode Invalid)
	set_tests_properties(emulator_present_mode_invalid PROPERTIES WILL_FAIL TRUE)
	add_test(NAME ime_dialog COMMAND $<TARGET_FILE:ime_dialog_tests>)
	add_test(NAME upsync2_shader_ir COMMAND $<TARGET_FILE:shader_cfg_tests> --upsync2-only)
	add_test(NAME upsync2_opcodes COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --new-opcodes-only)
	add_test(NAME shader_cfg COMMAND $<TARGET_FILE:shader_cfg_tests>)
	# Post-dominators against a path oracle, before and after structurizing (BryKytyPS5 197b094d).
	add_test(NAME shader_cfg_post_dominators
		COMMAND $<TARGET_FILE:shader_cfg_tests> --post-dominators-only)
	add_test(NAME registered_shader_code_on COMMAND $<TARGET_FILE:shader_cfg_tests> --registered-shader-code-only)
	set_tests_properties(registered_shader_code_on PROPERTIES ENVIRONMENT "KYTY_REGISTERED_SHADER_CODE=1")
	add_test(NAME registered_shader_code_off COMMAND $<TARGET_FILE:shader_cfg_tests> --registered-shader-code-only)
	set_tests_properties(registered_shader_code_off PROPERTIES ENVIRONMENT "KYTY_REGISTERED_SHADER_CODE=0")
	add_test(NAME registered_shader_code_cert COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-code-cert-only)
	set_tests_properties(registered_shader_code_cert PROPERTIES ENVIRONMENT "KYTY_REGISTERED_SHADER_CODE=1")
	add_test(NAME scalar_provenance COMMAND $<TARGET_FILE:scalar_provenance_tests>)
	add_test(NAME image_page_table COMMAND $<TARGET_FILE:image_page_table_tests>)
	add_test(NAME image_cache_gc_policy COMMAND $<TARGET_FILE:image_cache_gc_policy_tests>)
	add_test(NAME ampr_counter_bank COMMAND $<TARGET_FILE:ampr_counter_bank_tests>)
	add_test(NAME draw_prep COMMAND $<TARGET_FILE:draw_prep_tests>)
	add_test(NAME eop_timestamp COMMAND $<TARGET_FILE:eop_timestamp_tests>)
	add_test(NAME host_input_pulse COMMAND $<TARGET_FILE:host_input_pulse_tests>)
	add_test(NAME repeat_trace COMMAND $<TARGET_FILE:repeat_trace_tests>)
	add_test(NAME binding_path COMMAND $<TARGET_FILE:binding_path_tests>)
	add_test(NAME tile_hardware_reference COMMAND $<TARGET_FILE:tile_hardware_reference_tests>)
	add_test(NAME cp_recorder COMMAND $<TARGET_FILE:cp_recorder_tests>)
	add_test(NAME cp_sequencer COMMAND $<TARGET_FILE:cp_sequencer_tests>)
	add_test(NAME image_exact_lookup
	         COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-exact-lookup-only)
	add_test(NAME image_pressure_retirement
	         COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-pressure-retirement-only)
	add_test(NAME present_box_downscale
	         COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --present-box-downscale-only)
	add_test(NAME memory_tracker COMMAND $<TARGET_FILE:memory_tracker_tests>)
	add_test(NAME page_manager COMMAND $<TARGET_FILE:page_manager_tests>)
	add_test(NAME page_manager_protect_reuse COMMAND $<TARGET_FILE:page_manager_tests>)
	add_test(NAME memory_tracker_protect_reuse COMMAND $<TARGET_FILE:memory_tracker_tests>)
	set_tests_properties(page_manager_protect_reuse memory_tracker_protect_reuse PROPERTIES
		ENVIRONMENT "KYTY_PAGE_PROTECT_REUSE=1" TIMEOUT 60)
	add_test(NAME archive_file COMMAND $<TARGET_FILE:archive_file_tests>)
	add_test(NAME avplayer_file COMMAND $<TARGET_FILE:avplayer_file_tests>)
	add_test(NAME bit_array COMMAND $<TARGET_FILE:bit_array_tests>)
	add_test(NAME profiler_counters COMMAND $<TARGET_FILE:profiler_counter_tests>)
	add_test(NAME cpu_placement COMMAND $<TARGET_FILE:cpu_placement_tests>)
	add_test(NAME cpu_placement_recorder COMMAND $<TARGET_FILE:cpu_placement_tests> --mode cp+recorder)
	add_test(NAME live_switch COMMAND $<TARGET_FILE:live_switch_tests>)
	add_test(NAME lod_stats_report COMMAND $<TARGET_FILE:lod_stats_report_tests>)
	add_test(NAME game_patch_filter COMMAND $<TARGET_FILE:game_patch_filter_tests>)
	add_test(NAME lod_stats_codegen
	         COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --lod-stats-codegen-only)
	add_test(NAME gi_probe_codegen
	         COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gi-probe-codegen-only)
	add_test(NAME gi_probe
	         COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gi-probe-only)
	# S_MEMREALTIME reads the host GPU clock (KYTY_REALTIME_CLOCK): codegen and a timed GPU wait.
	add_test(NAME s_memrealtime_clock
	         COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --s-memrealtime-only)
	# KYTY_DISPATCHER_CAP: an endless loop in a dispatcher-fallback shader ends on the GPU.
	add_test(NAME dispatcher_cap
	         COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --dispatcher-cap-only)
	add_test(NAME lru_cache COMMAND $<TARGET_FILE:lru_cache_tests>)
	add_test(NAME spirv_local_arrays COMMAND $<TARGET_FILE:spirv_local_arrays_tests>)
	add_test(NAME vram_budget COMMAND $<TARGET_FILE:vram_budget_tests>)
	add_test(NAME bda_hot_ranges COMMAND $<TARGET_FILE:bda_hot_ranges_tests>)
	add_test(NAME shader_vertex_metadata COMMAND $<TARGET_FILE:shader_vertex_metadata_tests>)
	add_test(NAME resource_materialization COMMAND $<TARGET_FILE:resource_materialization_tests>)
	add_test(NAME resource_tracking COMMAND $<TARGET_FILE:resource_tracking_tests>)
	add_test(NAME event_queue_lifetime COMMAND $<TARGET_FILE:event_queue_lifetime_tests>)
	add_test(NAME event_queue_semantics COMMAND $<TARGET_FILE:event_queue_semantics_tests>)
	add_test(NAME thread_service COMMAND $<TARGET_FILE:thread_service_tests>)
	add_test(NAME kernel_file_system COMMAND $<TARGET_FILE:kernel_file_system_tests>)
	add_test(NAME sync_on_address COMMAND $<TARGET_FILE:sync_on_address_tests>)
	add_test(NAME audio_out2_port COMMAND $<TARGET_FILE:audio_out2_port_tests>)
	add_test(NAME audio_out_timing COMMAND $<TARGET_FILE:audio_out_timing_tests>)
	add_test(NAME ngs2_sampler COMMAND $<TARGET_FILE:ngs2_sampler_tests>)
	add_test(NAME pad_haptics COMMAND $<TARGET_FILE:pad_haptics_tests>)
	add_test(NAME controller_settings COMMAND $<TARGET_FILE:controller_settings_tests>)
	add_test(NAME save_data_memory COMMAND $<TARGET_FILE:save_data_memory_tests>)
	add_test(NAME png_enc COMMAND $<TARGET_FILE:png_enc_tests>)
	add_test(NAME ces COMMAND $<TARGET_FILE:ces_tests>)
	add_test(NAME save_data_dialog COMMAND $<TARGET_FILE:save_data_dialog_tests>)
	add_test(NAME http_uri_parse COMMAND $<TARGET_FILE:http_uri_parse_tests>)
	add_test(NAME shader_recompiler_compute COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	add_test(NAME shader_recompiler_alignbyte
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --alignbyte-only)
	add_test(NAME graphics_pipeline_rasterization
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --polygon-mode-only)
	# KYTY_SKIP_INACTIVE_PS (upstream 6956b454f): a stale PS whose exports are all disabled.
	add_test(NAME graphics_pipeline_rasterization_skip_inactive_ps
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --polygon-mode-only)
	set_tests_properties(graphics_pipeline_rasterization_skip_inactive_ps PROPERTIES
		ENVIRONMENT "KYTY_SKIP_INACTIVE_PS=1")
	add_test(NAME graphics_draw_offsets
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-offset-only)
	add_test(NAME guest_stack_exit
		COMMAND $<TARGET_FILE:virtual_memory_allocation_tests> --guest-stack-exit-only)
	add_test(NAME virtual_memory_allocation
		COMMAND $<TARGET_FILE:virtual_memory_allocation_tests>)
	# KYTY_CLAMP_RANGE_MEMO: verify mode and off.
	add_test(NAME virtual_memory_allocation_clamp_memo_verify
		COMMAND $<TARGET_FILE:virtual_memory_allocation_tests>)
	set_tests_properties(virtual_memory_allocation_clamp_memo_verify PROPERTIES
		ENVIRONMENT "KYTY_CLAMP_RANGE_MEMO_VERIFY=exit")
	add_test(NAME virtual_memory_allocation_clamp_memo_off
		COMMAND $<TARGET_FILE:virtual_memory_allocation_tests>)
	set_tests_properties(virtual_memory_allocation_clamp_memo_off PROPERTIES
		ENVIRONMENT "KYTY_CLAMP_RANGE_MEMO=0")
	add_test(NAME guest_red_zone_patcher
		COMMAND $<TARGET_FILE:virtual_memory_allocation_tests> --red-zone-patcher-only)
	add_test(NAME command_scheduler_timeline
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --scheduler-only)
	# Draw-path pops that wait for the priority runner again, and the runner wait's spin.
	add_test(NAME command_scheduler_ready_ops_wait
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --scheduler-only)
	set_tests_properties(command_scheduler_ready_ops_wait PROPERTIES
		ENVIRONMENT "KYTY_PENDING_OPS_NOWAIT=0")
	# KYTY_GPU_OP_PROFILE_STAMPS: a per-op capture and a pass-stamp capture of the same work.
	add_test(NAME gpu_op_profiler_stamps
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gpu-op-profile-only)
	add_test(NAME command_scheduler_priority_spin
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --scheduler-only)
	set_tests_properties(command_scheduler_priority_spin PROPERTIES
		ENVIRONMENT "KYTY_PRIORITY_WAIT_SPIN_US=500")
	add_test(NAME stream_buffer_ring
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --stream-buffer-only)
	# SRT readiness over a render target the CPU overwrote (Crash 4, KYTY_DCC_GPU=1 EXIT).
	add_test(NAME cpu_overwritten_image
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --cpu-overwritten-image-only)
	add_test(NAME gpu_command_lane
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gpu-command-lane-only)
	add_test(NAME pm4_context_state
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --context-state-only)
	add_test(NAME gpu_tiler
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gpu-tiler-only)
	add_test(NAME tile_layout
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --tile-layout-only)

	add_test(NAME texture_cache_layered_image
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --layered-image-only)
	add_test(NAME host_image_allocation
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --host-image-allocation-only)
	add_test(NAME texture_cache_image_views
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-view-cache-only)
	add_test(NAME texture_cache_storage_sampled
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --storage-sampled-only)
	add_test(NAME texture_cache_null_image_init
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --null-image-init-only)
	add_test(NAME fragment_subgroup_size
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --fragment-subgroup-only)
	set_tests_properties(fragment_subgroup_size PROPERTIES TIMEOUT 30)
	add_test(NAME texture_cache_depth_readback
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --depth-readback-only)
	add_test(NAME compute_meta_clear_classification
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --compute-meta-clear-only)
	add_test(NAME cpu_copy_page_skip_ranges
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-cache-range-only)
	add_test(NAME cpu_copy_page_skip_alias_bytes
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --alias-bytes-only)
	add_test(NAME cpu_copy_page_skip_dirty_gc
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-cache-gc-only)
	set_tests_properties(cpu_copy_page_skip_ranges cpu_copy_page_skip_alias_bytes cpu_copy_page_skip_dirty_gc
		PROPERTIES ENVIRONMENT "KYTY_CPU_COPY_PAGE_SKIP=1")
	add_test(NAME buffer_cache_dirty_gc
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-cache-gc-only)
	add_test(NAME draw_prep_code_cert
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-code-cert-only)
	add_test(NAME draw_prep_code_cert_bytes
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-code-cert-only)
	set_tests_properties(draw_prep_code_cert_bytes PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP_CODE_DIGEST=0")
	# KYTY_BUFFER_RANGE_MEMO and KYTY_HOT_PAGE_CHECK_LIMIT: default, verify mode, both off.
	set(buffer_upload_test_env "KYTY_CP_RECORDER=0;KYTY_UPLOAD_DMA=0;KYTY_FAULT_AHEAD_KB=0;KYTY_HOT_PAGES=1;KYTY_HOT_PAGE_FRAMES=2;KYTY_HOT_PAGE_CHECK_LIMIT=0;KYTY_HOT_PAGE_QUIET_FRAMES=1000;KYTY_FALSE_SHARING_WRITES=1;KYTY_FALSE_SHARING_WRITES_VERIFY=1")
	foreach(upload_batch IN ITEMS 0 1)
		add_test(NAME buffer_upload_coalesce_batch${upload_batch}
			COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-upload-coalesce-only)
		set_tests_properties(buffer_upload_coalesce_batch${upload_batch} PROPERTIES TIMEOUT 60
			ENVIRONMENT "${buffer_upload_test_env};KYTY_UPLOAD_BATCH=${upload_batch}")
	endforeach()
	foreach(merge_hot IN ITEMS 0 1)
		add_test(NAME bda_hot_ranges_gpu_merge${merge_hot}
			COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-upload-coalesce-only)
		set_tests_properties(bda_hot_ranges_gpu_merge${merge_hot} PROPERTIES TIMEOUT 60
			ENVIRONMENT "${buffer_upload_test_env};KYTY_UPLOAD_BATCH=1;KYTY_BDA_INCREMENTAL_SYNC=1;KYTY_BDA_NEW_BUFFER_SYNC=1;KYTY_BDA_HOT_RANGES_MERGE=${merge_hot};KYTY_BDA_HOT_SYNC_VERIFY=exit;KYTY_BDA_DIRTY_LOG_VERIFY=exit;KYTY_PAGE_PROTECT_REUSE=1")
	endforeach()
	add_test(NAME buffer_range_memo
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-range-memo-only)
	add_test(NAME buffer_range_memo_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-range-memo-only)
	set_tests_properties(buffer_range_memo_verify PROPERTIES
		ENVIRONMENT "KYTY_BUFFER_RANGE_MEMO_VERIFY=exit")
	add_test(NAME buffer_range_memo_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-range-memo-only)
	set_tests_properties(buffer_range_memo_off PROPERTIES
		ENVIRONMENT "KYTY_BUFFER_RANGE_MEMO=0;KYTY_HOT_PAGE_CHECK_LIMIT=0")
	# KYTY_BUFFER_DIRTY_QUERY_COMBINED=0: the small-read decision from the two separate queries.
	add_test(NAME buffer_range_memo_separate_queries
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-range-memo-only)
	set_tests_properties(buffer_range_memo_separate_queries PROPERTIES
		ENVIRONMENT "KYTY_BUFFER_RANGE_MEMO=0;KYTY_BUFFER_DIRTY_QUERY_COMBINED=0;KYTY_TRACKER_RELAXED_QUERIES=0")
	# KYTY_TRACKER_RELAXED_QUERIES: verify mode (every lock-free answer checked against the locked
	# queries), and the locked queries with the range memo.
	add_test(NAME buffer_range_memo_relaxed_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-range-memo-only)
	set_tests_properties(buffer_range_memo_relaxed_verify PROPERTIES
		ENVIRONMENT "KYTY_TRACKER_RELAXED_VERIFY=exit")
	add_test(NAME buffer_range_memo_locked_queries
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-range-memo-only)
	set_tests_properties(buffer_range_memo_locked_queries PROPERTIES
		ENVIRONMENT "KYTY_TRACKER_RELAXED_QUERIES=0")
	# Real PM4 ordering points and RELEASE_MEM/WAIT_REG_MEM64, including the threaded resolver.
	foreach(lane IN ITEMS direct inline threaded completion)
		set(guest_sync_seq off)
		set(guest_sync_recorder 0)
		set(guest_sync_submission sync)
		set(guest_sync_label record)
		if(lane STREQUAL inline)
			set(guest_sync_seq inline)
		elseif(lane STREQUAL threaded OR lane STREQUAL completion)
			set(guest_sync_seq 1)
			set(guest_sync_recorder 1)
			set(guest_sync_submission queued)
		endif()
		if(lane STREQUAL completion)
			set(guest_sync_label completion)
		endif()
		add_test(NAME guest_sync_${lane}
			COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --guest-sync-only)
		set_tests_properties(guest_sync_${lane} PROPERTIES TIMEOUT 60
			ENVIRONMENT "KYTY_SYNC_EPOCH=1;KYTY_BDA_INCREMENTAL_SYNC=1;KYTY_BDA_SYNC_EPOCH=1;KYTY_BDA_SYNC_EPOCH_VERIFY=0;KYTY_CP_SEQ_VERIFY=0;KYTY_LABEL_MODE=${guest_sync_label};KYTY_CP_RECORDER=${guest_sync_recorder};KYTY_CP_SEQ=${guest_sync_seq};KYTY_SUBMISSION_MODE=${guest_sync_submission}")
	endforeach()

	# BDA passes after buffers registered between them (new buffer, grown/joined buffer): contents
	# only, so it holds for any pass strategy. Default and as the product runs (incremental).
	add_test(NAME bda_new_buffer
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --bda-new-buffer-only)
	add_test(NAME bda_new_buffer_incremental
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --bda-new-buffer-only)
	set_tests_properties(bda_new_buffer_incremental PROPERTIES
		ENVIRONMENT "KYTY_BDA_INCREMENTAL_SYNC=1")
	# KYTY_BDA_NEW_BUFFER_SYNC: new buffers synchronized alone, no full scan; the verify run follows
	# every dirty-log pass with the full scan it replaced and exits on a page it missed.
	add_test(NAME bda_new_buffer_sync
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --bda-new-buffer-only)
	set_tests_properties(bda_new_buffer_sync PROPERTIES
		ENVIRONMENT "KYTY_BDA_INCREMENTAL_SYNC=1;KYTY_BDA_NEW_BUFFER_SYNC=1")
	add_test(NAME bda_new_buffer_sync_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --bda-new-buffer-only)
	set_tests_properties(bda_new_buffer_sync_verify PROPERTIES
		ENVIRONMENT "KYTY_BDA_INCREMENTAL_SYNC=1;KYTY_BDA_NEW_BUFFER_SYNC=1;KYTY_BDA_DIRTY_LOG_VERIFY=exit")
	# KYTY_BDA_BATCH_PROTECT: the dirty-log pass collects, protects once, then copies. Contents
	# checked on the GPU; the verify run also follows each dirty-log pass with a full scan.
	add_test(NAME bda_new_buffer_batch_protect
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --bda-new-buffer-only)
	set_tests_properties(bda_new_buffer_batch_protect PROPERTIES
		ENVIRONMENT "KYTY_BDA_INCREMENTAL_SYNC=1;KYTY_BDA_NEW_BUFFER_SYNC=1;KYTY_BDA_BATCH_PROTECT=1")
	add_test(NAME bda_new_buffer_batch_protect_sync_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --bda-new-buffer-only)
	set_tests_properties(bda_new_buffer_batch_protect_sync_verify PROPERTIES
		ENVIRONMENT "KYTY_BDA_INCREMENTAL_SYNC=1;KYTY_BDA_NEW_BUFFER_SYNC=1;KYTY_BDA_BATCH_PROTECT=1;KYTY_BDA_DIRTY_LOG_VERIFY=exit")
	# Exercise the new-buffer upload before an otherwise valid epoch/submission skip, and mapping
	# invalidation while a native buffer survives. The fixture checks real GPU contents.
	foreach(bda_case same_epoch same_submission remap)
		add_test(NAME bda_new_buffer_${bda_case}
			COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --bda-new-buffer-only)
		set(bda_case_env "KYTY_BDA_INCREMENTAL_SYNC=1;KYTY_BDA_NEW_BUFFER_SYNC=1;KYTY_TEST_BDA_SAME_EPOCH=1")
		if(bda_case STREQUAL same_submission)
			list(APPEND bda_case_env "KYTY_BDA_SYNC_EPOCH=0" "KYTY_BDA_SYNC_PER_SUBMISSION=1")
		elseif(bda_case STREQUAL remap)
			list(APPEND bda_case_env "KYTY_TEST_BDA_REMAP=1" "KYTY_BDA_DIRTY_LOG_VERIFY=exit")
		endif()
		set_tests_properties(bda_new_buffer_${bda_case} PROPERTIES
			TIMEOUT 60 ENVIRONMENT "${bda_case_env}")
	endforeach()
	# KYTY_BDA_SYNC_EPOCH: default, with incremental synchronization (as the product runs), verify
	# mode (it needs the fault epoch of incremental synchronization), off.
	add_test(NAME bda_sync_epoch
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --bda-sync-epoch-only)
	add_test(NAME bda_sync_epoch_incremental
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --bda-sync-epoch-only)
	set_tests_properties(bda_sync_epoch_incremental PROPERTIES
		ENVIRONMENT "KYTY_BDA_INCREMENTAL_SYNC=1")
	add_test(NAME bda_sync_epoch_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --bda-sync-epoch-only)
	set_tests_properties(bda_sync_epoch_verify PROPERTIES
		ENVIRONMENT "KYTY_BDA_INCREMENTAL_SYNC=1;KYTY_BDA_SYNC_EPOCH_VERIFY=exit")
	add_test(NAME bda_sync_epoch_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --bda-sync-epoch-only)
	set_tests_properties(bda_sync_epoch_off PROPERTIES
		ENVIRONMENT "KYTY_BDA_SYNC_EPOCH=0")
	# KYTY_BDA_SYNC_PER_SUBMISSION (the check turns it on itself): plain and incremental passes.
	add_test(NAME bda_sync_per_submission
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --bda-sync-per-submission-only)
	add_test(NAME bda_sync_per_submission_incremental
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --bda-sync-per-submission-only)
	set_tests_properties(bda_sync_per_submission_incremental PROPERTIES
		ENVIRONMENT "KYTY_BDA_INCREMENTAL_SYNC=1")
	# KYTY_BDA_DIRTY_LOG (with incremental BDA synchronization): verify mode and off.
	add_test(NAME bda_sync_dirty_log_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --bda-sync-epoch-only)
	set_tests_properties(bda_sync_dirty_log_verify PROPERTIES
		ENVIRONMENT "KYTY_BDA_INCREMENTAL_SYNC=1;KYTY_BDA_DIRTY_LOG_VERIFY=exit;KYTY_BDA_HOT_SYNC_VERIFY=exit")
	add_test(NAME bda_sync_dirty_log_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --bda-sync-epoch-only)
	set_tests_properties(bda_sync_dirty_log_off PROPERTIES
		ENVIRONMENT "KYTY_BDA_INCREMENTAL_SYNC=1;KYTY_BDA_DIRTY_LOG=0")
	# KYTY_BINDING_EPOCH_MEMO: default, verify mode, off, and off through KYTY_SYNC_EPOCH=0.
	add_test(NAME binding_epoch_memo
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --binding-epoch-memo-only)
	add_test(NAME binding_epoch_memo_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --binding-epoch-memo-only)
	set_tests_properties(binding_epoch_memo_verify PROPERTIES
		ENVIRONMENT "KYTY_BINDING_EPOCH_MEMO_VERIFY=exit")
	add_test(NAME binding_epoch_memo_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --binding-epoch-memo-only)
	set_tests_properties(binding_epoch_memo_off PROPERTIES
		ENVIRONMENT "KYTY_BINDING_EPOCH_MEMO=0")
	add_test(NAME binding_epoch_memo_no_sync_epoch
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --binding-epoch-memo-only)
	set_tests_properties(binding_epoch_memo_no_sync_epoch PROPERTIES
		ENVIRONMENT "KYTY_SYNC_EPOCH=0")
	add_test(NAME binding_epoch_memo_cross_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --binding-epoch-memo-only)
	set_tests_properties(binding_epoch_memo_cross_off PROPERTIES
		ENVIRONMENT "KYTY_BINDING_MEMO_CROSS_EPOCH=0")
	# KYTY_WRITTEN_SYNC_SKIP: default, verify mode, off.
	add_test(NAME written_sync_skip
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --written-sync-skip-only)
	add_test(NAME written_sync_skip_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --written-sync-skip-only)
	set_tests_properties(written_sync_skip_verify PROPERTIES
		ENVIRONMENT "KYTY_WRITTEN_SYNC_SKIP_VERIFY=exit")
	add_test(NAME written_sync_skip_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --written-sync-skip-only)
	set_tests_properties(written_sync_skip_off PROPERTIES
		ENVIRONMENT "KYTY_WRITTEN_SYNC_SKIP=0")
	# KYTY_FALSE_SHARING_WRITES: off (default), on, on with the verify mode.
	add_test(NAME false_sharing_writes_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --false-sharing-writes-only)
	add_test(NAME false_sharing_writes
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --false-sharing-writes-only)
	set_tests_properties(false_sharing_writes PROPERTIES
		ENVIRONMENT "KYTY_FALSE_SHARING_WRITES=1")
	add_test(NAME false_sharing_writes_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --false-sharing-writes-only)
	set_tests_properties(false_sharing_writes_verify PROPERTIES
		ENVIRONMENT "KYTY_FALSE_SHARING_WRITES=1;KYTY_FALSE_SHARING_WRITES_VERIFY=1")
	# KYTY_UPLOAD_DEDUP_TABLE: off (default, the site's last upload only) and on.
	add_test(NAME shader_upload_dedup
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --shader-upload-dedup-only)
	add_test(NAME shader_upload_dedup_table
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --shader-upload-dedup-only)
	set_tests_properties(shader_upload_dedup_table PROPERTIES
		ENVIRONMENT "KYTY_UPLOAD_DEDUP_TABLE=1")
	# KYTY_GPU_WRITE_IMAGE_SKIP: default, verify mode (stops on a mismatch), off.
	add_test(NAME gpu_write_image_skip
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gpu-write-image-skip-only)
	add_test(NAME gpu_write_image_skip_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gpu-write-image-skip-only)
	set_tests_properties(gpu_write_image_skip_verify PROPERTIES
		ENVIRONMENT "KYTY_GPU_WRITE_IMAGE_SKIP_VERIFY=exit")
	add_test(NAME gpu_write_image_skip_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gpu-write-image-skip-only)
	set_tests_properties(gpu_write_image_skip_off PROPERTIES
		ENVIRONMENT "KYTY_GPU_WRITE_IMAGE_SKIP=0")
	# KYTY_DCC_GPU_REFRESH: the refreshed native inspection (with its verify mode) against the
	# CPU fallback, for a buffer-modified and a CPU-dirty image (the check sets the switches).
	add_test(NAME dcc_gpu_refresh
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --dcc-gpu-refresh-only)
	# KYTY_IMAGE_LRU_SKIP: the same LRU order with the touch skip on and off, its verify mode on
	# every decision, and the switch.
	add_test(NAME image_lru_skip
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-lru-skip-only)
	add_test(NAME image_lru_skip_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-lru-skip-only)
	set_tests_properties(image_lru_skip_verify PROPERTIES
		ENVIRONMENT "KYTY_IMAGE_LRU_SKIP_VERIFY=exit")
	add_test(NAME image_lru_skip_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-lru-skip-only)
	set_tests_properties(image_lru_skip_off PROPERTIES
		ENVIRONMENT "KYTY_IMAGE_LRU_SKIP=0")
	# KYTY_IMAGE_TRANSIT_SKIP: the no-op pre-check against GetBarriers, and Transit's skips (or
	# verify checks).
	add_test(NAME image_transit_skip
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-transit-skip-only)
	add_test(NAME image_transit_skip_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-transit-skip-only)
	set_tests_properties(image_transit_skip_verify PROPERTIES
		ENVIRONMENT "KYTY_IMAGE_TRANSIT_SKIP_VERIFY=exit")
	add_test(NAME image_transit_skip_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-transit-skip-only)
	set_tests_properties(image_transit_skip_off PROPERTIES
		ENVIRONMENT "KYTY_IMAGE_TRANSIT_SKIP=0")
	# Tracker-gap detectors: the page states they classify host writes by.
	add_test(NAME tracker_gap_detectors
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --tracker-gap-detectors-only)
	add_test(NAME tracker_gap_host_write_tracking_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --tracker-gap-detectors-only)
	set_tests_properties(tracker_gap_host_write_tracking_off PROPERTIES
		ENVIRONMENT "KYTY_HOST_WRITE_TRACKING=0")
	# KYTY_NATIVE_INDIRECT_MESH: the GPU conversion against the CPU path (default mode, and the
	# completion verify with exit), and the mesh shaders' parameter-block codegen.
	add_test(NAME mesh_indirect
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --mesh-indirect-only)
	add_test(NAME mesh_indirect_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --mesh-indirect-only)
	set_tests_properties(mesh_indirect_verify PROPERTIES
		ENVIRONMENT "KYTY_NATIVE_INDIRECT_MESH=exit")
	add_test(NAME mesh_indirect_on
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --mesh-indirect-only)
	set_tests_properties(mesh_indirect_on PROPERTIES
		ENVIRONMENT "KYTY_NATIVE_INDIRECT_MESH=1")
	add_test(NAME mesh_indirect_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --mesh-indirect-only)
	set_tests_properties(mesh_indirect_off PROPERTIES
		ENVIRONMENT "KYTY_NATIVE_INDIRECT_MESH=0")
	add_test(NAME mesh_indirect_codegen
		COMMAND $<TARGET_FILE:shader_cfg_tests> --mesh-indirect-only)
	# DPP row scans read by V_READLANE become native subgroup reductions (KYTY_LANE_REDUCTIONS);
	# the GPU checks of the reductions and of DPP inactive sources are --wave-reductions-only.
	add_test(NAME wave_row_reduction_codegen
		COMMAND $<TARGET_FILE:shader_cfg_tests> --wave-reduction-only)
	add_test(NAME wave_reductions
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --wave-reductions-only)
	# The set-associative sampler memo; KYTY_SET_REUSE_FRESH: default, off, verify.
	add_test(NAME sampler_memo
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --sampler-memo-only)
	add_test(NAME set_reuse_fresh
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --set-reuse-fresh-only)
	add_test(NAME set_reuse_fresh_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --set-reuse-fresh-only)
	set_tests_properties(set_reuse_fresh_off PROPERTIES ENVIRONMENT "KYTY_SET_REUSE_FRESH=0")
	add_test(NAME set_reuse_fresh_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --set-reuse-fresh-only)
	set_tests_properties(set_reuse_fresh_verify PROPERTIES ENVIRONMENT "KYTY_SET_REUSE_FRESH=verify")
	# KYTY_TEXTURE_MEMO_REVALIDATE: default, verify mode, off.
	add_test(NAME texture_memo_revalidate
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --texture-memo-revalidate-only)
	add_test(NAME texture_memo_revalidate_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --texture-memo-revalidate-only)
	set_tests_properties(texture_memo_revalidate_verify PROPERTIES
		ENVIRONMENT "KYTY_TEXTURE_MEMO_REVALIDATE_VERIFY=exit")
	add_test(NAME texture_memo_revalidate_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --texture-memo-revalidate-only)
	set_tests_properties(texture_memo_revalidate_off PROPERTIES
		ENVIRONMENT "KYTY_TEXTURE_MEMO_REVALIDATE=0")
	# KYTY_DRAW_SEQUENCE_FAST: default, off, and the verify mode over the whole compute/renderer
	# suite (every repeated target lookup and stage texture set is checked against the full path).
	# The same run verifies KYTY_RENDER_STATE_FAST (every partial copy and target entry resolved
	# without value-initialisation against the full one).
	add_test(NAME draw_sequence
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-sequence-only)
	add_test(NAME draw_sequence_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-sequence-only)
	set_tests_properties(draw_sequence_off PROPERTIES
		ENVIRONMENT "KYTY_DRAW_SEQUENCE_FAST=0")
	add_test(NAME draw_sequence_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	set_tests_properties(draw_sequence_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_SEQUENCE_VERIFY=exit;KYTY_RENDER_STATE_VERIFY=exit")
	# KYTY_CP_COMMIT (renderer/cpCommit.h): every part on, alone and with the draw-sequence verify
	# mode over the whole compute/renderer suite.
	add_test(NAME cp_commit
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-sequence-only)
	set_tests_properties(cp_commit PROPERTIES
		ENVIRONMENT "KYTY_CP_COMMIT=all")
	add_test(NAME cp_commit_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	set_tests_properties(cp_commit_verify PROPERTIES
		ENVIRONMENT "KYTY_CP_COMMIT=all;KYTY_DRAW_SEQUENCE_VERIFY=exit;KYTY_RENDER_STATE_VERIFY=exit")
	# KYTY_UPLOAD_DMA_HOST_COPY: the DMA worker copies staged guest bytes (default) or the command
	# processor does.
	add_test(NAME upload_dma
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --upload-dma-only)
	add_test(NAME upload_dma_host_copy_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --upload-dma-only)
	set_tests_properties(upload_dma_host_copy_off PROPERTIES
		ENVIRONMENT "KYTY_UPLOAD_DMA_HOST_COPY=0")
	# KYTY_RENDER_STATE_FAST: default and off.
	add_test(NAME render_state
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --render-state-only)
	add_test(NAME render_state_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --render-state-only)
	set_tests_properties(render_state_off PROPERTIES
		ENVIRONMENT "KYTY_RENDER_STATE_FAST=0")
	# KYTY_DRAW_RUN (drawPrep/drawRun.h): run-level commit of structure-sharing draws against the
	# serial draws, off, on and verified, inline and parallel; and the engine suite with runs on.
	add_test(NAME draw_run_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-run-only)
	set_tests_properties(draw_run_off PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=inline")
	add_test(NAME draw_run_inline
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-run-only)
	set_tests_properties(draw_run_inline PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=inline;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_RUN=1")
	add_test(NAME draw_run_inline_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-run-only)
	set_tests_properties(draw_run_inline_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=inline;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_RUN=exit")
	add_test(NAME draw_run_parallel_bindings
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-run-only)
	set_tests_properties(draw_run_parallel_bindings PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=1;KYTY_DRAW_RUN=1")
	add_test(NAME draw_run_parallel_bindings_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-run-only)
	set_tests_properties(draw_run_parallel_bindings_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=1;KYTY_DRAW_RUN=exit")
	add_test(NAME draw_prep_engine_parallel_run
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_run PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=1;KYTY_DRAW_RUN=1")
	add_test(NAME draw_prep_engine_parallel_run_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_run_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=1;KYTY_DRAW_RUN=exit")
	# KYTY_DRAW_RUN_ACQUIRE: draws that keep their predecessor's attachment acquisition.
	add_test(NAME draw_run_inline_acquire
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-run-only)
	set_tests_properties(draw_run_inline_acquire PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=inline;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_RUN=1;KYTY_DRAW_RUN_ACQUIRE=1")
	add_test(NAME draw_run_inline_acquire_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-run-only)
	set_tests_properties(draw_run_inline_acquire_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=inline;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_RUN=exit;KYTY_DRAW_RUN_ACQUIRE=1")
	add_test(NAME draw_run_parallel_bindings_acquire
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-run-only)
	set_tests_properties(draw_run_parallel_bindings_acquire PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=1;KYTY_DRAW_RUN=1;KYTY_DRAW_RUN_ACQUIRE=1")
	add_test(NAME draw_run_parallel_bindings_acquire_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-run-only)
	set_tests_properties(draw_run_parallel_bindings_acquire_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=1;KYTY_DRAW_RUN=exit;KYTY_DRAW_RUN_ACQUIRE=1")
	add_test(NAME draw_prep_engine_parallel_run_acquire
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_run_acquire PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=1;KYTY_DRAW_RUN=1;KYTY_DRAW_RUN_ACQUIRE=1")
	# The KYTY_DRAW_RUN* flags switched live (common/liveSwitch.h): on, then off again.
	add_test(NAME draw_run_live
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-run-live)
	set_tests_properties(draw_run_live PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=1")
	# KYTY_DRAW_RUN_PUSH: continuations push only their per-draw descriptors.
	add_test(NAME draw_run_inline_push
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-run-only)
	set_tests_properties(draw_run_inline_push PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=inline;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_RUN=1;KYTY_DRAW_RUN_PUSH=1")
	add_test(NAME draw_run_parallel_bindings_push_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-run-only)
	set_tests_properties(draw_run_parallel_bindings_push_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=1;KYTY_DRAW_RUN=exit;KYTY_DRAW_RUN_PUSH=1")
	add_test(NAME draw_run_parallel_bindings_all
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-run-only)
	set_tests_properties(draw_run_parallel_bindings_all PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=1;KYTY_DRAW_RUN=1;KYTY_DRAW_RUN_ACQUIRE=1;KYTY_DRAW_RUN_PUSH=1")
	add_test(NAME draw_prep_engine_parallel_run_all
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_run_all PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=1;KYTY_DRAW_RUN=1;KYTY_DRAW_RUN_ACQUIRE=1;KYTY_DRAW_RUN_PUSH=1")
	# Draw-prep end to end: serial vs command-processor draws, in each engine mode.
	add_test(NAME draw_prep_engine
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	add_test(NAME draw_prep_engine_inline
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_inline PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=inline;KYTY_DRAW_PREP_VERIFY=exit")
	add_test(NAME draw_prep_engine_parallel
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit")
	add_test(NAME draw_prep_engine_parallel_fence
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_fence PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_REG_INDIRECT_WINDOW=0")
	# Value certificates (byte and digest comparisons), and the log-mode audit on top of them.
	add_test(NAME draw_prep_engine_parallel_value
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_value PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_CERT=value")
	add_test(NAME draw_prep_engine_parallel_value_audit
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_value_audit PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_CERT=value;KYTY_DRAW_PREP_LOG_AUDIT=1")
	# KYTY_BACKING_INPLACE and KYTY_DRAW_PREP_LOCKFREE_HINT: verify modes, and both off.
	add_test(NAME draw_prep_engine_parallel_inplace_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_inplace_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_BACKING_INPLACE_VERIFY=exit;KYTY_DRAW_PREP_LOCKFREE_HINT_VERIFY=exit;KYTY_TRACKER_RELAXED_VERIFY=exit;KYTY_BINDING_EPOCH_MEMO_VERIFY=exit;KYTY_DRAW_SEQUENCE_VERIFY=exit;KYTY_RENDER_STATE_VERIFY=exit")
	add_test(NAME draw_prep_engine_render_state_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_render_state_off PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_RENDER_STATE_FAST=0")
	add_test(NAME draw_prep_engine_parallel_copied
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_copied PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_BACKING_INPLACE=0;KYTY_DRAW_PREP_LOCKFREE_HINT=0")
	# KYTY_DRAW_PREP_BINDINGS (P4b): binding plans with every item compared (inline: plans computed
	# on the command processor, counted exactly; parallel, and parallel with one stealing worker),
	# plans used as they are, and the whole compute/renderer suite under the verify mode.
	add_test(NAME draw_prep_engine_inline_bindings_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_inline_bindings_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=inline;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=1;KYTY_DRAW_PREP_BINDINGS_VERIFY=exit")
	add_test(NAME draw_prep_engine_inline_bindings
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_inline_bindings PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=inline;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=1")
	add_test(NAME draw_prep_engine_parallel_bindings_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_bindings_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=1;KYTY_DRAW_PREP_BINDINGS_VERIFY=exit")
	add_test(NAME draw_prep_engine_parallel_bindings
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_bindings PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=1")
	add_test(NAME draw_prep_engine_parallel_steal_bindings_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_steal_bindings_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_STEAL=1;KYTY_DRAW_PREP_WORKERS=1;KYTY_DRAW_PREP_HOT=1;KYTY_DRAW_PREP_BINDINGS=1;KYTY_DRAW_PREP_BINDINGS_VERIFY=exit")
	add_test(NAME draw_prep_engine_inline_bindings_parts
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_inline_bindings_parts PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=inline;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=buffers,userdata,samplers,textures;KYTY_DRAW_PREP_BINDINGS_VERIFY=exit")
	# P4b-1b parts alone: the check verdict and the dynamic viewports, used and verified.
	add_test(NAME draw_prep_engine_inline_bindings_checks
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_inline_bindings_checks PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=inline;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=hwcheck,dynamic")
	add_test(NAME draw_prep_engine_inline_bindings_checks_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_inline_bindings_checks_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=inline;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=hwcheck,dynamic;KYTY_DRAW_PREP_BINDINGS_VERIFY=exit")
	# P4b-2 texture memo hints and runs: alone, verified, and with the draw-sequence texture
	# repeats off (the view runs then take every view).
	add_test(NAME draw_prep_engine_inline_bindings_texturememo
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_inline_bindings_texturememo PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=inline;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=texturememo")
	add_test(NAME draw_prep_engine_inline_bindings_texturememo_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_inline_bindings_texturememo_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=inline;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=texturememo;KYTY_DRAW_PREP_BINDINGS_VERIFY=exit")
	add_test(NAME draw_prep_engine_inline_bindings_texturememo_nosequence
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_inline_bindings_texturememo_nosequence PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=inline;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=texturememo;KYTY_DRAW_SEQUENCE_FAST=0")
	add_test(NAME draw_prep_engine_parallel_bindings_texturememo_nosequence_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_bindings_texturememo_nosequence_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=1;KYTY_DRAW_PREP_BINDINGS_VERIFY=exit;KYTY_DRAW_SEQUENCE_FAST=0")
	add_test(NAME shader_recompiler_compute_bindings_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	set_tests_properties(shader_recompiler_compute_bindings_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_BINDINGS=1;KYTY_DRAW_PREP_BINDINGS_VERIFY=exit")
	# KYTY_DESCRIPTOR_OFFSET_AUDIT (P4b-D0, a measurement): the whole suite with the audit's digests.
	add_test(NAME shader_recompiler_compute_offset_audit
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	set_tests_properties(shader_recompiler_compute_offset_audit PROPERTIES
		ENVIRONMENT "KYTY_DESCRIPTOR_OFFSET_AUDIT=1;KYTY_DESCRIPTOR_SET_REUSE_AUDIT=1")
	# KYTY_DRAW_PREP_STEAL (off by default): the command processor stealing whenever a slot waits,
	# with six workers and with a single one (a held head with unclaimed slots behind it, where it
	# steals most).
	add_test(NAME draw_prep_engine_parallel_steal
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_steal PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_STEAL=1")
	add_test(NAME draw_prep_engine_parallel_steal_one_worker
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_steal_one_worker PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_STEAL=1;KYTY_DRAW_PREP_WORKERS=1;KYTY_DRAW_PREP_HOT=1")
	# KYTY_DRAW_PREP_CERT_RANGES: the preparing thread's certificate ranges checked against the
	# commit-time list (parallel, inline, and the log audit over value certificates), the commit-time
	# path (=commit), and the whole compute/renderer suite under the verify mode.
	add_test(NAME draw_prep_engine_parallel_cert_ranges_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_cert_ranges_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_CERT_RANGES_VERIFY=exit")
	add_test(NAME draw_prep_engine_inline_cert_ranges_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_inline_cert_ranges_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=inline;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_CERT_RANGES_VERIFY=exit")
	add_test(NAME draw_prep_engine_parallel_value_audit_cert_ranges_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_value_audit_cert_ranges_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_CERT=value;KYTY_DRAW_PREP_LOG_AUDIT=1;KYTY_DRAW_PREP_CERT_RANGES_VERIFY=exit")
	add_test(NAME draw_prep_engine_parallel_cert_ranges_commit
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_cert_ranges_commit PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_CERT_RANGES=commit")
	add_test(NAME shader_recompiler_compute_cert_ranges_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	set_tests_properties(shader_recompiler_compute_cert_ranges_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_CERT_RANGES_VERIFY=exit")
	# KYTY_CP_SEQ=inline (cpOps.h): every op through the op ring, executed at once; with
	# KYTY_CP_SEQ_VERIFY the reference front compares every op (exit: stop at the first
	# difference; count mode also checks that a front-state difference is detected). The PM4 and
	# lane checks, the whole compute/renderer suite (also with the draw-prep and CP recorder verify
	# modes), draw prep parallel and inline, and the EOP timestamp modes.
	add_test(NAME cp_seq_inline
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --cp-seq-only)
	set_tests_properties(cp_seq_inline PROPERTIES ENVIRONMENT "KYTY_CP_SEQ=inline")
	add_test(NAME cp_seq_inline_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --cp-seq-only)
	set_tests_properties(cp_seq_inline_verify PROPERTIES
		ENVIRONMENT "KYTY_CP_SEQ=inline;KYTY_CP_SEQ_VERIFY=exit")
	add_test(NAME cp_seq_inline_verify_count
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --cp-seq-only)
	set_tests_properties(cp_seq_inline_verify_count PROPERTIES
		ENVIRONMENT "KYTY_CP_SEQ=inline;KYTY_CP_SEQ_VERIFY=1")
	add_test(NAME cp_seq_direct
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --cp-seq-only)
	add_test(NAME shader_recompiler_compute_cp_seq_inline
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	set_tests_properties(shader_recompiler_compute_cp_seq_inline PROPERTIES
		ENVIRONMENT "KYTY_CP_SEQ=inline")
	add_test(NAME shader_recompiler_compute_cp_seq_inline_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	set_tests_properties(shader_recompiler_compute_cp_seq_inline_verify PROPERTIES
		ENVIRONMENT "KYTY_CP_SEQ=inline;KYTY_CP_SEQ_VERIFY=exit;KYTY_DRAW_PREP_VERIFY=exit")
	add_test(NAME shader_recompiler_compute_cp_seq_inline_recorder
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	set_tests_properties(shader_recompiler_compute_cp_seq_inline_recorder PROPERTIES
		ENVIRONMENT "KYTY_CP_SEQ=inline;KYTY_CP_SEQ_VERIFY=exit;KYTY_CP_RECORDER=1;KYTY_CP_RECORDER_VERIFY=exit")
	add_test(NAME draw_prep_engine_parallel_cp_seq_inline_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_cp_seq_inline_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_CP_SEQ=inline;KYTY_CP_SEQ_VERIFY=exit")
	add_test(NAME draw_prep_engine_inline_cp_seq_inline_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_inline_cp_seq_inline_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=inline;KYTY_DRAW_PREP_VERIFY=exit;KYTY_CP_SEQ=inline;KYTY_CP_SEQ_VERIFY=exit")
	add_test(NAME eop_timestamps_gpu_cp_seq_inline_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --eop-timestamps-only)
	set_tests_properties(eop_timestamps_gpu_cp_seq_inline_verify PROPERTIES
		ENVIRONMENT "KYTY_EOP_TIMESTAMPS=gpu;KYTY_CP_SEQ=inline;KYTY_CP_SEQ_VERIFY=exit")
	add_test(NAME eop_timestamps_gpu_deferred_cp_seq_inline_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --eop-timestamps-only)
	set_tests_properties(eop_timestamps_gpu_deferred_cp_seq_inline_verify PROPERTIES
		ENVIRONMENT "KYTY_EOP_TIMESTAMPS=gpu;KYTY_LABEL_MODE=completion;KYTY_CP_SEQ=inline;KYTY_CP_SEQ_VERIFY=exit")
	# KYTY_CP_SEQ=1 (cpSequencer.h): the graphics front on the sequencer thread, the GPU thread
	# resolving its ops. The PM4 checks and GuestGpu submissions (the lane test: waits, the frame
	# fence, interrupts), the draw-prep engine's draws sequenced with parallel (published per op),
	# inline and off draw prep (register snapshots), the whole suite with the verify modes, with
	# the CP recorder, and the EOP timestamp modes.
	add_test(NAME cp_seq_thread
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --cp-seq-only)
	set_tests_properties(cp_seq_thread PROPERTIES ENVIRONMENT "KYTY_CP_SEQ=1")
	add_test(NAME cp_seq_thread_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --cp-seq-only)
	set_tests_properties(cp_seq_thread_verify PROPERTIES
		ENVIRONMENT "KYTY_CP_SEQ=1;KYTY_CP_SEQ_VERIFY=exit")
	# The park/wake protocol: no spinning on either side, a 2-slot window and 4 snapshots, so the
	# sequencer parks at every barrier, full window and full snapshot ring.
	add_test(NAME cp_seq_thread_park_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --cp-seq-only)
	set_tests_properties(cp_seq_thread_park_verify PROPERTIES
		ENVIRONMENT "KYTY_CP_SEQ=1;KYTY_CP_SEQ_VERIFY=exit;KYTY_CP_SEQ_SPIN_US=0;KYTY_CP_SEQ_RESOLVER_SPIN_US=0;KYTY_CP_SEQ_SNAPSHOTS=4")
	add_test(NAME draw_prep_engine_parallel_cp_seq_thread_park
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_cp_seq_thread_park PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_WINDOW=2;KYTY_DRAW_PREP_VERIFY=exit;KYTY_CP_SEQ=1;KYTY_CP_SEQ_VERIFY=exit;KYTY_CP_SEQ_SPIN_US=0;KYTY_CP_SEQ_RESOLVER_SPIN_US=0;KYTY_CP_SEQ_SNAPSHOTS=4")
	add_test(NAME gpu_command_lane_cp_seq_thread_park
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gpu-command-lane-only)
	set_tests_properties(gpu_command_lane_cp_seq_thread_park PROPERTIES
		ENVIRONMENT "KYTY_CP_SEQ=1;KYTY_CP_SEQ_SPIN_US=0;KYTY_CP_SEQ_RESOLVER_SPIN_US=0;KYTY_CP_SEQ_SNAPSHOTS=4")
	add_test(NAME gpu_command_lane_cp_seq_thread_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gpu-command-lane-only)
	set_tests_properties(gpu_command_lane_cp_seq_thread_verify PROPERTIES
		ENVIRONMENT "KYTY_CP_SEQ=1;KYTY_CP_SEQ_VERIFY=exit")
	add_test(NAME draw_prep_engine_parallel_cp_seq_thread_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_cp_seq_thread_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_CP_SEQ=1;KYTY_CP_SEQ_VERIFY=exit")
	add_test(NAME draw_prep_engine_parallel_steal_cp_seq_thread
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_steal_cp_seq_thread PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_DRAW_PREP_STEAL=1;KYTY_DRAW_PREP_WORKERS=1;KYTY_CP_SEQ=1")
	add_test(NAME draw_prep_engine_inline_cp_seq_thread_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_inline_cp_seq_thread_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=inline;KYTY_DRAW_PREP_VERIFY=exit;KYTY_CP_SEQ=1;KYTY_CP_SEQ_VERIFY=exit")
	add_test(NAME draw_prep_engine_off_cp_seq_thread_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_off_cp_seq_thread_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=off;KYTY_CP_SEQ=1;KYTY_CP_SEQ_VERIFY=exit")
	# KYTY_CP_SEQ_PREFETCH (P3c): draws after a "wait for idle" published speculatively and
	# adopted, dropped (mismatch: retired unused), with a 2-slot window and no spinning (the
	# speculative parse waits for window space), and the lane and whole-suite checks with it on.
	add_test(NAME draw_prep_engine_parallel_cp_seq_thread_prefetch_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_cp_seq_thread_prefetch_verify PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_CP_SEQ=1;KYTY_CP_SEQ_VERIFY=exit;KYTY_CP_SEQ_PREFETCH=1")
	add_test(NAME draw_prep_engine_parallel_cp_seq_thread_prefetch_mismatch
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_cp_seq_thread_prefetch_mismatch PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_CP_SEQ=1;KYTY_CP_SEQ_VERIFY=exit;KYTY_CP_SEQ_PREFETCH=mismatch")
	add_test(NAME draw_prep_engine_parallel_cp_seq_thread_park_prefetch
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_cp_seq_thread_park_prefetch PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_WINDOW=2;KYTY_DRAW_PREP_VERIFY=exit;KYTY_CP_SEQ=1;KYTY_CP_SEQ_VERIFY=exit;KYTY_CP_SEQ_SPIN_US=0;KYTY_CP_SEQ_RESOLVER_SPIN_US=0;KYTY_CP_SEQ_SNAPSHOTS=4;KYTY_CP_SEQ_PREFETCH=1")
	add_test(NAME cp_seq_thread_prefetch_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --cp-seq-only)
	set_tests_properties(cp_seq_thread_prefetch_verify PROPERTIES
		ENVIRONMENT "KYTY_CP_SEQ=1;KYTY_CP_SEQ_VERIFY=exit;KYTY_CP_SEQ_PREFETCH=1")
	add_test(NAME gpu_command_lane_cp_seq_thread_prefetch_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gpu-command-lane-only)
	set_tests_properties(gpu_command_lane_cp_seq_thread_prefetch_verify PROPERTIES
		ENVIRONMENT "KYTY_CP_SEQ=1;KYTY_CP_SEQ_VERIFY=exit;KYTY_CP_SEQ_PREFETCH=1")
	add_test(NAME shader_recompiler_compute_cp_seq_thread_prefetch_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	set_tests_properties(shader_recompiler_compute_cp_seq_thread_prefetch_verify PROPERTIES
		ENVIRONMENT "KYTY_CP_SEQ=1;KYTY_CP_SEQ_VERIFY=exit;KYTY_DRAW_PREP_VERIFY=exit;KYTY_CP_SEQ_PREFETCH=1")
	# KYTY_SUBMIT_MIN_INTERVAL_US: optional submits (idle flushes, end-of-pipe batches) at most every
	# 2 ms; the results are those of every other mode (the lane check predicts tick advances per
	# flush request and is not run with it).
	add_test(NAME draw_prep_engine_parallel_cp_seq_thread_submit_interval
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_parallel_cp_seq_thread_submit_interval PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_CP_SEQ=1;KYTY_CP_SEQ_VERIFY=exit;KYTY_SUBMIT_MIN_INTERVAL_US=2000")
	add_test(NAME shader_recompiler_compute_cp_seq_thread_submit_interval
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	set_tests_properties(shader_recompiler_compute_cp_seq_thread_submit_interval PROPERTIES
		ENVIRONMENT "KYTY_CP_SEQ=1;KYTY_CP_SEQ_VERIFY=exit;KYTY_SUBMIT_MIN_INTERVAL_US=2000")
	add_test(NAME shader_recompiler_compute_submit_interval
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	set_tests_properties(shader_recompiler_compute_submit_interval PROPERTIES
		ENVIRONMENT "KYTY_SUBMIT_MIN_INTERVAL_US=2000")
	add_test(NAME shader_recompiler_compute_cp_seq_thread
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	set_tests_properties(shader_recompiler_compute_cp_seq_thread PROPERTIES
		ENVIRONMENT "KYTY_CP_SEQ=1")
	add_test(NAME shader_recompiler_compute_cp_seq_thread_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	set_tests_properties(shader_recompiler_compute_cp_seq_thread_verify PROPERTIES
		ENVIRONMENT "KYTY_CP_SEQ=1;KYTY_CP_SEQ_VERIFY=exit;KYTY_DRAW_PREP_VERIFY=exit")
	add_test(NAME shader_recompiler_compute_cp_seq_thread_recorder
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	set_tests_properties(shader_recompiler_compute_cp_seq_thread_recorder PROPERTIES
		ENVIRONMENT "KYTY_CP_SEQ=1;KYTY_CP_SEQ_VERIFY=exit;KYTY_CP_RECORDER=1;KYTY_CP_RECORDER_VERIFY=exit")
	add_test(NAME eop_timestamps_gpu_cp_seq_thread_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --eop-timestamps-only)
	set_tests_properties(eop_timestamps_gpu_cp_seq_thread_verify PROPERTIES
		ENVIRONMENT "KYTY_EOP_TIMESTAMPS=gpu;KYTY_CP_SEQ=1;KYTY_CP_SEQ_VERIFY=exit")
	# KYTY_EOP_TIMESTAMPS: record time (default), GPU times, their verify mode, and GPU times
	# delivered by deferred label writes (every label deferred).
	add_test(NAME eop_timestamps
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --eop-timestamps-only)
	add_test(NAME eop_timestamps_gpu
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --eop-timestamps-only)
	set_tests_properties(eop_timestamps_gpu PROPERTIES ENVIRONMENT "KYTY_EOP_TIMESTAMPS=gpu")
	add_test(NAME eop_timestamps_gpu_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --eop-timestamps-only)
	set_tests_properties(eop_timestamps_gpu_verify PROPERTIES
		ENVIRONMENT "KYTY_EOP_TIMESTAMPS=gpu-verify")
	add_test(NAME eop_timestamps_gpu_deferred
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --eop-timestamps-only)
	set_tests_properties(eop_timestamps_gpu_deferred PROPERTIES
		ENVIRONMENT "KYTY_EOP_TIMESTAMPS=gpu;KYTY_LABEL_MODE=completion")
	add_test(NAME draw_prep_code_cert_copied
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-code-cert-only)
	set_tests_properties(draw_prep_code_cert_copied PROPERTIES
		ENVIRONMENT "KYTY_BACKING_INPLACE=0")
	add_test(NAME buffer_cache_eager_readback
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --readback-eager-only)
	# KYTY_ALIAS_BYTES: byte ownership across partially overlapping writes (default), with image
	# writebacks on, and off (whole-image ownership: the checks' failures are only reported).
	add_test(NAME texture_cache_alias_bytes
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --alias-bytes-only)
	add_test(NAME texture_cache_alias_bytes_writeback
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --alias-bytes-only)
	set_tests_properties(texture_cache_alias_bytes_writeback PROPERTIES
		ENVIRONMENT "KYTY_IMAGE_WRITEBACK_ON_GPU_WRITE=1")
	add_test(NAME texture_cache_alias_bytes_off
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --alias-bytes-only)
	set_tests_properties(texture_cache_alias_bytes_off PROPERTIES
		ENVIRONMENT "KYTY_ALIAS_BYTES=0")
	add_test(NAME texture_cache_texel_sync_gpu_dirty
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --texel-sync-gpu-dirty-only)
	add_test(NAME texture_cache_texel_sync_gpu_dirty_whole_images
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --texel-sync-gpu-dirty-only)
	set_tests_properties(texture_cache_texel_sync_gpu_dirty_whole_images PROPERTIES
		ENVIRONMENT "KYTY_ALIAS_BYTES=0")
	# KYTY_SRT_VARIANT_READS: loop-variant scalar reads and runtime S_BUFFER_LOAD V#s through BDA.
	add_test(NAME srt_variant_reads
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --srt-variant-only)
	# KYTY_PROGRAM_CACHE (pipeline/programDiskCache.h): source keys, codegen fingerprint and cache
	# file (truncation, corruption, identity, verification); programs reloaded from the file
	# render what translated ones render, with and without verify mode; and the renderer suites
	# with the cache on and every reloaded program compared with a fresh translation (the files
	# stay in the build directory, so later runs start warm).
	add_test(NAME program_cache
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --program-cache-only)
	add_test(NAME async_pipeline
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --async-pipeline-only)
	set_tests_properties(async_pipeline PROPERTIES TIMEOUT 90
		ENVIRONMENT "KYTY_ASYNC_PIPELINES=1;KYTY_ASYNC_PIPELINE_WAIT_MS=20;KYTY_TEST_ASYNC_PIPELINE_PUBLISH_DURING_WAIT=0;KYTY_PIPELINE_LIBRARY=0;KYTY_PROGRAM_CACHE=0;KYTY_PIPELINE_CACHE_SAVE=0;KYTY_DRAW_PREP=off")
	add_test(NAME async_pipeline_wait_publication
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --async-pipeline-only)
	set_tests_properties(async_pipeline_wait_publication PROPERTIES TIMEOUT 90
		ENVIRONMENT "KYTY_ASYNC_PIPELINES=1;KYTY_ASYNC_PIPELINE_WAIT_MS=1000;KYTY_TEST_ASYNC_PIPELINE_PUBLISH_DURING_WAIT=1;KYTY_PIPELINE_LIBRARY=0;KYTY_PROGRAM_CACHE=0;KYTY_PIPELINE_CACHE_SAVE=0;KYTY_DRAW_PREP=off")
	add_test(NAME async_pipeline_wait_zero
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --async-pipeline-only)
	set_tests_properties(async_pipeline_wait_zero PROPERTIES TIMEOUT 90
		ENVIRONMENT "KYTY_ASYNC_PIPELINES=1;KYTY_ASYNC_PIPELINE_WAIT_MS=0;KYTY_TEST_ASYNC_PIPELINE_PUBLISH_DURING_WAIT=0;KYTY_PIPELINE_LIBRARY=0;KYTY_PROGRAM_CACHE=0;KYTY_PIPELINE_CACHE_SAVE=0;KYTY_DRAW_PREP=off")
	add_test(NAME program_cache_gpu
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --program-cache-gpu-only)
	add_test(NAME program_cache_gpu_verify
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --program-cache-gpu-only)
	set_tests_properties(program_cache_gpu_verify PROPERTIES
		ENVIRONMENT "KYTY_PROGRAM_CACHE_VERIFY=exit")
	add_test(NAME program_cache_gpu_background
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --program-cache-gpu-only)
	set_tests_properties(program_cache_gpu_background PROPERTIES
		ENVIRONMENT "KYTY_PROGRAM_CACHE_VERIFY=background")
	# Verify catches a stored record that differs from a fresh translation, and heals it.
	add_test(NAME program_cache_verify_tamper
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --program-cache-tamper-only)
	set_tests_properties(program_cache_verify_tamper PROPERTIES
		ENVIRONMENT "KYTY_PROGRAM_CACHE_VERIFY=1")
	add_test(NAME program_cache_verify_tamper_background
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --program-cache-tamper-only)
	set_tests_properties(program_cache_verify_tamper_background PROPERTIES
		ENVIRONMENT "KYTY_PROGRAM_CACHE_VERIFY=background")
	add_test(NAME shader_recompiler_compute_program_cache
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	set_tests_properties(shader_recompiler_compute_program_cache PROPERTIES
		ENVIRONMENT "KYTY_PROGRAM_CACHE=1;KYTY_PROGRAM_CACHE_PATH=program-cache-suite.bin;KYTY_PROGRAM_CACHE_VERIFY=exit")
	add_test(NAME draw_prep_engine_program_cache
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-prep-engine-only)
	set_tests_properties(draw_prep_engine_program_cache PROPERTIES
		ENVIRONMENT "KYTY_DRAW_PREP=parallel;KYTY_DRAW_PREP_VERIFY=exit;KYTY_PROGRAM_CACHE=1;KYTY_PROGRAM_CACHE_PATH=program-cache-draw-prep.bin;KYTY_PROGRAM_CACHE_VERIFY=exit")

	# KYTY_CP_RECORDER (renderer/commandRecorder.h): the recorder's own GPU checks (recorder
	# thread, 1 MiB ring, verify log mode), and the renderer and compute suites with the recorder
	# thread or the inline replay under verify=exit (per-packet argument hashes, command-buffer
	# digests, ownership hooks), with synchronous submission (the tests' default) and through
	# the submission broker (KYTY_SUBMISSION_MODE=queued, as the game runs).
	set(cp_recorder_env KYTY_CP_RECORDER=1 KYTY_CP_RECORDER_VERIFY=exit)
	set(cp_recorder_inline_env KYTY_CP_RECORDER=inline KYTY_CP_RECORDER_VERIFY=exit)
	add_test(NAME cp_recorder_gpu
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --cp-recorder-only)
	add_test(NAME cp_recorder_gpu_queued
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --cp-recorder-only)
	set_tests_properties(cp_recorder_gpu_queued PROPERTIES
		ENVIRONMENT "KYTY_SUBMISSION_MODE=queued")
	add_test(NAME cp_recorder_gpu_inline
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --cp-recorder-only)
	set_tests_properties(cp_recorder_gpu_inline PROPERTIES
		ENVIRONMENT "KYTY_CP_RECORDER=inline")
	add_test(NAME shader_recompiler_compute_cp_recorder
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	set_tests_properties(shader_recompiler_compute_cp_recorder PROPERTIES
		ENVIRONMENT "${cp_recorder_env}")
	add_test(NAME shader_recompiler_compute_cp_recorder_queued
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	set_tests_properties(shader_recompiler_compute_cp_recorder_queued PROPERTIES
		ENVIRONMENT "${cp_recorder_env};KYTY_SUBMISSION_MODE=queued")
	add_test(NAME shader_recompiler_compute_cp_recorder_inline
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	set_tests_properties(shader_recompiler_compute_cp_recorder_inline PROPERTIES
		ENVIRONMENT "${cp_recorder_inline_env}")
	# Thread (queued submission) and inline variants of a focused mode; extra arguments are extra
	# environment entries.
	function(add_cp_recorder_variants test_name mode_arg)
		set(thread_env KYTY_CP_RECORDER=1 KYTY_CP_RECORDER_VERIFY=exit KYTY_SUBMISSION_MODE=queued
			${ARGN})
		set(inline_env KYTY_CP_RECORDER=inline KYTY_CP_RECORDER_VERIFY=exit ${ARGN})
		add_test(NAME ${test_name}_cp_recorder
			COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> ${mode_arg})
		set_tests_properties(${test_name}_cp_recorder PROPERTIES ENVIRONMENT "${thread_env}")
		add_test(NAME ${test_name}_cp_recorder_inline
			COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> ${mode_arg})
		set_tests_properties(${test_name}_cp_recorder_inline PROPERTIES ENVIRONMENT "${inline_env}")
	endfunction()
	add_cp_recorder_variants(draw_prep_engine --draw-prep-engine-only
		KYTY_DRAW_PREP=parallel KYTY_DRAW_PREP_VERIFY=exit)
	add_cp_recorder_variants(draw_prep_engine_bindings --draw-prep-engine-only
		KYTY_DRAW_PREP=parallel KYTY_DRAW_PREP_VERIFY=exit KYTY_DRAW_PREP_BINDINGS=1
		KYTY_DRAW_PREP_BINDINGS_VERIFY=exit)
	add_cp_recorder_variants(graphics_pipeline_rasterization --polygon-mode-only)
	add_cp_recorder_variants(command_scheduler_timeline --scheduler-only)
	add_cp_recorder_variants(gpu_command_lane --gpu-command-lane-only)
	add_cp_recorder_variants(pm4_context_state --context-state-only)
	add_cp_recorder_variants(stream_buffer_ring --stream-buffer-only)
	add_cp_recorder_variants(texture_cache_layered_image --layered-image-only)
	add_cp_recorder_variants(compute_meta_clear_classification --compute-meta-clear-only)
	add_cp_recorder_variants(buffer_cache_dirty_gc --buffer-cache-gc-only)
	add_cp_recorder_variants(buffer_cache_eager_readback --readback-eager-only)
	# KYTY_EOP_TIMESTAMPS=gpu with the recorder: its query writes and slot resets are recorder
	# packets (WriteTimestamp2, ResetQueryPool); the clock writes drain nothing. gpu-verify checks
	# the published values.
	add_cp_recorder_variants(eop_timestamps_gpu --eop-timestamps-only KYTY_EOP_TIMESTAMPS=gpu)
	add_cp_recorder_variants(eop_timestamps_gpu_verify --eop-timestamps-only
		KYTY_EOP_TIMESTAMPS=gpu-verify)
	# The recorder under VK_LAYER_KHRONOS_validation (KYTY_TEST_VULKAN_VALIDATION=1: core and
	# thread-safety checks, i.e. command-pool, descriptor-pool, command-buffer and queue use from
	# two threads at once, and descriptor sets updated while a recorded or pending buffer binds
	# them). Each test fails on any validation error; <name>_validation is the same mode with the
	# recorder off (the baseline). Label: validation.
	function(add_cp_recorder_validation test_name mode_arg)
		set(off_env KYTY_CP_RECORDER=0 KYTY_TEST_VULKAN_VALIDATION=1 ${ARGN})
		set(thread_env KYTY_CP_RECORDER=1 KYTY_CP_RECORDER_VERIFY=exit KYTY_SUBMISSION_MODE=queued
			KYTY_TEST_VULKAN_VALIDATION=1 ${ARGN})
		set(inline_env KYTY_CP_RECORDER=inline KYTY_CP_RECORDER_VERIFY=exit
			KYTY_TEST_VULKAN_VALIDATION=1 ${ARGN})
		# --cp-recorder-only needs a recorder: no baseline.
		if(NOT test_name STREQUAL "cp_recorder_gpu")
			add_test(NAME ${test_name}_validation
				COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> ${mode_arg})
			set_tests_properties(${test_name}_validation PROPERTIES
				ENVIRONMENT "${off_env}" LABELS validation)
		endif()
		add_test(NAME ${test_name}_cp_recorder_validation
			COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> ${mode_arg})
		set_tests_properties(${test_name}_cp_recorder_validation PROPERTIES
			ENVIRONMENT "${thread_env}" LABELS validation)
		add_test(NAME ${test_name}_cp_recorder_inline_validation
			COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> ${mode_arg})
		set_tests_properties(${test_name}_cp_recorder_inline_validation PROPERTIES
			ENVIRONMENT "${inline_env}" LABELS validation)
	endfunction()
	add_cp_recorder_validation(cp_recorder_gpu --cp-recorder-only)
	add_cp_recorder_validation(draw_prep_engine --draw-prep-engine-only
		KYTY_DRAW_PREP=parallel KYTY_DRAW_PREP_VERIFY=exit)
	add_cp_recorder_validation(gpu_command_lane --gpu-command-lane-only)
	add_cp_recorder_validation(stream_buffer_ring --stream-buffer-only)
	add_cp_recorder_validation(texture_cache_layered_image --layered-image-only)
	add_cp_recorder_validation(buffer_cache_eager_readback --readback-eager-only)

	if(WIN32)
		# These tests still depend on the Windows multisample-depth path.
		add_test(NAME texture_cache_image_overlap
			COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-overlap-only)
		add_test(NAME texture_cache_htile_clear
			COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --htile-clear-only)
		add_test(NAME buffer_cache_ranges
			COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-cache-range-only)
		add_cp_recorder_variants(texture_cache_image_overlap --image-overlap-only)
		add_cp_recorder_variants(texture_cache_htile_clear --htile-clear-only)
		add_cp_recorder_variants(buffer_cache_ranges --buffer-cache-range-only)
	endif()

	# A failed check fails its test from its message too, not only from the exit status (which
	# a harness can lose, or a process exit can overtake when the check runs on another thread).
	# The failure formats of the test programs: "<Suite>: failed: ...", "<Suite>: <case> failed
	# ...", "FAILED: ...", "... test failed: ..." and "SaveDataMemoryTests:<line>: ...".
	# failure_message_fails_test checks the expression itself: a program that prints a failure
	# and exits with status 0 must fail (inverted by WILL_FAIL).
	add_test(NAME failure_message_fails_test
		COMMAND ${CMAKE_COMMAND} -E echo "SelfTests: failed: a check failed, the process exited 0")
	set_tests_properties(failure_message_fails_test PROPERTIES WILL_FAIL TRUE)
	get_property(kyty_test_names DIRECTORY PROPERTY TESTS)
	set_tests_properties(${kyty_test_names} PROPERTIES FAIL_REGULAR_EXPRESSION
		"Tests: failed: ;Tests: [^\n]* failed at ;Tests: [^\n]* failed: ;FAILED: ;test failed: ;SaveDataMemoryTests:[0-9]+: ")

	add_custom_target(kyty_tests DEPENDS
		http_wait_request_tests
		av_player_sync_tests
		apr_host_file_pool_tests
		device_lost_report_tests
		pointer_scan_tests
		emergency_save_tests
		program_cache_emergency_tests
		program_cache_preload_tests
		ime_dialog_tests
		shader_cfg_tests
		scalar_provenance_tests
		image_page_table_tests
		memory_tracker_tests
		page_manager_tests
		archive_file_tests
		avplayer_file_tests
		bit_array_tests
		lod_stats_report_tests
		game_patch_filter_tests
		lru_cache_tests
		spirv_local_arrays_tests
		vram_budget_tests
		bda_hot_ranges_tests
		shader_vertex_metadata_tests
		resource_materialization_tests
		resource_tracking_tests
		binding_path_tests
		cp_recorder_tests
		event_queue_lifetime_tests
		kernel_file_system_tests
		sync_on_address_tests
		shader_recompiler_compute_tests
		virtual_memory_allocation_tests
		audio_out2_port_tests
		audio_out_timing_tests
		ngs2_sampler_tests
		pad_haptics_tests
		controller_settings_tests
		save_data_memory_tests
		png_enc_tests
		ces_tests
		save_data_dialog_tests
		http_uri_parse_tests
	)
endif()

# Test Darwin signal-context restoration for SSE4a instructions.
if(APPLE AND (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR
		(NOT CMAKE_OSX_ARCHITECTURES AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")))
	add_executable(macos_sse4a_tests EXCLUDE_FROM_ALL
		"${KYTY_TESTS_DIR}/macosSse4aTests.cpp"
		"${KYTY_SOURCE_DIR}/loader/x64InstructionEmulator.cpp"
	)
	target_link_libraries(macos_sse4a_tests common fmt::fmt Zydis::Zydis)
	target_include_directories(macos_sse4a_tests PRIVATE ${inc_headers})
	if(BUILD_TESTING)
		add_test(NAME macos_sse4a COMMAND $<TARGET_FILE:macos_sse4a_tests>)
		set_tests_properties(macos_sse4a PROPERTIES TIMEOUT 15)
		add_dependencies(kyty_tests macos_sse4a_tests)
	endif()
endif()

add_executable(kyty_emulator "${KYTY_SOURCE_DIR}/main.cpp" ${kyty_emulator_src})
configure_macos_guest_address_space(kyty_emulator)

option(KYTY_EMULATOR_IPO "Enable release interprocedural optimization for the emulator target" OFF)
if(KYTY_EMULATOR_IPO)
	set_property(TARGET kyty_emulator PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

target_link_libraries(kyty_emulator ${kyty_emulator_link_libraries})
if (WIN32)
	target_link_libraries(kyty_emulator opengl32)
	target_link_libraries(kyty_emulator iphlpapi)
	target_link_libraries(kyty_emulator setupapi)
	target_link_libraries(kyty_emulator ws2_32)
endif()
if (KYTY_CLANG_CL)
	target_link_libraries(kyty_emulator onecore)
endif()
if (CLANG AND NOT KYTY_CLANG_CL)
	target_link_libraries(kyty_emulator pthread)
endif()
# dlopen/dlsym/dladdr for RenderDoc.
target_link_libraries(kyty_emulator ${CMAKE_DL_LIBS})
target_include_directories(kyty_emulator PRIVATE ${inc_headers})

clang_tidy_check(kyty_emulator "" "${check_headers}" "${inc_headers}")

include_what_you_use(kyty_emulator "${inc_headers}")

set(KYTY_EMULATOR_MAP_NAME "kyty_emulator_${KYTY_COMPILER_ID}_${KYTY_LINKER_ID}.map")

set(KYTY_EMULATOR_MAP_LINK_PATH "${CMAKE_CURRENT_BINARY_DIR}/${KYTY_EMULATOR_MAP_NAME}")
set(KYTY_EMULATOR_PDB_LINK_PATH "${CMAKE_CURRENT_BINARY_DIR}/kyty_emulator.pdb")

if(KYTY_CLANG_CL)
	target_link_options(kyty_emulator PRIVATE
		"/DYNAMICBASE:NO"
		"/DEBUG:FULL"
		"/PDB:${KYTY_EMULATOR_PDB_LINK_PATH}"
		"/lldmap:${KYTY_EMULATOR_MAP_LINK_PATH}"
	)
	add_custom_command(TARGET kyty_emulator POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${KYTY_THIRD_PARTY_DIR}/winpthread/bin/libwinpthread-1.dll" $<TARGET_FILE_DIR:kyty_emulator>/libwinpthread-1.dll)
elseif(WIN32 OR LINUX)
	set_target_properties(kyty_emulator PROPERTIES LINK_FLAGS "${KYTY_LD_OPTIONS}")
	# LINK_FLAGS is a space-delimited string: a -Map path containing spaces would
	# be split into bogus arguments, so pass it as a discrete link option instead.
	target_link_options(kyty_emulator PRIVATE "-Wl,-Map=${KYTY_EMULATOR_MAP_LINK_PATH}")
endif()

if(LINUX)
    target_link_options(kyty_emulator PRIVATE
        $<$<CONFIG:Release>:-Wl,--strip-all>
    )
endif()

if(APPLE)
	# The emulator writes x86-64 trampolines/PLT stubs into memory and executes them, and
	# runs guest code under Rosetta. That requires the JIT / unsigned-executable-memory
	# entitlements, so re-sign the binary after every link (an unsigned/relinked binary
	# reverts to the hardened defaults and aborts when it first executes written code).
	add_custom_command(TARGET kyty_emulator POST_BUILD
		COMMAND codesign -s - --force --options runtime
			--entitlements "${KYTY_SOURCE_DIR}/macos_jit.entitlements"
			$<TARGET_FILE:kyty_emulator>
		COMMENT "Codesign kyty_emulator with JIT entitlements (macOS)")
endif()

install(TARGETS kyty_emulator DESTINATION .)
if(KYTY_BUILD_LAUNCHER)
	add_subdirectory("${KYTY_SOURCE_DIR}/launcher" "${CMAKE_CURRENT_BINARY_DIR}/launcher")

	if(LINUX)
		target_link_options(launcher PRIVATE $<$<CONFIG:Release>:-Wl,--strip-all>)
	endif()
endif()
if(KYTY_CLANG_CL)
	install(FILES "${KYTY_THIRD_PARTY_DIR}/winpthread/bin/libwinpthread-1.dll" DESTINATION .)
endif()
