# cpu_check_startup_fail: cpu_check_startup_tests with KYTY_TEST_CPU_CHECK=fail must exit with code 3
# and the CPU message on standard error before any static initializer or main has printed.
execute_process(
	COMMAND "${CMAKE_COMMAND}" -E env KYTY_TEST_CPU_CHECK=fail "${PROGRAM}"
	RESULT_VARIABLE result
	OUTPUT_VARIABLE out
	ERROR_VARIABLE err
	TIMEOUT 20)
message(STATUS "exit: ${result}\nstdout: ${out}\nstderr: ${err}")
if(NOT result EQUAL 3)
	message(FATAL_ERROR "expected exit code 3, got ${result}")
endif()
if(out MATCHES "static initializer ran" OR out MATCHES "main ran")
	message(FATAL_ERROR "program code ran before the CPU check ended the process")
endif()
if(NOT err MATCHES "This build needs a CPU with AVX2, BMI2 and FMA" OR NOT err MATCHES "KYTY_TEST_CPU_CHECK=fail")
	message(FATAL_ERROR "the CPU message is not on standard error")
endif()
