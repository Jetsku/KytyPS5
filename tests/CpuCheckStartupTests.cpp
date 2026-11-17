// The start-up CPU check runs before every C++ static initializer: linked with kyty_cpu_check
// compiled for x86-64-v3 (KYTY_CPU_CHECK_LEVEL=3), this program prints a line from a static
// initializer and from main. With KYTY_TEST_CPU_CHECK=fail it must end (exit code 3, the message
// on standard error) before either line (CTest checks the output: cpu_check_startup_*).
#include <cstdio>

namespace {

struct Announce {
	Announce() {
		std::printf("static initializer ran\n");
		std::fflush(stdout);
	}
};

const Announce g_announce;

} // namespace

int main() {
	std::printf("main ran\n");
	return 0;
}
