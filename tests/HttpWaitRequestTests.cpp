#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "libs/errno.h"
#include "libs/network.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {
namespace Http = Libs::Network::Http;
using namespace Libs::Network;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "HttpWaitRequestTests: FAILED: %s\n", message);
		std::exit(1);
	}
}
} // namespace

int main() {
	Common::InitializeThreads();
	Config::Initialize();
	Log::Initialize();
	Initialize();
	const int pool = Net::NetPoolCreate("http-tests", 65536, 0);
	const int ssl = Ssl::SslInit(65536);
	const int ctx = Http::HttpInit(pool, ssl, 65536);
	Check(ctx > 0, "create context");
	Http::HttpEpollHandle epoll = nullptr;
	Check(Http::HttpCreateEpoll(ctx, &epoll) == OK, "create epoll");
	Http::HttpNBEvent events[2] {};
	const auto start = std::chrono::steady_clock::now();
	Check(Http::HttpWaitRequest(epoll, events, 1, 20000) == 0, "empty wait times out");
	Check(std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(20),
	      "empty wait must honor timeout (microseconds)");
	Check(Http::HttpWaitRequest(epoll, events, 0, 0) == HTTP_ERROR_INVALID_VALUE,
	      "reject zero output capacity");
	Check(Http::HttpWaitRequest(epoll, events, 1, -2) == HTTP_ERROR_INVALID_VALUE,
	      "reject invalid timeout");
	Check(Http::HttpWaitRequest(epoll, nullptr, 1, 0) == HTTP_ERROR_INVALID_VALUE,
	      "reject null output");
	const int tmpl = Http::HttpCreateTemplate(ctx, "http-tests", 2, 0);
	const int conn = Http::HttpCreateConnectionWithURL(tmpl, "http://localhost/", 0);
	const int first = Http::HttpCreateRequestWithURL2(conn, "GET", "http://localhost/one", 0);
	const int second = Http::HttpCreateRequestWithURL2(conn, "GET", "http://localhost/two", 0);
	Check(first > 0 && second > 0, "create requests");
	int first_arg = 1, second_arg = 2;
	Check(Http::HttpSetEpoll(first, epoll, &first_arg) == OK, "attach first request");
	Check(Http::HttpSetEpoll(second, epoll, &second_arg) == OK, "attach second request");
	Check(Http::HttpSendRequest(first, nullptr, 0) == HTTP_ERROR_TIMEOUT, "offline send result");
	Check(Http::HttpSendRequest(second, nullptr, 0) == HTTP_ERROR_TIMEOUT, "second send result");
	Check(Http::HttpWaitRequest(epoll, events, 1, 0) == 1, "bounded event delivery");
	Check(events[0].id == first && events[0].user_arg == &first_arg &&
	      (events[0].events & 8) != 0, "first failure event retains request and user data");
	Check(Http::HttpWaitRequest(epoll, events, 2, 0) == 1 && events[0].id == second &&
	      events[0].user_arg == &second_arg, "remaining event survives capacity limit");
	Check(Http::HttpWaitRequest(epoll, events, 2, 0) == 0, "events consumed once");
	Check(Http::HttpSendRequest(first, nullptr, 0) == HTTP_ERROR_TIMEOUT &&
	      Http::HttpSendRequest(first, nullptr, 0) == HTTP_ERROR_TIMEOUT,
	      "repeated send failure");
	Check(Http::HttpWaitRequest(epoll, events, 2, 0) == 1 && events[0].id == first,
	      "same request has bounded pending completion");
	Check(Http::HttpSendRequest(first, nullptr, 0) == HTTP_ERROR_TIMEOUT, "queue before detach");
	Check(Http::HttpUnsetEpoll(first) == OK, "detach request");
	Check(Http::HttpWaitRequest(epoll, events, 2, 0) == 0, "detach clears pending event");
	Check(Http::HttpSendRequest(first, nullptr, 0) == HTTP_ERROR_TIMEOUT, "send detached request");
	Check(Http::HttpWaitRequest(epoll, events, 2, 0) == 0, "detached request does not publish");
	Check(Http::HttpAbortRequest(second) == OK, "abort request");
	Check(Http::HttpWaitRequest(epoll, events, 2, 0) == 1 && events[0].id == second &&
	      (events[0].events & 0x10) != 0, "abort publishes hangup");

	std::atomic<int> result {-999};
	std::thread waiter([&] { result = Http::HttpWaitRequest(epoll, events, 2, 2000000); });
	std::this_thread::sleep_for(std::chrono::milliseconds(30));
	Check(result == -999, "wait blocks before completion");
	Check(Http::HttpSendRequest(second, nullptr, 0) == HTTP_ERROR_TIMEOUT, "wake waiter");
	waiter.join();
	Check(result == 1 && events[0].id == second, "send wakes blocked waiter");
	Check(Http::HttpSendRequest(second, nullptr, 0) == HTTP_ERROR_TIMEOUT, "queue before delete");
	Check(Http::HttpDeleteRequest(second) == OK, "delete pending request");
	Check(Http::HttpWaitRequest(epoll, events, 2, 0) == 0, "deletion removes pending completion");

	result = -999;
	std::thread destroyed_waiter([&] { result = Http::HttpWaitRequest(epoll, events, 2, -1); });
	std::this_thread::sleep_for(std::chrono::milliseconds(30));
	Check(result == -999, "infinite wait blocks");
	Check(Http::HttpDestroyEpoll(ctx, epoll) == OK, "destroy wakes waiter");
	destroyed_waiter.join();
	// If the host did not schedule the waiter before destruction, it correctly
	// sees an invalid handle instead. A waiter already inside must be aborted.
	Check(result == HTTP_ERROR_ABORTED || result == HTTP_ERROR_INVALID_ID,
	      "destroyed wait is released or rejects a stale handle");
	Check(Http::HttpWaitRequest(epoll, events, 2, 0) == HTTP_ERROR_INVALID_ID,
	      "stale handle rejected");
	Check(Http::HttpDeleteRequest(first) == OK, "delete remaining request");
	Check(Http::HttpDeleteConnection(conn) == OK && Http::HttpDeleteTemplate(tmpl) == OK,
	      "delete connection and template");
	Check(Http::HttpCreateEpoll(ctx, &epoll) == OK, "create context termination waiter");
	result = -999;
	std::thread term_waiter([&] { result = Http::HttpWaitRequest(epoll, events, 2, -1); });
	std::this_thread::sleep_for(std::chrono::milliseconds(30));
	Check(result == -999, "context wait blocks");
	Check(Http::HttpTerm(ctx) == OK, "terminate context");
	term_waiter.join();
	Check(result == HTTP_ERROR_ABORTED || result == HTTP_ERROR_INVALID_ID,
	      "context termination releases waiter or rejects a stale handle");
	Check(Http::HttpCreateEpoll(ctx, &epoll) == HTTP_ERROR_INVALID_ID && epoll == nullptr,
	      "cannot create epoll in terminated context");
	Check(Ssl::SslTerm(ssl) == OK && Net::NetPoolDestroy(pool) == OK, "cleanup");
	Shutdown();
	Log::Shutdown();
	Config::Shutdown();
	std::puts("HttpWaitRequestTests: passed");
	return 0;
}
