#include "graphics/host_gpu/gpuBreadcrumbs.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace Libs::Graphics::GpuBreadcrumbs {

namespace {

struct Breadcrumb {
	uint32_t sequence = 0;
	uint32_t kind     = 0;
	uint64_t hash0    = 0;
	uint64_t hash1    = 0;
};

constexpr size_t RING_SIZE = 1u << 16u;

std::array<Breadcrumb, RING_SIZE> g_ring {};
std::atomic<uint32_t>             g_sequence {0};
std::atomic<bool>                 g_dumped {false};
vk::Buffer                        g_buffer     = nullptr;
VmaAllocation                     g_allocation = nullptr;
// [0]: the last sequence that reached the top of the pipe, [1]: the last that left its bottom.
volatile uint32_t*                g_markers = nullptr;

const char* KindName(uint32_t kind) {
	switch (kind) {
		case Dispatch: return "dispatch";
		case DispatchIndirect: return "dispatch-indirect";
		case Draw: return "draw";
		case DrawIndirect: return "draw-indirect";
		case Mesh: return "mesh";
		default: return "unknown";
	}
}

template <typename... Args>
void Line(const char* format, Args... args) {
	std::printf(format, args...);
	LOGF(format, args...);
}

} // namespace

bool Requested() {
	static const bool requested = [] {
		const char* value = std::getenv("KYTY_GPU_BREADCRUMBS");
		return value != nullptr && value[0] != '\0' && value[0] != '0';
	}();
	return requested;
}

void Init(GraphicContext& graphics) {
	if (!graphics.buffer_marker_enabled || graphics.allocator == nullptr) {
		return;
	}
	vk::BufferCreateInfo buffer {};
	buffer.size  = 64;
	buffer.usage = vk::BufferUsageFlagBits::eTransferDst;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	allocate.flags =
	    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	allocate.requiredFlags =
	    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	VmaAllocationInfo info {};
	VkBuffer          handle = VK_NULL_HANDLE;
	const auto        raw    = static_cast<VkBufferCreateInfo>(buffer);
	if (vmaCreateBuffer(graphics.allocator, &raw, &allocate, &handle, &g_allocation, &info) !=
	        VK_SUCCESS ||
	    info.pMappedData == nullptr) {
		LOGF("GPU breadcrumbs: could not allocate the marker buffer; breadcrumbs off\n");
		graphics.buffer_marker_enabled = false;
		return;
	}
	g_buffer  = handle;
	g_markers = static_cast<volatile uint32_t*>(info.pMappedData);
	for (uint32_t i = 0; i < 16; i++) {
		g_markers[i] = 0;
	}
	Log::WriteToConsoleAndLog(
	    "GPU breadcrumbs: on (VK_AMD_buffer_marker); a device loss lists the shaders in flight\n");
}

uint32_t Begin(vk::CommandBuffer buffer, uint32_t kind, uint64_t hash0, uint64_t hash1) {
	if (g_markers == nullptr || !buffer) {
		return 0;
	}
	auto sequence = g_sequence.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (sequence == 0) {
		sequence = g_sequence.fetch_add(1, std::memory_order_relaxed) + 1u;
	}
	auto& crumb    = g_ring[sequence % RING_SIZE];
	crumb.kind     = kind;
	crumb.hash0    = hash0;
	crumb.hash1    = hash1;
	crumb.sequence = sequence;
	buffer.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eTopOfPipe, g_buffer, 0, sequence);
	return sequence;
}

void End(vk::CommandBuffer buffer, uint32_t sequence) {
	if (sequence == 0 || g_markers == nullptr || !buffer) {
		return;
	}
	buffer.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eBottomOfPipe, g_buffer,
	                            sizeof(uint32_t), sequence);
}

void Dump() {
	if (g_markers == nullptr || g_dumped.exchange(true)) {
		return;
	}
	const uint32_t started  = g_markers[0];
	const uint32_t finished = g_markers[1];
	const uint32_t recorded = g_sequence.load(std::memory_order_relaxed);
	Line("--- GPU breadcrumbs: last started seq=%u, last finished seq=%u, last recorded seq=%u ---\n",
	     started, finished, recorded);
	// Everything after the last finished command up to the last started one was on the GPU.
	const uint32_t first = finished + 1u;
	const uint32_t last  = std::max(started, first);
	const uint32_t count = std::min<uint32_t>(last - first + 1u, 48u);
	std::string    skip_list;
	for (uint32_t i = 0; i < count; i++) {
		const auto  sequence = first + i;
		const auto& crumb    = g_ring[sequence % RING_SIZE];
		if (crumb.sequence != sequence) {
			continue;
		}
		const bool running = sequence <= started;
		Line("  [%s] seq=%u %s shader=0x%016" PRIx64 " shader2=0x%016" PRIx64 "\n",
		     running ? "running" : "queued", sequence, KindName(crumb.kind), crumb.hash0,
		     crumb.hash1);
		if (!running) {
			continue;
		}
		for (const auto hash: {crumb.hash0, crumb.hash1}) {
			if (hash == 0) {
				continue;
			}
			char text[24];
			std::snprintf(text, sizeof(text), "0x%016" PRIx64, hash);
			if (skip_list.find(text) == std::string::npos) {
				skip_list += skip_list.empty() ? "" : ",";
				skip_list += text;
			}
		}
	}
	if (!skip_list.empty()) {
		Line("  shaders on the GPU at the device loss: %s\n", skip_list.c_str());
	}
	std::fflush(stdout);
}

} // namespace Libs::Graphics::GpuBreadcrumbs

namespace Libs::Graphics::ShaderSkip {

bool Requested(uint64_t shader_hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> result;
		const char*           value = std::getenv("KYTY_SKIP_SHADER_HASHES");
		if (value == nullptr) {
			return result;
		}
		const std::string_view list(value);
		size_t                 start = 0;
		while (start < list.size()) {
			const auto end = std::min(list.find(',', start), list.size());
			if (end > start) {
				const auto hash =
				    std::strtoull(std::string(list.substr(start, end - start)).c_str(), nullptr, 16);
				if (hash != 0) {
					result.push_back(hash);
					char text[96];
					std::snprintf(text, sizeof(text),
					              "ShaderSkip: skipping the draws and dispatches of shader "
					              "0x%016" PRIx64 "\n",
					              hash);
					Log::WriteToConsoleAndLog(text);
				}
			}
			start = end + 1;
		}
		return result;
	}();
	return !hashes.empty() && std::find(hashes.begin(), hashes.end(), shader_hash) != hashes.end();
}

} // namespace Libs::Graphics::ShaderSkip
