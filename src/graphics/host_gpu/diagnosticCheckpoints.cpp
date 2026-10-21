#include "graphics/host_gpu/vulkanCommon.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/aftermath.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/gpuBreadcrumbs.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr size_t CHECKPOINT_RING_SIZE = 1u << 16u;

std::array<DiagnosticCheckpoint, CHECKPOINT_RING_SIZE> g_checkpoints {};
std::atomic<uint64_t>                                  g_checkpoint_sequence {0};

const char* OpName(uint32_t op) {
	switch (op) {
		case 0: return "DispatchDirect";
		case 1: return "DrawIndex";
		case 2: return "DrawIndexAuto";
		case 3: return "EopWrite";
		case 4: return "EopInterrupt";
		case 5: return "EopWriteBack";
		case 6: return "EopFlip";
		case 7: return "EopWriteBackFlip";
		case 8: return "EopOnlyFlip";
		default: return "Unknown";
	}
}

void Print(const char* stage, const DiagnosticCheckpoint& checkpoint) {
	std::printf("  [%s] seq=%" PRIu64 " op=%s submit=%" PRIu64 " args=%u,%u,%u,%u,0x%016" PRIx64
	            " shaders=0x%016" PRIx64 ",0x%016" PRIx64 "\n",
	            stage, checkpoint.sequence, OpName(checkpoint.op), checkpoint.submit_id,
	            checkpoint.arg0, checkpoint.arg1, checkpoint.arg2, checkpoint.arg3,
	            checkpoint.arg4, checkpoint.shader0, checkpoint.shader1);
	LOGF("  [%s] seq=%" PRIu64 " op=%s submit=%" PRIu64 " args=%u,%u,%u,%u,0x%016" PRIx64
	     " shaders=0x%016" PRIx64 ",0x%016" PRIx64 "\n",
	     stage, checkpoint.sequence, OpName(checkpoint.op), checkpoint.submit_id, checkpoint.arg0,
	     checkpoint.arg1, checkpoint.arg2, checkpoint.arg3, checkpoint.arg4, checkpoint.shader0,
	     checkpoint.shader1);
}

struct Breadcrumb {
	uint32_t sequence = 0;
	uint32_t kind     = 0;
	uint64_t hash0    = 0;
	uint64_t hash1    = 0;
};

constexpr size_t BREADCRUMB_RING_SIZE = 1u << 16u;

std::array<Breadcrumb, BREADCRUMB_RING_SIZE> g_breadcrumbs {};
std::atomic<uint32_t>                        g_breadcrumb_sequence {0};
vk::Buffer                                   g_breadcrumb_buffer = nullptr;
VmaAllocation                                g_breadcrumb_allocation = nullptr;
// [0]: the last sequence that reached the top of the pipe, [1]: the last that left its bottom.
volatile uint32_t*                           g_breadcrumb_markers = nullptr;

const char* BreadcrumbKindName(uint32_t kind) {
	switch (kind) {
		case GpuBreadcrumbs::Dispatch: return "dispatch";
		case GpuBreadcrumbs::DispatchIndirect: return "dispatch-indirect";
		case GpuBreadcrumbs::Draw: return "draw";
		case GpuBreadcrumbs::DrawIndirect: return "draw-indirect";
		case GpuBreadcrumbs::Mesh: return "mesh";
		default: return "unknown";
	}
}

void BreadcrumbLine(const char* format, auto... args) {
	std::printf(format, args...);
	LOGF(format, args...);
}

void DumpGpuBreadcrumbs() {
	if (g_breadcrumb_markers == nullptr) {
		return;
	}
	const uint32_t started  = g_breadcrumb_markers[0];
	const uint32_t finished = g_breadcrumb_markers[1];
	const uint32_t recorded = g_breadcrumb_sequence.load(std::memory_order_relaxed);
	BreadcrumbLine("--- GPU breadcrumbs: last started seq=%u, last finished seq=%u, last recorded "
	               "seq=%u ---\n",
	               started, finished, recorded);
	// Everything after the last finished command up to the last started one was on the GPU.
	const uint32_t first = finished + 1u;
	const uint32_t last  = std::max(started, first);
	const uint32_t count = std::min<uint32_t>(last - first + 1u, 48u);
	std::string    skip_list;
	for (uint32_t i = 0; i < count; i++) {
		const auto  sequence = first + i;
		const auto& crumb    = g_breadcrumbs[sequence % BREADCRUMB_RING_SIZE];
		if (crumb.sequence != sequence) {
			continue;
		}
		const bool running = sequence <= started;
		BreadcrumbLine("  [%s] seq=%u %s shader=0x%016" PRIx64 " shader2=0x%016" PRIx64 "\n",
		               running ? "running" : "queued", sequence, BreadcrumbKindName(crumb.kind),
		               crumb.hash0, crumb.hash1);
		if (running) {
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
	}
	if (!skip_list.empty()) {
		BreadcrumbLine("  shaders on the GPU at the device loss: %s\n", skip_list.c_str());
	}
	std::fflush(stdout);
}

} // namespace

void InitGpuBreadcrumbs(GraphicContext& graphics) {
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
	if (vmaCreateBuffer(graphics.allocator, &raw, &allocate, &handle, &g_breadcrumb_allocation,
	                    &info) != VK_SUCCESS ||
	    info.pMappedData == nullptr) {
		LOGF("GPU breadcrumbs: could not allocate the marker buffer; breadcrumbs off\n");
		graphics.buffer_marker_enabled = false;
		return;
	}
	g_breadcrumb_buffer  = handle;
	g_breadcrumb_markers = static_cast<volatile uint32_t*>(info.pMappedData);
	for (uint32_t i = 0; i < 16; i++) {
		g_breadcrumb_markers[i] = 0;
	}
	LOGF("GPU breadcrumbs: on (VK_AMD_buffer_marker); a device loss lists the shaders in flight\n");
}

uint32_t GpuBreadcrumbs::Begin(const CommandRecorder& recorder, uint32_t kind, uint64_t hash0,
                               uint64_t hash1) {
	if (g_breadcrumb_markers == nullptr) {
		return 0;
	}
	auto sequence = g_breadcrumb_sequence.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (sequence == 0) {
		sequence = g_breadcrumb_sequence.fetch_add(1, std::memory_order_relaxed) + 1u;
	}
	auto& crumb    = g_breadcrumbs[sequence % BREADCRUMB_RING_SIZE];
	crumb.kind     = kind;
	crumb.hash0    = hash0;
	crumb.hash1    = hash1;
	crumb.sequence = sequence;
	recorder.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eTopOfPipe, g_breadcrumb_buffer, 0,
	                              sequence);
	return sequence;
}

void GpuBreadcrumbs::End(const CommandRecorder& recorder, uint32_t sequence) {
	if (sequence == 0 || g_breadcrumb_markers == nullptr) {
		return;
	}
	recorder.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eBottomOfPipe, g_breadcrumb_buffer,
	                              sizeof(uint32_t), sequence);
}

const DiagnosticCheckpoint* RecordDiagnosticCheckpoint(const DiagnosticCheckpoint& checkpoint) {
	const auto sequence = g_checkpoint_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
	auto&      slot     = g_checkpoints[sequence % CHECKPOINT_RING_SIZE];
	slot                = checkpoint;
	slot.sequence       = sequence;
	return &slot;
}

static void DumpDeviceFault(GraphicContext& graphics) {
	if (!graphics.device_fault_enabled) {
		return;
	}
	vk::DeviceFaultCountsEXT counts {};
	auto result = graphics.device.getFaultInfoEXT(&counts, nullptr);
	if (result != vk::Result::eSuccess && result != vk::Result::eIncomplete) {
		std::printf("--- Device fault: query failed: %s ---\n", vk::to_string(result).c_str());
		return;
	}
	std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<vk::DeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
	std::vector<uint8_t>                       binary(static_cast<size_t>(counts.vendorBinarySize));
	vk::DeviceFaultInfoEXT                     info {};
	info.pAddressInfos = addresses.empty() ? nullptr : addresses.data();
	info.pVendorInfos  = vendors.empty() ? nullptr : vendors.data();
	info.pVendorBinaryData = binary.empty() ? nullptr : binary.data();
	result = graphics.device.getFaultInfoEXT(&counts, &info);
	std::printf("--- Device fault (%s): \"%s\" addresses=%u vendor=%u binary=%" PRIu64 " ---\n",
	            vk::to_string(result).c_str(), info.description.data(), counts.addressInfoCount,
	            counts.vendorInfoCount, static_cast<uint64_t>(counts.vendorBinarySize));
	LOGF("--- Device fault (%s): \"%s\" addresses=%u vendor=%u binary=%" PRIu64 " ---\n",
	     vk::to_string(result).c_str(), info.description.data(), counts.addressInfoCount,
	     counts.vendorInfoCount, static_cast<uint64_t>(counts.vendorBinarySize));
	if (!binary.empty()) {
		if (auto* file = std::fopen("_device_fault.nv-gpudmp", "wb"); file != nullptr) {
			std::fwrite(binary.data(), 1, binary.size(), file);
			std::fclose(file);
			std::printf("  vendor binary written to _device_fault.nv-gpudmp\n");
		}
	}
	for (uint32_t i = 0; i < counts.addressInfoCount; i++) {
		const auto& address = addresses[i];
		std::printf("  address[%u]: %s at 0x%016" PRIx64 " (precision 0x%" PRIx64 ")\n", i,
		            vk::to_string(address.addressType).c_str(),
		            static_cast<uint64_t>(address.reportedAddress),
		            static_cast<uint64_t>(address.addressPrecision));
		LOGF("  address[%u]: %s at 0x%016" PRIx64 " (precision 0x%" PRIx64 ")\n", i,
		     vk::to_string(address.addressType).c_str(),
		     static_cast<uint64_t>(address.reportedAddress),
		     static_cast<uint64_t>(address.addressPrecision));
	}
	for (uint32_t i = 0; i < counts.vendorInfoCount; i++) {
		const auto& vendor = vendors[i];
		std::printf("  vendor[%u]: \"%s\" code=0x%016" PRIx64 " data=0x%016" PRIx64 "\n", i,
		            vendor.description.data(), static_cast<uint64_t>(vendor.vendorFaultCode),
		            static_cast<uint64_t>(vendor.vendorFaultData));
		LOGF("  vendor[%u]: \"%s\" code=0x%016" PRIx64 " data=0x%016" PRIx64 "\n", i,
		     vendor.description.data(), static_cast<uint64_t>(vendor.vendorFaultCode),
		     static_cast<uint64_t>(vendor.vendorFaultData));
	}
	std::fflush(stdout);
}

void DumpDeviceLossDiagnostics(GraphicContext& graphics) {
	Aftermath::WaitForCrashDump();
	DumpDeviceFault(graphics);
	DumpGpuBreadcrumbs();
	if (!graphics.diagnostic_checkpoints_enabled || graphics.queue == nullptr) {
		return;
	}
	std::vector<vk::CheckpointDataNV> data;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		data = graphics.queue.getCheckpointDataNV();
	}
	std::printf("--- Diagnostic checkpoints (%zu, last recorded seq=%" PRIu64 ") ---\n",
	            data.size(), g_checkpoint_sequence.load(std::memory_order_relaxed));
	LOGF("--- Diagnostic checkpoints (%zu, last recorded seq=%" PRIu64 ") ---\n", data.size(),
	     g_checkpoint_sequence.load(std::memory_order_relaxed));
	for (const auto& entry: data) {
		const auto* checkpoint = static_cast<const DiagnosticCheckpoint*>(entry.pCheckpointMarker);
		const auto  stage      = vk::to_string(entry.stage);
		const bool  in_ring    = checkpoint >= g_checkpoints.data() &&
		                      checkpoint < g_checkpoints.data() + g_checkpoints.size();
		std::printf("  [%s] marker=%p ring=[%p, %p) %s\n", stage.c_str(),
		            static_cast<const void*>(checkpoint),
		            static_cast<const void*>(g_checkpoints.data()),
		            static_cast<const void*>(g_checkpoints.data() + g_checkpoints.size()),
		            in_ring ? "inside" : "OUTSIDE");
		if (checkpoint == nullptr || !in_ring) {
			continue;
		}
		Print(stage.c_str(), *checkpoint);
	}
	std::fflush(stdout);
}

} // namespace Libs::Graphics
