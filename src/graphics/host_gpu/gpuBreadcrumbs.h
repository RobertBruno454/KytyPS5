#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUBREADCRUMBS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUBREADCRUMBS_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

// Diagnostics: KYTY_GPU_BREADCRUMBS=1 (VK_AMD_buffer_marker). Each dispatch and draw writes its
// sequence number to host memory at the top of the pipe before it runs and at the bottom of the
// pipe after it finished. After a device loss Dump() prints the commands between the last
// finished and the last started one: the guest shaders the GPU was running, in the
// KYTY_SKIP_SHADER_HASHES format. AMD drivers have no NVIDIA-style checkpoints. Off by default.
namespace Libs::Graphics {

struct GraphicContext;

namespace GpuBreadcrumbs {

enum Kind : uint32_t { Dispatch = 1, DispatchIndirect = 2, Draw = 3, DrawIndirect = 4, Mesh = 5 };

// KYTY_GPU_BREADCRUMBS is set: the device should enable VK_AMD_buffer_marker.
[[nodiscard]] bool Requested();
void               Init(GraphicContext& graphics);
// Returns the command's sequence number, 0 when breadcrumbs are off.
[[nodiscard]] uint32_t Begin(vk::CommandBuffer buffer, uint32_t kind, uint64_t hash0,
                             uint64_t hash1);
void                   End(vk::CommandBuffer buffer, uint32_t sequence);
// Prints the in-flight commands once; safe to call from any thread after a device loss.
void Dump();

} // namespace GpuBreadcrumbs
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_GPUBREADCRUMBS_H_
