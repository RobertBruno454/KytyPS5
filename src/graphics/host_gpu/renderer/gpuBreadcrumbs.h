#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUBREADCRUMBS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUBREADCRUMBS_H_

#include <cstdint>

// Diagnostics: KYTY_GPU_BREADCRUMBS=1 (AMD, VK_AMD_buffer_marker). Each dispatch and draw writes
// its sequence number to host memory at the top of the pipe before it runs and at the bottom of
// the pipe after it finished. After a device loss DumpDeviceLossDiagnostics prints the commands
// between the last finished and the last started one: the guest shaders the GPU was running.
// Implemented in diagnosticCheckpoints.cpp. Off by default.
namespace Libs::Graphics {

class CommandRecorder;

namespace GpuBreadcrumbs {

enum Kind : uint32_t { Dispatch = 1, DispatchIndirect = 2, Draw = 3, DrawIndirect = 4, Mesh = 5 };

// Returns the command's sequence number, 0 when breadcrumbs are off.
[[nodiscard]] uint32_t Begin(const CommandRecorder& recorder, uint32_t kind, uint64_t hash0,
                             uint64_t hash1);
void                   End(const CommandRecorder& recorder, uint32_t sequence);

} // namespace GpuBreadcrumbs
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUBREADCRUMBS_H_
