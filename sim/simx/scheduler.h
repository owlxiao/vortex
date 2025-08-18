#pragma once

#include "emulator.h"

#include <bitset>

namespace vortex {

class WarpScheduler {
public:
  virtual ~WarpScheduler() = default;
  virtual int schedule(const WarpMask &active_warps,
                       const WarpMask &stalled_warps,
                       uint32_t num_warps) = 0;
  virtual void reset() {}
};

class FCFSScheduler : public WarpScheduler {
public:
  void reset() override {}

  int schedule(const WarpMask &active_warps,
               const WarpMask &stalled_warps,
               uint32_t num_warps) override {
    // First warp in the active set
    for (size_t wid = 0, nw = num_warps; wid < nw; ++wid) {
      bool warp_active = active_warps.test(wid);
      bool warp_stalled = stalled_warps.test(wid);
      if (warp_active && !warp_stalled) {
        return wid;
      }
    }
    return -1; // No active warps
  }
};

} // namespace vortex