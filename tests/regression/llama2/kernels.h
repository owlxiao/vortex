#ifndef __KERNELS_H__
#define __KERNELS_H__

#include <cstdint>

typedef struct {
  uint64_t xout_addr;
  uint64_t x_addr;
  uint64_t w_addr;
  uint32_t n;
  uint32_t d;
} matmul_arg_t;

#endif // __KERNELS_H__