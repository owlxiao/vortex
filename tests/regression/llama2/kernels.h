#ifndef __KERNELS_H__
#define __KERNELS_H__

#include <cstdint>

typedef struct {
  uint64_t o_addr;
  uint64_t x_addr;
  uint64_t weight_addr;
  uint64_t size;
} rmsnorm_arg_t;

#endif // __KERNELS_H__