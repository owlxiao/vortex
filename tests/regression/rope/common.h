#pragma once

#include <cstdint>

typedef struct {
  uint64_t q_addr;
  uint64_t k_addr;
  uint32_t dim;
  uint32_t head_size;
  uint32_t kv_dim;
  uint32_t pos;
} rope_arg_t;