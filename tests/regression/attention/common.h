#pragma once

#include <cstdint>

typedef struct {
  uint64_t sxb_addr;
  uint64_t sq_addr;
  uint64_t satt_addr;
  uint64_t key_cache_addr;
  uint64_t value_cache_addr;

  uint32_t n_heads;
  uint32_t seq_len;
  uint32_t head_size;
  uint32_t kv_dim;
  uint32_t kv_mul;
  uint32_t pos;
  uint32_t loff;
} attention_arg_t;