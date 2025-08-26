#include "common.h"

#include <cstdlib>
#include <vortex.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <unistd.h>
#include <vector>

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 == _ret)                                           \
      break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    cleanup();                                               \
    exit(-1);                                                \
  } while (false)

static const char *kernel_file = "kernel.vxbin";
static vx_device_h device = nullptr;

void cleanup() {
  if (device)
    vx_dev_close(device);
}

///////////////////////////////////////////////////////////////////////////////

static uint32_t size = 256;

static void show_usage() {
  std::cout << "Vortex Rope Test." << std::endl;
  std::cout << "Usage: rope_test [options]" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -h            Show this help message." << std::endl;
  std::cout << "  -k <file>     Specify the kernel file to use. Default is 'kernel.vxbin'." << std::endl;
  std::cout << "  -n <size>     Specify the size of the input vectors. Default is 256." << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "hk:n:")) != -1) {
    switch (c) {
    case 'h':
      show_usage();
      exit(0);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'n':
      size = atoi(optarg);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
void softmax(float *x, int size) {
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

void multihead_attention_ref(float *sxb, float *sq,
                             float *satt, float *key_cache, float *value_cache,
                             int n_heads, int seq_len, int head_size, int kv_dim,
                             int kv_mul, int pos, int loff) {
  for (int h = 0; h < n_heads; h++) {
    // get the query vector for this head
    float *q = sq + h * head_size;
    // attention scores for this head
    float *att = satt + h * seq_len;
    // iterate over all timesteps, including the current one
    for (int t = 0; t <= pos; t++) {
      // get the key vector for this head and at this timestep
      float *k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
      // calculate the attention score as the dot product of q and k
      float score = 0.0f;
      for (int i = 0; i < head_size; i++) {
        score += q[i] * k[i];
      }
      score /= sqrtf(head_size);
      // save the score to the attention buffer
      att[t] = score;
    }

    // softmax the scores to get attention weights, from 0..pos inclusively
    softmax(att, pos + 1);

    // weighted sum of the values, store back into xb
    float *xb = sxb + h * head_size;
    memset(xb, 0, head_size * sizeof(float));
    for (int t = 0; t <= pos; t++) {
      // get the value vector for this head and at this timestep
      float *v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
      // get the attention weight for this timestep
      float a = att[t];
      // accumulate the weighted value into xb
      for (int i = 0; i < head_size; i++) {
        xb[i] += a * v[i];
      }
    }
  }
}

void multihead_attention_vx(float *sxb, float *sq,
                            float *satt, float *key_cache, float *value_cache,
                            int n_heads, int seq_len, int head_size, int kv_dim,
                            int kv_mul, int pos, int loff) {
  vx_buffer_h sxb_buf = NULL;
  vx_buffer_h sq_buf = NULL;
  vx_buffer_h satt_buf = NULL;
  vx_buffer_h key_cache_buf = NULL;
  vx_buffer_h value_cache_buf = NULL;
  vx_buffer_h vx_attention_knl_buf = NULL;
  attention_arg_t args = {};

  // Allocate buffers
  // sxb (dim,) size: n_heads * head_size * sizeof(float)
  size_t sxb_size = n_heads * head_size * sizeof(float);
  RT_CHECK(vx_mem_alloc(device, sxb_size, VX_MEM_READ_WRITE, &sxb_buf));
  RT_CHECK(vx_mem_address(sxb_buf, &args.sxb_addr));

  // sq (dim,) size: n_heads * head_size * sizeof(float)
  size_t sq_size = n_heads * head_size * sizeof(float);
  RT_CHECK(vx_mem_alloc(device, sq_size, VX_MEM_READ, &sq_buf));
  RT_CHECK(vx_mem_address(sq_buf, &args.sq_addr));

  // satt (n_heads, seq_len) size: n_heads * seq_len * sizeof(float)
  size_t satt_size = n_heads * seq_len * sizeof(float);
  RT_CHECK(vx_mem_alloc(device, satt_size, VX_MEM_READ_WRITE, &satt_buf));
  RT_CHECK(vx_mem_address(satt_buf, &args.satt_addr));

  // key_cache (n_layers, seq_len, kv_dim) size: n_layers * seq_len * kv_dim *
  // sizeof(float)
  size_t key_cache_size = n_heads * seq_len * kv_dim * sizeof(float);
  RT_CHECK(vx_mem_alloc(device, key_cache_size, VX_MEM_READ, &key_cache_buf));
  RT_CHECK(vx_mem_address(key_cache_buf, &args.key_cache_addr));

  // value_cache (n_layers, seq_len, kv_dim) size: n_layers * seq_len * kv_dim *
  // sizeof(float)
  size_t value_cache_size = n_heads * seq_len * kv_dim * sizeof(float);
  RT_CHECK(
      vx_mem_alloc(device, value_cache_size, VX_MEM_READ, &value_cache_buf));
  RT_CHECK(vx_mem_address(value_cache_buf, &args.value_cache_addr));

  // Upload sxb to device
  RT_CHECK(vx_copy_to_dev(sxb_buf, sxb, 0, sxb_size));
  // Upload sq to device
  RT_CHECK(vx_copy_to_dev(sq_buf, sq, 0, sq_size));
  // Upload satt to device
  RT_CHECK(vx_copy_to_dev(satt_buf, satt, 0, satt_size));
  // Upload key_cache to device
  RT_CHECK(vx_copy_to_dev(key_cache_buf, key_cache, 0, key_cache_size));
  // Upload value_cache to device
  RT_CHECK(vx_copy_to_dev(value_cache_buf, value_cache, 0, value_cache_size));

  // Upload kernel arguments
  args.n_heads = n_heads;
  args.seq_len = seq_len;
  args.head_size = head_size;
  args.kv_dim = kv_dim;
  args.kv_mul = kv_mul;
  args.pos = pos;
  args.loff = loff;

  vx_buffer_h attention_args_buffer;
  RT_CHECK(vx_upload_bytes(device, &args, sizeof(attention_arg_t),
                           &attention_args_buffer));

  // Upload kernel
  RT_CHECK(vx_upload_kernel_file(device, kernel_file,
                                 &vx_attention_knl_buf));
  // Start the kernel
  RT_CHECK(vx_start(device, vx_attention_knl_buf, attention_args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // Download the output
  RT_CHECK(vx_copy_from_dev(sxb, sxb_buf, 0, sxb_size));

  // Free the buffers
  RT_CHECK(vx_mem_free(sxb_buf));
  RT_CHECK(vx_mem_free(sq_buf));
  RT_CHECK(vx_mem_free(satt_buf));
  RT_CHECK(vx_mem_free(key_cache_buf));
  RT_CHECK(vx_mem_free(value_cache_buf));
  RT_CHECK(vx_mem_free(attention_args_buffer));
  RT_CHECK(vx_mem_free(vx_attention_knl_buf));
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  // open device connection
  std::cout << "Open device..." << std::endl;
  RT_CHECK(vx_dev_open(&device));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  // Attention parameters
  int n_heads = 4;
  int head_size = std::min(size / n_heads, 64u);
  int seq_len = 8;
  int kv_dim = head_size * n_heads;
  int kv_mul = 2;
  int pos = seq_len / 2;
  int loff = 1;

  std::vector<float> h_sxb(n_heads * head_size);
  std::vector<float> h_sq(n_heads * head_size);
  std::vector<float> h_satt(n_heads * seq_len);
  std::vector<float> h_key_cache(n_heads * seq_len * kv_dim);
  std::vector<float> h_value_cache(n_heads * seq_len * kv_dim);

  std::generate(h_sxb.begin(), h_sxb.end(), [&]() { return dis(gen); });
  std::generate(h_sq.begin(), h_sq.end(), [&]() { return dis(gen); });
  std::generate(h_satt.begin(), h_satt.end(), [&]() { return dis(gen); });
  std::generate(h_key_cache.begin(), h_key_cache.end(), [&]() { return dis(gen); });
  std::generate(h_value_cache.begin(), h_value_cache.end(), [&]() { return dis(gen); });

  std::vector<float> ref_sxb = h_sxb;
  std::vector<float> ref_sq = h_sq;
  std::vector<float> ref_satt = h_satt;
  std::vector<float> ref_key_cache = h_key_cache;
  std::vector<float> ref_value_cache = h_value_cache;

  // Run attention
  multihead_attention_vx(h_sxb.data(), h_sq.data(), h_satt.data(),
                         h_key_cache.data(), h_value_cache.data(),
                         n_heads, seq_len, head_size, kv_dim, kv_mul, pos, loff);

  multihead_attention_ref(ref_sxb.data(), ref_sq.data(), ref_satt.data(),
                          ref_key_cache.data(), ref_value_cache.data(),
                          n_heads, seq_len, head_size, kv_dim, kv_mul, pos, loff);

  // Verify the results
  std::cout << "Verifying results..." << std::endl;
  int errors = 0;

  for (size_t i = 0; i < h_sxb.size(); i++) {
    if (std::abs(h_sxb[i] - ref_sxb[i]) > 1e-3) {
      if (errors < 10) {
        std::cout << "Mismatch at index " << i << ": got " << h_sxb[i]
                  << ", expected " << ref_sxb[i] << std::endl;
      }
      errors++;
    }
  }
  if (errors == 0) {
    std::cout << "Results match!" << std::endl;
  } else {
    std::cout << "Total mismatches: " << errors << std::endl;
    cleanup();
    return -1;
  }

  cleanup();
  return 0;
}