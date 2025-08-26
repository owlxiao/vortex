#include "common.h"

#include <cstdlib>
#include <vortex.h>

#include <algorithm>
#include <cmath>
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

void rope_ref(int dim, int kv_dim, int head_size, float pos, float *q,
              float *k) {
  for (int i = 0; i < dim; i += 2) {
    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
    for (int v = 0; v < rotn; v++) {
      float *vec = v == 0 ? q : k; // the vector to rotate (query or key)
      float v0 = vec[i];
      float v1 = vec[i + 1];
      vec[i] = v0 * fcr - v1 * fci;
      vec[i + 1] = v0 * fci + v1 * fcr;
    }
  }
}

void rope_vx(int dim, int kv_dim, int head_size, float pos, float *q,
             float *k) {
  vx_buffer_h knl_buf = NULL;
  vx_buffer_h q_buf = NULL;
  vx_buffer_h k_buf = NULL;
  rope_arg_t args = {};

  // Allocate buffers
  // q (dim,) size: dim * sizeof(float)
  size_t q_size = dim * sizeof(float);
  RT_CHECK(vx_mem_alloc(device, q_size, VX_MEM_READ_WRITE, &q_buf));
  RT_CHECK(vx_mem_address(q_buf, &args.q_addr));

  // k (dim,) size: dim * sizeof(float)
  size_t k_size = dim * sizeof(float);
  RT_CHECK(vx_mem_alloc(device, k_size, VX_MEM_READ_WRITE, &k_buf));
  RT_CHECK(vx_mem_address(k_buf, &args.k_addr));

  // Upload q to device
  RT_CHECK(vx_copy_to_dev(q_buf, q, 0, q_size));
  // Upload k to device
  RT_CHECK(vx_copy_to_dev(k_buf, k, 0, k_size));

  // Upload kernel arguments
  args.dim = dim;
  args.head_size = head_size;
  args.kv_dim = kv_dim;
  args.pos = pos;

  vx_buffer_h rope_args_buffer;
  RT_CHECK(
      vx_upload_bytes(device, &args, sizeof(rope_arg_t), &rope_args_buffer));

  // Upload kernel
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &knl_buf));

  // Start the kernel
  RT_CHECK(vx_start(device, knl_buf, rope_args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // Download the output
  RT_CHECK(vx_copy_from_dev(q, q_buf, 0, q_size));
  RT_CHECK(vx_copy_from_dev(k, k_buf, 0, k_size));

  // Free the buffers
  RT_CHECK(vx_mem_free(q_buf));
  RT_CHECK(vx_mem_free(k_buf));
  RT_CHECK(vx_mem_free(rope_args_buffer));
  RT_CHECK(vx_mem_free(knl_buf));
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  // open device connection
  std::cout << "Open device..." << std::endl;
  RT_CHECK(vx_dev_open(&device));

  std::vector<float> h_q(size);
  std::vector<float> h_k(size);
  std::vector<float> ref_q(size);
  std::vector<float> ref_k(size);

  // Fill h_q and h_k with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
  std::generate(h_q.begin(), h_q.end(), [&]() { return dis(gen); });
  std::generate(h_k.begin(), h_k.end(), [&]() { return dis(gen); });

  // Make copies for reference
  std::copy(h_q.begin(), h_q.end(), ref_q.begin());
  std::copy(h_k.begin(), h_k.end(), ref_k.begin());

  // Run RoPE
  int dim = size;
  int kv_dim = size;
  int head_size = std::min(size, 64u);
  float pos = 5.0f;

  std::cout << "Running RoPE with parameters: " << std::endl;
  std::cout << "  dim: " << dim << std::endl;
  std::cout << "  kv_dim: " << kv_dim << std::endl;
  std::cout << "  head_size: " << head_size << std::endl;
  std::cout << "  pos: " << pos << std::endl;

  rope_vx(dim, kv_dim, head_size, pos, h_q.data(), h_k.data());
  rope_ref(dim, kv_dim, head_size, pos, ref_q.data(), ref_k.data());

  // Verify the results
  std::cout << "Verifying results..." << std::endl;
  int errors = 0;

  for (int i = 0; i < (int)size; ++i) {
    float diff_q = std::abs(h_q[i] - ref_q[i]);
    float diff_k = std::abs(h_k[i] - ref_k[i]);

    if (diff_q > 1e-5f || diff_k > 1e-5f) {
      std::cout << "Mismatch at index " << i << ": "
                << "q_vx=" << h_q[i] << " q_ref=" << ref_q[i] << " diff=" << diff_q
                << ", k_vx=" << h_k[i] << " k_ref=" << ref_k[i] << " diff=" << diff_k << std::endl;
      errors++;
    }
  }

  if (errors == 0) {
    std::cout << "Verification PASSED!" << std::endl;
  } else {
    std::cout << "Verification FAILED with " << errors << " errors!" << std::endl;
    cleanup();
    return -1;
  }

  cleanup();
  return 0;
}