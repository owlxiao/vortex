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

void swiglu_ref(float *hb, float *hb2, int hidden_dim) {
  for (int i = 0; i < hidden_dim; i++) {
    float val = hb[i];
    // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    val *= (1.0f / (1.0f + expf(-val)));
    // elementwise multiply with w3(x)
    val *= hb2[i];
    hb[i] = val;
  }
}

void swiglu_vx(float *hb, float *hb2, int hidden_dim) {
  vx_buffer_h vx_swiglu_knl_buf = nullptr;
  vx_buffer_h hb_buf = nullptr;
  vx_buffer_h hb2_buf = nullptr;
  swiglu_arg_t args = {};

  // Allocate buffers
  // hb (hidden_dim,) size: hidden_dim * sizeof(float)
  size_t hb_size = hidden_dim * sizeof(float);
  RT_CHECK(vx_mem_alloc(device, hb_size, VX_MEM_READ_WRITE, &hb_buf));
  RT_CHECK(vx_mem_address(hb_buf, &args.hb_addr));

  // hb2 (hidden_dim,) size: hidden_dim * sizeof(float)
  size_t hb2_size = hidden_dim * sizeof(float);
  RT_CHECK(vx_mem_alloc(device, hb2_size, VX_MEM_READ_WRITE, &hb2_buf));
  RT_CHECK(vx_mem_address(hb2_buf, &args.hb2_addr));

  // Upload hb to device
  RT_CHECK(vx_copy_to_dev(hb_buf, hb, 0, hb_size));
  // Upload hb2 to device
  RT_CHECK(vx_copy_to_dev(hb2_buf, hb2, 0, hb2_size));

  // Upload kernel arguments
  args.hidden_dim = hidden_dim;

  vx_buffer_h swiglu_args_buffer;
  RT_CHECK(vx_upload_bytes(device, &args, sizeof(swiglu_arg_t),
                           &swiglu_args_buffer));

  // Upload kernel
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &vx_swiglu_knl_buf));

  // Start the kernel
  RT_CHECK(vx_start(device, vx_swiglu_knl_buf, swiglu_args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // Download the output
  RT_CHECK(vx_copy_from_dev(hb, hb_buf, 0, hb_size));

  // Free the buffers
  RT_CHECK(vx_mem_free(hb_buf));
  RT_CHECK(vx_mem_free(hb2_buf));
  RT_CHECK(vx_mem_free(swiglu_args_buffer));
  RT_CHECK(vx_mem_free(vx_swiglu_knl_buf));
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  // open device connection
  std::cout << "Open device..." << std::endl;
  RT_CHECK(vx_dev_open(&device));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  // SwiGLU parameters
  int hidden_dim = size;
  std::cout << "SwiGLU hidden_dim: " << hidden_dim << std::endl;

  std::vector<float> h_hb(hidden_dim);
  std::vector<float> h_hb2(hidden_dim);

  std::generate(h_hb.begin(), h_hb.end(), [&]() { return dis(gen); });
  std::generate(h_hb2.begin(), h_hb2.end(), [&]() { return dis(gen); });

  std::vector<float> ref_hb = h_hb;
  std::vector<float> ref_hb2 = h_hb2;

  // Run SwiGLU
  std::cout << "Running SwiGLU..." << std::endl;
  swiglu_vx(h_hb.data(), h_hb2.data(), hidden_dim);
  swiglu_ref(ref_hb.data(), ref_hb2.data(), hidden_dim);

  // Verify results
  int errors = 0;

  for (int i = 0; i < hidden_dim; i++) {
    if (std::fabs(h_hb[i] - ref_hb[i]) > 1e-3) {
      if (errors < 10) {
        std::cout << "Mismatch at index " << i << ": got " << h_hb[i]
                  << ", expected " << ref_hb[i] << std::endl;
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