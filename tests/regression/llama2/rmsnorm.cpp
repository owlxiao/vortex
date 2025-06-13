#include "kernels.h"

#include <vx_intrinsics.h>
#include <vx_print.h>

#include <cmath>

void rmsnorm(float *o, float *x, float *weight, int size) {
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x[j] * x[j];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);

  // normalize and scale
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x[j]);
  }
}

int main() {
  auto *__UNIFORM__ arg = (rmsnorm_arg_t *)csr_read(VX_CSR_MSCRATCH);

  rmsnorm((float *)arg->o_addr, (float *)arg->x_addr, (float *)arg->weight_addr, arg->size);

  return 0;
}