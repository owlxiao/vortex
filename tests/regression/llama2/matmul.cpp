#include "kernels.h"

#include <vx_intrinsics.h>
#include <vx_print.h>

void kernel(matmul_arg_t *arg) {
  auto xout = reinterpret_cast<float *>(arg->xout_addr);
  auto x = reinterpret_cast<float *>(arg->x_addr);
  auto w = reinterpret_cast<float *>(arg->w_addr);
  auto n = arg->n;
  auto d = arg->d;

  for (int i = 0; i < d; ++i) {
    float val(0.0f);

    for (int j = 0; j < n; ++j)
      val += w[i * n + j] * x[j];

    xout[i] = val;
  }
}

int main() {
  auto *arg = (matmul_arg_t *)csr_read(VX_CSR_MSCRATCH);
  kernel(arg);

  return 0;
}