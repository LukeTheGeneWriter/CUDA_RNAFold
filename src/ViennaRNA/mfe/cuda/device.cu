/*
 *  Device discovery for the CUDA MFE backend.
 *
 *  The first .cu translation unit in the tree. It exists as much to exercise
 *  the nvcc build rule as to answer the question: everything downstream of here
 *  depends on nvcc objects linking correctly into a libtool convenience library
 *  alongside ordinary C, and that is worth proving before eight thousand lines
 *  of kernels arrive.
 */

#include <cuda_runtime.h>

extern "C" unsigned int
vrna_cuda_device_count(void)
{
  int         n     = 0;
  cudaError_t err   = cudaGetDeviceCount(&n);

  /*
   * No device, no driver, or a driver/runtime mismatch are all ordinary
   * conditions on a machine that simply has no GPU. They mean "decline", not
   * "fail" -- so the error is swallowed deliberately rather than reported.
   */
  if (err != cudaSuccess) {
    cudaGetLastError();     /* clear the sticky error for any later caller */
    return 0u;
  }

  return (n > 0) ? (unsigned int)n : 0u;
}
