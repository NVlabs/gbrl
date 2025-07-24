//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//////////////////////////////////////////////////////////////////////////////
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_utils.h"
#include "cuda_types.h"



void get_grid_dimensions(int n_elements, int& blocks, int& threads_per_block) {
    const int max_threads_per_block = THREADS_PER_BLOCK; // You can adjust this based on your GPU

    if (n_elements < WARP_SIZE) {
        threads_per_block = WARP_SIZE;
        blocks = 1;
        return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    blocks = n_elements / max_threads_per_block + 1;

    threads_per_block = 1 << static_cast<int>(ceilf(log2f(static_cast<float>(n_elements) / blocks)));
    threads_per_block = (threads_per_block > deviceProp.maxThreadsPerBlock) ? deviceProp.maxThreadsPerBlock : threads_per_block;
}