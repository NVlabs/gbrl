//////////////////////////////////////////////////////////////////////////////
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
//  property and proprietary rights in and to this material, related
//  documentation and any modifications thereto. Any use, reproduction,
//  disclosure or distribution of this material and related documentation
//  without an express license agreement from NVIDIA CORPORATION or
//  its affiliates is strictly prohibited.
//
//////////////////////////////////////////////////////////////////////////////
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif
void get_grid_dimensions(int elements, int& blocks, int& threads_per_block);

#ifdef __cplusplus
}
#endif

#endif 



