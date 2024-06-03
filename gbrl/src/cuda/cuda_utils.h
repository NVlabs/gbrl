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



