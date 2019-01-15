#ifndef CUZFP_DECODE3_CUH
#define CUZFP_DECODE3_CUH

#include "shared.h"
#include "decode.cuh"
#include "type_info.cuh"

namespace cuZFP {

template<typename Scalar> 
__device__ __host__ inline 
void scatter_partial3(const Scalar* q, Scalar* p, int nx, int ny, int nz, int sx, int sy, int sz)
{
  uint x, y, z;
  for (z = 0; z < nz; z++, p += sz - ny * sy, q += 4 * (4 - ny))
    for (y = 0; y < ny; y++, p += sy - nx * sx, q += 4 - nx)
      for (x = 0; x < nx; x++, p += sx, q++)
        *p = *q;
}

template<typename Scalar> 
__device__ __host__ inline 
void scatter3(const Scalar* q, Scalar* p, int sx, int sy, int sz)
{
  uint x, y, z;
  for (z = 0; z < 4; z++, p += sz - 4 * sy)
    for (y = 0; y < 4; y++, p += sy - 4 * sx)
      for (x = 0; x < 4; x++, p += sx)
        *p = *q++;
}


template<class Scalar, int BlockSize>
__global__
void
cudaDecode3(Word *blocks,
            Word *offset_table,
            Scalar *out,
            const uint3 dims,
            const int3 stride,
            const uint3 padded_dims,
            const int param,
            const zfp_mode mode,
            const uint chunk_size)
{
  
  typedef unsigned long long int ull;
  typedef long long int ll;

  const ull blockId = blockIdx.x +
                      blockIdx.y * gridDim.x +
                      gridDim.x * gridDim.y * blockIdx.z;
  const ull thread_idx = blockId * blockDim.x + threadIdx.x;
  const int total_blocks = (padded_dims.x * padded_dims.y * padded_dims.z) / 64;

  ll bit_offset;
  int bmax, block_idx;

  switch(mode) {
    case zfp_mode_fixed_rate:
      block_idx = thread_idx;
      bmax = MIN(thread_idx + 1, total_blocks);
      bit_offset = param * thread_idx;
      break;
    case zfp_mode_fixed_accuracy:
    case zfp_mode_fixed_precision:
      block_idx = thread_idx * chunk_size;
      bmax = MIN(block_idx + chunk_size, total_blocks);
      bit_offset = offset_table[thread_idx];
      break;
  }

  BlockReader<BlockSize> reader(blocks, bit_offset);

  for (; block_idx < bmax; block_idx++)
  {
    Scalar result[BlockSize] = {0};
    zfp_decode<Scalar,BlockSize>(reader, result, param, mode, 3);

    // logical block dims
    uint3 block_dims;
    block_dims.x = padded_dims.x >> 2;
    block_dims.y = padded_dims.y >> 2;
    block_dims.z = padded_dims.z >> 2;
    // logical pos in 3d array
    uint3 block;
    block.x = (block_idx % block_dims.x) * 4;
    block.y = ((block_idx/ block_dims.x) % block_dims.y) * 4;
    block.z = (block_idx/ (block_dims.x * block_dims.y)) * 4;

    // default strides
    const ll offset = (ll)block.x * stride.x + (ll)block.y * stride.y + (ll)block.z * stride.z;
    bool partial = false;
    if(block.x + 4 > dims.x) partial = true;
    if(block.y + 4 > dims.y) partial = true;
    if(block.z + 4 > dims.z) partial = true;
    if(partial)
    {
      const uint nx = block.x + 4u > dims.x ? dims.x - block.x : 4;
      const uint ny = block.y + 4u > dims.y ? dims.y - block.y : 4;
      const uint nz = block.z + 4u > dims.z ? dims.z - block.z : 4;
      scatter_partial3(result, out + offset, nx, ny, nz, stride.x, stride.y, stride.z);
    }
    else
    {
      scatter3(result, out + offset, stride.x, stride.y, stride.z);
    }
  }
}

template<class Scalar>
size_t decode3launch(uint3 dims,
                     int3 stride,
                     Word *stream,
                     Word *offset_table,
                     Scalar *d_data,
                     int param,
                     zfp_mode mode,
                     uint chunk_size)
{
  /* TODO: optimize block size, possibly based on array dimensions and mode */
  int cuda_block_size;
  size_t stream_bytes, total_blocks;

  uint3 zfp_pad(dims); 
  // ensure that we have block sizes
  // that are a multiple of 4
  if(zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if(zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;
  if(zfp_pad.z % 4 != 0) zfp_pad.z += 4 - dims.z % 4;

  const int zfp_blocks = (zfp_pad.x * zfp_pad.y * zfp_pad.z) / 64; 

  switch(mode) {
    case zfp_mode_fixed_rate:
      cuda_block_size = 128;
      stream_bytes = calc_device_mem3d(zfp_pad, param);
      total_blocks = zfp_blocks;
      if(zfp_blocks % cuda_block_size != 0)
        total_blocks += (cuda_block_size - zfp_blocks % cuda_block_size);
      break;
    case zfp_mode_fixed_accuracy:
    case zfp_mode_fixed_precision:
      /* Round array to multiple of 4, chunks to multiple of cuda block size */
      cuda_block_size = 64;
      /* Set stream bytes to the actual size */
      stream_bytes = 1;
      total_blocks = (zfp_blocks + chunk_size - 1) / chunk_size;
      break;
  }
  dim3 block_size = dim3(cuda_block_size, 1, 1);
  dim3 grid_size = calculate_grid_size(total_blocks, cuda_block_size);

#ifdef CUDA_ZFP_RATE_PRINT
  // setup some timing code
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
#endif

  cudaDecode3<Scalar, 64> << < grid_size, block_size >> >
    (stream,
     offset_table,
     d_data,
     dims,
     stride,
     zfp_pad,
     param,
     mode,
     chunk_size);

#ifdef CUDA_ZFP_RATE_PRINT
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaStreamSynchronize(0);

  float miliseconds = 0;
  cudaEventElapsedTime(&miliseconds, start, stop);
  float seconds = miliseconds / 1000.f;
  float rate = (float(dims.x * dims.y * dims.z) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("%.2f",rate);
//  printf("Decode elapsed time: %.5f (s)\n", seconds);
//  printf("# decode3 rate: %.2f (GB / sec) %d\n", rate, param);
#endif

  return stream_bytes;
}

template<class Scalar>
size_t decode3(uint3 dims, 
               int3 stride,
               Word  *stream,
               Word *offset_table,
               Scalar *d_data,
               int param,
               zfp_mode mode,
               uint chunk_size)
{
	return decode3launch<Scalar>(dims, stride, stream, offset_table, d_data, param, mode, chunk_size);
}

} // namespace cuZFP

#endif
