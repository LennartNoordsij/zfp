#ifndef CUZFP_DECODE2_CUH
#define CUZFP_DECODE2_CUH

#include "shared.h"
#include "decode.cuh"
#include "type_info.cuh"

namespace cuZFP {

template<typename Scalar> 
__device__ __host__ inline 
void scatter_partial2(const Scalar* q, Scalar* p, int nx, int ny, int sx, int sy)
{
  uint x, y;
  for (y = 0; y < ny; y++, p += sy - nx * sx, q += 4 - nx)
    for (x = 0; x < nx; x++, p += sx, q++)
      *p = *q;
}

template<typename Scalar> 
__device__ __host__ inline 
void scatter2(const Scalar* q, Scalar* p, int sx, int sy)
{
  uint x, y;
  for (y = 0; y < 4; y++, p += sy - 4 * sx)
    for (x = 0; x < 4; x++, p += sx)
      *p = *q++;
}

template<class Scalar, int BlockSize>
__global__
void
cudaDecode2(Word *blocks,
            Word *offset_table,
            Scalar *out,
            const uint2 dims,
            const int2 stride,
            const uint2 padded_dims,
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
  const int total_blocks = (padded_dims.x * padded_dims.y) / 16;

  ll bit_offset;
  uint bmax, block_idx;

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

  // logical block dims
  uint2 block_dims;
  block_dims.x = padded_dims.x >> 2;
  block_dims.y = padded_dims.y >> 2;

  BlockReader<BlockSize> reader(blocks, bit_offset);

  for (; block_idx < bmax; block_idx++)
  {
    Scalar result[BlockSize] = {0};
    zfp_decode<Scalar,BlockSize>(reader, result, param, mode, 2);

    // logical pos in 3d array
    uint2 block;
    block.x = (block_idx % block_dims.x) * 4;
    block.y = ((block_idx/ block_dims.x) % block_dims.y) * 4;

    const ll offset = (ll)block.x * stride.x + (ll)block.y * stride.y;

    if(block.x + 4 > dims.x || block.y + 4 > dims.y)
    {
      const uint nx = block.x + 4 > dims.x ? dims.x - block.x : 4;
      const uint ny = block.y + 4 > dims.y ? dims.y - block.y : 4;
      scatter_partial2(result, out + offset, nx, ny, stride.x, stride.y);
    }
    else
    {
      scatter2(result, out + offset, stride.x, stride.y);
    }
  }
}


template<class Scalar>
size_t decode2launch(uint2 dims,
                     int2 stride,
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
  uint2 zfp_pad(dims);
  // ensure that we have block sizes
  // that are a multiple of 4
  if(zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if(zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;

  const int zfp_blocks = (zfp_pad.x * zfp_pad.y) / 16; 

  switch(mode) {
    case zfp_mode_fixed_rate:
      cuda_block_size = 128;
      stream_bytes = calc_device_mem2d(zfp_pad, param);
      total_blocks = zfp_blocks;
      if(zfp_blocks % cuda_block_size != 0)
        total_blocks += (cuda_block_size - zfp_blocks % cuda_block_size);
      break;
    case zfp_mode_fixed_accuracy:
    case zfp_mode_fixed_precision:
      /* Round array to multiple of 4, chunks to multiple of cuda block size */
      cuda_block_size = 64;
      /* TODO: Encode chunk size in zfp header or as input argument
      TODO: Set stream bytes to the actual size */
      stream_bytes = 1;
      total_blocks = (zfp_blocks + chunk_size - 1) / chunk_size;
      if(total_blocks % cuda_block_size != 0)
        total_blocks += (cuda_block_size - total_blocks % cuda_block_size);
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

  cudaDecode2<Scalar, 16> << < grid_size, block_size >> >
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
  float rate = (float(dims.x * dims.y) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("Decode elapsed time: %.5f (s)\n", seconds);
  printf("# decode2 rate: %.2f (GB / sec) %d\n", rate, param);
#endif

  return stream_bytes;
}



template<class Scalar>
size_t decode2(uint2 dims, 
               int2 stride,
               Word *stream,
               Word *offset_table,
               Scalar *d_data,
               int param,
               zfp_mode mode,
               uint chunk_size)
{
 return decode2launch<Scalar>(dims, stride, stream, offset_table, d_data, param, mode, chunk_size);
}

} // namespace cuZFP

#endif
