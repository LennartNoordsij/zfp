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
            Word *side_channel,
            Scalar *out,
            const uint2 dims,
            const int2 stride,
            const uint2 padded_dims,
            const uint total_blocks,
            const int param,
            const uint chunk_size,
            const zfp_mode mode,
            const side_channel_type table_type)
{
  typedef unsigned long long int ull;
  typedef long long int ll;

  const uint blockId = blockIdx.x +
                      blockIdx.y * gridDim.x +
                      gridDim.x * gridDim.y * blockIdx.z;
  const uint chunk_idx = blockId * blockDim.x + threadIdx.x;
  const int warp_idx = blockId * blockDim.x / 32;
  const int thread_idx = threadIdx.x;

  ll bit_offset;
  if (mode == zfp_mode_fixed_rate)
    bit_offset = param * chunk_idx;
  else if (table_type == offset){
    bit_offset = side_channel[chunk_idx];
  }
  else if (table_type == hybrid) {
    __shared__ uint64 offsets[32];
    uint64* data64 = (uint64 *)side_channel;
    uint16* data16 = (uint16 *)side_channel;
    data16 += warp_idx * 36 + 3;
    offsets[thread_idx] = (uint64)data16[thread_idx];
    offsets[0] = data64[warp_idx * 9];
    int j;
    
    for (int i = 0; i < 5; i++) {
      j = (1 << i);
      if (thread_idx + j < 32) {
        offsets[thread_idx + j] += offsets[thread_idx];
      }
      __syncthreads();
    }
    bit_offset = offsets[thread_idx];
  }

  // logical block dims
  uint2 block_dims;
  block_dims.x = padded_dims.x >> 2;
  block_dims.y = padded_dims.y >> 2;

  BlockReader<BlockSize> reader(blocks, bit_offset);
  uint block_idx = chunk_idx * chunk_size;
  const uint lim = MIN(block_idx + chunk_size, total_blocks);
  
  for (; block_idx < lim; block_idx++) {
    Scalar result[BlockSize] = {0};
    zfp_decode<Scalar,BlockSize>(reader, result, param, mode, 2);

    // logical pos in 3d array
    uint2 block;
    block.x = (block_idx % block_dims.x) * 4;
    block.y = ((block_idx / block_dims.x) % block_dims.y) * 4;

    const ll offset = (ll)block.x * stride.x + (ll)block.y * stride.y;

    if(block.x + 4 > dims.x || block.y + 4 > dims.y) {
      const uint nx = block.x + 4 > dims.x ? dims.x - block.x : 4;
      const uint ny = block.y + 4 > dims.y ? dims.y - block.y : 4;
      scatter_partial2(result, out + offset, nx, ny, stride.x, stride.y);
    }
    else
      scatter2(result, out + offset, stride.x, stride.y);
  }
}


template<class Scalar>
size_t decode2launch(uint2 dims,
                     int2 stride,
                     Word *stream,
                     Word *side_channel,
                     Scalar *d_data,
                     int param,
                     uint chunk_size,
                     zfp_mode mode, 
                     side_channel_type table_type)
{
  uint2 zfp_pad(dims);
  // ensure that we have block sizes
  // that are a multiple of 4
  if(zfp_pad.x % 4 != 0) zfp_pad.x += 4 - dims.x % 4;
  if(zfp_pad.y % 4 != 0) zfp_pad.y += 4 - dims.y % 4;
  const uint zfp_blocks = (zfp_pad.x / 4) * (zfp_pad.y / 4); 

  /* Block size fixed to 32 in this version, needed for hybrid functionality */
  size_t cuda_block_size = 32;
  /* TODO: remove nonzero stream_bytes requirement */
  size_t stream_bytes = 1;
  size_t chunks = (zfp_blocks + (size_t)chunk_size - 1) / chunk_size;
  if(chunks % cuda_block_size != 0)
    chunks += (cuda_block_size - chunks % cuda_block_size);
  dim3 block_size = dim3(cuda_block_size, 1, 1);
  dim3 grid_size = calculate_grid_size(chunks, cuda_block_size);

#ifdef CUDA_ZFP_RATE_PRINT
  // setup some timing code
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
#endif

  cudaDecode2<Scalar, 16> << < grid_size, block_size >> >
    (stream,
     side_channel,
     d_data,
     dims,
     stride,
     zfp_pad,
     zfp_blocks,
     param,
     chunk_size,
     mode,
     table_type);

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
  printf("%f", rate);
#endif
  return stream_bytes;
}



template<class Scalar>
size_t decode2(uint2 dims, 
               int2 stride,
               Word *stream,
               Word *side_channel,
               Scalar *d_data,
               int param,
               uint chunk_size,
               zfp_mode mode,
               side_channel_type table_type)
{
 return decode2launch<Scalar>(dims, stride, stream, side_channel, d_data, param, chunk_size, mode, table_type);
}

} // namespace cuZFP

#endif
