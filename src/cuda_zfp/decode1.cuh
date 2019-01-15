#ifndef CUZFP_DECODE1_CUH
#define CUZFP_DECODE1_CUH

#include "shared.h"
#include "decode.cuh"
#include "type_info.cuh"

namespace cuZFP {


template<typename Scalar> 
__device__ __host__ inline 
void scatter_partial1(const Scalar* q, Scalar* p, int nx, int sx)
{
  uint x;
  for (x = 0; x < nx; x++, p += sx)
   *p = *q++;
}

template<typename Scalar> 
__device__ __host__ inline 
void scatter1(const Scalar* q, Scalar* p, int sx)
{
  uint x;
  for (x = 0; x < 4; x++, p += sx)
    *p = *q++;
}

template<class Scalar>
__global__
void
cudaDecode1(Word *blocks,
            Word *offset_table,
            Scalar *out,
            const uint dim,
            const int stride,
            const uint padded_dim,
            const uint total_blocks,
            const int param,
            const zfp_mode mode,
            const uint chunk_size)
{
  typedef unsigned long long int ull;
  typedef long long int ll;
  typedef typename zfp_traits<Scalar>::UInt UInt;
  typedef typename zfp_traits<Scalar>::Int Int;

  const ull blockId = blockIdx.x +
                      blockIdx.y * gridDim.x +
                      gridDim.x  * gridDim.y * blockIdx.z;

  const ull thread_idx = blockId * blockDim.x + threadIdx.x;
  
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

  BlockReader<4> reader(blocks, bit_offset);

  for (; block_idx < bmax; block_idx++)
  {
    Scalar result[4] = {0};
    zfp_decode<Scalar,4>(reader, result, param, mode, 1);

    uint block;
    block = block_idx * 4ull;
    const ll offset = (ll)block * stride;
    if(block + 4 > dim)
    {
      const uint nx = 4u - (padded_dim - dim);
      scatter_partial1(result, out + offset, nx, stride);
    }
    else
    {
      scatter1(result, out + offset, stride);
    }
  }
}

template<class Scalar>
size_t decode1launch(uint dim, 
                     int stride,
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
  uint zfp_pad(dim);
  if(zfp_pad % 4 != 0) zfp_pad += 4 - dim % 4;
  uint zfp_blocks = (zfp_pad) / 4; 
  if(dim % 4 != 0)  zfp_blocks = (dim + (4 - dim % 4)) / 4;

  switch(mode) {
    case zfp_mode_fixed_rate:
      cuda_block_size = 128;
      stream_bytes = calc_device_mem1d(zfp_pad, param);
      total_blocks = zfp_blocks;
      if(zfp_blocks % cuda_block_size != 0)
        total_blocks += (cuda_block_size - zfp_blocks % cuda_block_size);
      break;
    case zfp_mode_fixed_accuracy:
    case zfp_mode_fixed_precision:
      /* TODO: Encode chunk size in zfp header or as input argument
      TODO: Set stream bytes to the actual size */
      cuda_block_size = 64;
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

  cudaDecode1<Scalar> << < grid_size, block_size >> >
    (stream,
     offset_table,
     d_data,
     dim,
     stride,
     zfp_pad,
     zfp_blocks, // total blocks to decode
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
  float rate = (float(dim) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("%.2f",rate);
//  printf("Decode elapsed time: %.5f (s)\n", seconds);
//  printf("# decode1 rate: %.2f (GB / sec) %d\n", rate, param);
#endif
  return stream_bytes;
}

template<class Scalar>
size_t decode1(int dim,
               int stride,
               Word *stream,
               Word *offset_table,
               Scalar *d_data,
               int param,
               zfp_mode mode,
               uint chunk_size)
{
	return decode1launch<Scalar>(dim, stride, stream, offset_table, d_data, param, mode, chunk_size);
}

} // namespace cuZFP

#endif
