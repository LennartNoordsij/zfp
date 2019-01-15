#include <assert.h>

#include "cuZFP.h"

#include "encode1.cuh"
#include "encode2.cuh"
#include "encode3.cuh"

#include "decode1.cuh"
#include "decode2.cuh"
#include "decode3.cuh"

#include "ErrorCheck.h"

#include "constant_setup.cuh"
#include "pointers.cuh"
#include "type_info.cuh"
#include <iostream>
#include <assert.h>

// we need to know about bitstream, but we don't 
// want duplicate symbols.
#ifndef inline_
  #define inline_ inline
#endif

#include "../inline/bitstream.c"
namespace internal 
{ 
  
bool is_contigous3d(const uint dims[3], const int3 &stride, long long int &offset)
{
  typedef long long int int64;
  int64 idims[3];
  idims[0] = dims[0];
  idims[1] = dims[1];
  idims[2] = dims[2];

  int64 imin = std::min(stride.x,0) * (idims[0] - 1) + 
               std::min(stride.y,0) * (idims[1] - 1) + 
               std::min(stride.z,0) * (idims[2] - 1);

  int64 imax = std::max(stride.x,0) * (idims[0] - 1) + 
               std::max(stride.y,0) * (idims[1] - 1) + 
               std::max(stride.z,0) * (idims[2] - 1);
  offset = imin;
  int64 ns = idims[0] * idims[1] * idims[2];

  return (imax - imin + 1 == ns);
}

bool is_contigous2d(const uint dims[3], const int3 &stride, long long int &offset)
{
  typedef long long int int64;
  int64 idims[2];
  idims[0] = dims[0];
  idims[1] = dims[1];

  int64 imin = std::min(stride.x,0) * (idims[0] - 1) + 
               std::min(stride.y,0) * (idims[1] - 1);

  int64  imax = std::max(stride.x,0) * (idims[0] - 1) + 
                std::max(stride.y,0) * (idims[1] - 1); 

  offset = imin;
  return (imax - imin + 1) == (idims[0] * idims[1]);
}

bool is_contigous1d(uint dim, const int &stride, long long int &offset)
{
  offset = 0;
  if(stride < 0) offset = stride * (int(dim) - 1);
  return std::abs(stride) == 1;
}

bool is_contigous(const uint dims[3], const int3 &stride, long long int &offset)
{
  int d = 0;
  
  if(dims[0] != 0) d++;
  if(dims[1] != 0) d++;
  if(dims[2] != 0) d++;

  if(d == 3)
  {
    return is_contigous3d(dims, stride, offset);
  }
  else if(d == 2)
  {
   return is_contigous2d(dims, stride, offset);
  }
  else
  {
    return is_contigous1d(dims[0], stride.x, offset);
  } 
}
//
// encode expects device pointers
//
template<typename T>
size_t encode(uint dims[3], int3 stride, int bits_per_block, T *d_data, Word *d_stream)
{

  int d = 0;
  size_t len = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      d++;
      len *= dims[i];
    }
  }

  ErrorCheck errors;
  size_t stream_size = 0;
  if(d == 1)
  {
    int dim = dims[0];
    int sx = stride.x;
    cuZFP::ConstantSetup::setup_1d();
    stream_size = cuZFP::encode1<T>(dim, sx, d_data, d_stream, bits_per_block); 
  }
  else if(d == 2)
  {
    uint2 ndims = make_uint2(dims[0], dims[1]);
    int2 s;
    s.x = stride.x; 
    s.y = stride.y; 
    cuZFP::ConstantSetup::setup_2d();
    stream_size = cuZFP::encode2<T>(ndims, s, d_data, d_stream, bits_per_block); 
  }
  else if(d == 3)
  {
    int3 s;
    s.x = stride.x; 
    s.y = stride.y; 
    s.z = stride.z; 
    uint3 ndims = make_uint3(dims[0], dims[1], dims[2]);
    cuZFP::ConstantSetup::setup_3d();
    stream_size = cuZFP::encode<T>(ndims, s, d_data, d_stream, bits_per_block); 
  }

  errors.chk("Encode");

  return stream_size; 
}

template<typename T>
size_t decode(uint ndims[3], int3 stride, Word *stream, Word *offset_table, T *out, int param, zfp_mode mode, uint chunk_size)
{
  int d = 0;
  size_t out_size = 1;
  size_t stream_bytes = 0;
  for(int i = 0; i < 3; ++i)
  {
    if(ndims[i] != 0)
    {
      d++;
      out_size *= ndims[i];
    }
  }

  if(d == 3)
  {
    uint3 dims = make_uint3(ndims[0], ndims[1], ndims[2]);

    int3 s;
    s.x = stride.x; 
    s.y = stride.y; 
    s.z = stride.z; 

    cuZFP::ConstantSetup::setup_3d();
    stream_bytes = cuZFP::decode3<T>(dims, s, stream, offset_table, out, param, mode, chunk_size);
  }
  else if(d == 1)
  {
    uint dim = ndims[0];
    int sx = stride.x;

    cuZFP::ConstantSetup::setup_1d();
    stream_bytes = cuZFP::decode1<T>(dim, sx, stream, offset_table, out, param, mode, chunk_size);
  }
  else if(d == 2)
  {
    uint2 dims;
    dims.x = ndims[0];
    dims.y = ndims[1];

    int2 s;
    s.x = stride.x; 
    s.y = stride.y; 

    cuZFP::ConstantSetup::setup_2d();
    stream_bytes = cuZFP::decode2<T>(dims, s, stream, offset_table, out, param, mode, chunk_size);
  }
  else std::cerr<<" d ==  "<<d<<" not implemented\n";
 
  return stream_bytes;
}

Word *setup_device_stream(zfp_stream *stream,const zfp_field *field)
{
  bool stream_device = cuZFP::is_gpu_ptr(stream->stream->begin);
  assert(sizeof(word) == sizeof(Word)); // "CUDA version currently only supports 64bit words");

  if(stream_device)
  {
    return (Word*) stream->stream->begin;
  } 

  Word *d_stream = NULL;
  /* read the size of the buffer from the zfp_stream struct */
  size_t size = stream->size;
  if (cudaSuccess != cudaMalloc(&d_stream, size))
    std::cerr<<"failed to allocate device memory for stream\n";
  if (cudaSuccess != cudaMemcpy(d_stream, stream->stream->begin, size, cudaMemcpyHostToDevice))
    std::cerr<<"failed to copy stream from host to device\n";
  return d_stream;
}

Word *setup_device_offset_table(zfp_stream *stream, const size_t size)
{
  bool stream_device = cuZFP::is_gpu_ptr(stream->offset_table);
  assert(sizeof(uint64) == sizeof(Word)); // "CUDA version currently only supports 64bit words");

  if(stream_device)
  {
    return (Word*) stream->offset_table;
  }

  Word *d_offset_table = NULL;
  if (cudaSuccess != cudaMalloc(&d_offset_table, size))
    std::cerr<<"failed to allocate device memory for offset_table\n";
  if (cudaSuccess != cudaMemcpy(d_offset_table, stream->offset_table, size, cudaMemcpyHostToDevice))
    std::cerr<<"failed to copy stream from host to device\n";
  return d_offset_table;
}

void * offset_void(zfp_type type, void *ptr, long long int offset)
{
  void * offset_ptr = NULL;
  if(type == zfp_type_float)
  {
    float* data = (float*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  else if(type == zfp_type_double)
  {
    double* data = (double*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  else if(type == zfp_type_int32)
  {
    int * data = (int*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  else if(type == zfp_type_int64)
  {
    long long int * data = (long long int*) ptr;
    offset_ptr = (void*)(&data[offset]);
  }
  return offset_ptr;
}

void *setup_device_field(const zfp_field *field, const int3 &stride, long long int &offset)
{
  bool field_device = cuZFP::is_gpu_ptr(field->data);

  if(field_device)
  {
    offset = 0;
    return field->data;
  }
  
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  size_t type_size = zfp_type_size(field->type);

  size_t field_size = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      field_size *= dims[i];
    }
  }

  bool contig = internal::is_contigous(dims, stride, offset);
  
  void * host_ptr = offset_void(field->type, field->data, offset);;

  void *d_data = NULL;
  if(contig)
  {
    size_t field_bytes = type_size * field_size;
    if (cudaSuccess != cudaMalloc(&d_data, field_bytes))
      std::cerr<<"failed to allocate device memory for field\n";

    cudaMemcpy(d_data, host_ptr, field_bytes, cudaMemcpyHostToDevice);
  }
  return offset_void(field->type, d_data, -offset);
}

void cleanup_device_ptr(void *orig_ptr, void *d_ptr, size_t bytes, long long int offset, zfp_type type)
{
  bool device = cuZFP::is_gpu_ptr(orig_ptr);
  if(device)
  {
    return;
  }
  // from whence it came
  void *d_offset_ptr = offset_void(type, d_ptr, offset);
  void *h_offset_ptr = offset_void(type, orig_ptr, offset);

  if(bytes > 0)
  {
    cudaMemcpy(h_offset_ptr, d_offset_ptr, bytes, cudaMemcpyDeviceToHost);
  }

  cudaFree(d_offset_ptr);
}

} // namespace internal

size_t
cuda_compress(zfp_stream *stream, const zfp_field *field)
{
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  int3 stride;  
  stride.x = field->sx ? field->sx : 1;
  stride.y = field->sy ? field->sy : field->nx;
  stride.z = field->sz ? field->sz : field->nx * field->ny;
  
  size_t stream_bytes = 0;
  long long int offset = 0; 
  void *d_data = internal::setup_device_field(field, stride, offset);

  if(d_data == NULL)
  {
    // null means the array is non-contiguous host mem which is not supported
    return 0;
  }

  Word *d_stream = internal::setup_device_stream(stream, field);

  if(field->type == zfp_type_float)
  {
    float* data = (float*) d_data;
    stream_bytes = internal::encode<float>(dims, stride, (int)stream->maxbits, data, d_stream);
  }
  else if(field->type == zfp_type_double)
  {
    double* data = (double*) d_data;
    stream_bytes = internal::encode<double>(dims, stride, (int)stream->maxbits, data, d_stream);
  }
  else if(field->type == zfp_type_int32)
  {
    int * data = (int*) d_data;
    stream_bytes = internal::encode<int>(dims, stride, (int)stream->maxbits, data, d_stream);
  }
  else if(field->type == zfp_type_int64)
  {
    long long int * data = (long long int*) d_data;
    stream_bytes = internal::encode<long long int>(dims, stride, (int)stream->maxbits, data, d_stream);
  }

  internal::cleanup_device_ptr(stream->stream->begin, d_stream, stream_bytes, 0, field->type);
  internal::cleanup_device_ptr(field->data, d_data, 0, offset, field->type);

  // zfp wants to flush the stream.
  // set bits to wsize because we already did that.
  size_t compressed_size = stream_bytes / sizeof(Word);
  stream->stream->bits = wsize;
  // set stream pointer to end of stream
  stream->stream->ptr = stream->stream->begin + compressed_size;

  return stream_bytes;
}
  
void 
cuda_decompress(zfp_stream *stream, zfp_field *field)
{
  uint dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  int3 stride;  
  stride.x = field->sx ? field->sx : 1;
  stride.y = field->sy ? field->sy : field->nx;
  stride.z = field->sz ? field->sz : field->nx * field->ny;

  size_t decoded_bytes = 0;
  long long int offset = 0;
  void *d_data = internal::setup_device_field(field, stride, offset);

  if(d_data == NULL)
  {
    // null means the array is non-contiguous host mem which is not supported
    return;
  }

  Word *d_stream = internal::setup_device_stream(stream, field);
  Word * d_offset_table = NULL;
  zfp_mode mode = zfp_stream_compression_mode(stream);
  uint chunk_size = zfp_stream_cuda_chunk_size(stream);

  /* parameter needed to decode the bitstream differs per execution policy */
  size_t table_size;
  uint blocks, chunks;
  int param;
  switch(mode) {
    case zfp_mode_fixed_rate:
      param = (int)stream->maxbits;
      break;
    case zfp_mode_fixed_accuracy:
    case zfp_mode_fixed_precision:
      if (!chunk_size) {
        std::cerr<<"chunk size not specified for CUDA variable rate decompression\n";
        exit(EXIT_FAILURE);
      }
      blocks = 1;
      if (dims[0]) blocks *= ((dims[0] + 3)/4);
      if (dims[1]) blocks *= ((dims[1] + 3)/4);
      if (dims[2]) blocks *= ((dims[2] + 3)/4);
      chunks = (blocks + chunk_size - 1) / chunk_size;
      table_size = (size_t)chunks * sizeof(Word);
      /* TODO: make the data type of offset table coherent with uint64 of the C functions */
      d_offset_table = internal::setup_device_offset_table(stream, table_size);
      param = (mode == zfp_mode_fixed_accuracy ? (int)stream->minexp : (int)stream->maxprec);
      break;
    default :
      std::cerr<<"Custom execution is not supported in this CUDA release\n";
      exit(EXIT_FAILURE);
  }

  if(field->type == zfp_type_float)
  {
    float *data = (float*) d_data;
    decoded_bytes = internal::decode(dims, stride, d_stream, d_offset_table, data, param, mode, chunk_size);
    d_data = (void*) data;
  }
  else if(field->type == zfp_type_double)
  {
    double *data = (double*) d_data;
    decoded_bytes = internal::decode(dims, stride, d_stream, d_offset_table, data, param, mode, chunk_size);
    d_data = (void*) data;
  }
  else if(field->type == zfp_type_int32)
  {
    int *data = (int*) d_data;
    decoded_bytes = internal::decode(dims, stride, d_stream, d_offset_table, data, param, mode, chunk_size);
    d_data = (void*) data;
  }
  else if(field->type == zfp_type_int64)
  {
    long long int *data = (long long int*) d_data;
    decoded_bytes = internal::decode(dims, stride, d_stream, d_offset_table, data, param, mode, chunk_size);
    d_data = (void*) data;
  }
  else
  {
    std::cerr<<"Cannot decompress: type unknown\n";
  }

  size_t type_size = zfp_type_size(field->type);

  size_t field_size = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      field_size *= dims[i];
    }
  }

  size_t bytes = type_size * field_size;
  internal::cleanup_device_ptr(stream->stream, d_stream,0, 0, field->type);
  internal::cleanup_device_ptr(field->data, d_data, bytes, offset, field->type);

  // this is how zfp determines if this was a success
  /* If the mode is fixed accuracy or precision, decoded_bytes is not computed on correct execution */
  size_t words_read;
  switch(mode) {
    case zfp_mode_fixed_rate:
      words_read = decoded_bytes / sizeof(Word);
      break;
    case zfp_mode_fixed_accuracy:
    case zfp_mode_fixed_precision:
      /* TODO: get correct type for cleanup or make custom cleanup function (currently only accepts zfp_type types) */
      internal::cleanup_device_ptr(stream->offset_table, d_offset_table, table_size, 0, zfp_type_int64);
      words_read = stream->size / sizeof(Word);
      break;
  }
  stream->stream->bits = wsize;
  // set stream pointer to end of stream
  stream->stream->ptr = stream->stream->begin + words_read;

}

