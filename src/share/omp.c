#ifdef _OPENMP
#include <omp.h>
#include <stdio.h>

/* number of omp threads to use */
static int
thread_count_omp(const zfp_stream* stream)
{
  int count = stream->exec.params.omp.threads;
  /* if no thread count is specified, use default number of threads */
  if (!count)
    count = omp_get_max_threads();
  return count;
}

/* number of chunks to partition array into */
static uint
chunk_count_omp(const zfp_stream* stream, uint blocks, uint threads)
{
  uint chunk_size = stream->exec.params.omp.chunk_size;
  /* if no chunk size is specified, assign one chunk per thread */
  uint chunks = chunk_size ? (blocks + chunk_size - 1) / chunk_size : threads;
  return MIN(chunks, blocks);
}

/* TODO: consider moving this to zfp.c to allow serial side channel encoding */
static int
encode_side_channel(const zfp_stream* stream, uint blocks)
{
  const side_channel_type table_type = stream->side_channel->type;
  const uint16* length_table = stream->side_channel->length_table;
  if (table_type != none) {
    if (table_type == offset) {
      uint chunk_size = stream->side_channel->chunk_size;
      uint64* offset_table = (uint64*)stream->side_channel->side_channel_data;
      uint64 sum = 0;
      int i, chunk, block, chunks;
      chunks = (blocks + chunk_size - 1) / chunk_size;
      for (chunk = 0, block = 0; chunk < chunks; chunk++) {
        offset_table[chunk] = sum;
        for (i = 0; i < chunk_size; i++, block++) {
          sum += length_table[block];
        }
      }
    }
    else if (table_type == hybrid) {
      const uint chunk_size = stream->side_channel->chunk_size;
      const uint chunks = (blocks + chunk_size - 1) / chunk_size;
      uint16* hybrid_table16 = (uint16*)stream->side_channel->side_channel_data;
      uint64* hybrid_table64 = (uint64*)stream->side_channel->side_channel_data;
      const uint partitions = (chunks + PARTITION_SIZE - 1) / PARTITION_SIZE;
      uint j = 0;
      uint lim = 0;
      uint64 sum = 0;
      uint16 partialsum = 0;
      uint i = 0;
      uint chunk = 0;
      uint a = 0;
      for (; i < partitions; i++) {
      /* store the offset for partition i on position i*PARTITION_BYTES */
        hybrid_table64[i * 9] = sum;
        for (chunk = 0; chunk < PARTITION_SIZE; chunk++) {
          partialsum = 0;
          for (a = 0; a < chunk_size && j < blocks; a++, j++) {
            partialsum += length_table[j];
          }
          hybrid_table16[i * 36 + 4 + chunk] = partialsum;
          sum += partialsum;
        }
      }
      /* Finish the last partial partition */
      for(; chunk < PARTITION_SIZE; chunk++) {
        hybrid_table16[i * 36 + 4 + chunk] = 0;
      }
    }
    else if (table_type == length) {
      /* possible encoding of a new lengths table 
      current implementation: write fixed size (2 bytes per block) lengths table to file */
      if (stream->side_channel->side_channel_data != length_table)
        fprintf(stderr,"side channel information is not the original lengths table\n");
    }
    else {
      fprintf(stderr,"unknown side-channel information type to be encoded\n");
    }
  }
}

#endif
