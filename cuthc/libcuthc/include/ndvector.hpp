
#pragma once
#include <cstddef>
#include <cstdio>
#include <iostream>

#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#include <thrust/zip_function.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>

#include "tuple.hpp"

template <int N>
class NDVector;

template <int N>
auto zip_iterator(NDVector<N> &ndvec)
{
  return thrust::make_zip_iterator([&]<size_t... Is>(
                                       std::index_sequence<Is...>)
                                   { return thrust::make_tuple(ndvec.vecs[Is].begin()...); }(std::make_index_sequence<N>{}));
};

// return the zip iterator for the first M columns
template <int N, int M>
auto zip_head_iterator(NDVector<N> &ndvec)
{
  return thrust::make_zip_iterator([&]<size_t... Is>(
                                       std::index_sequence<Is...>)
                                   { return thrust::make_tuple(ndvec.vecs[Is].begin()...); }(std::make_index_sequence<M>{}));
};

// a N-dimensional vector, each dimension is a device vector
template <int N>
class NDVector
{

public:
  size_t dim;
  size_t size;
  std::array<rmm::device_vector<int>, N> vecs;
  rmm::device_vector<int> indices;
  typedef std::array<int, N> OrderArray;

  NDVector() : size(0) {}

  NDVector(size_t size) : size(size)
  {
    for (int i = 0; i < N; i++)
    {
      vecs[i].resize(size);
    }
    dim = N;
  }

  void host_copy_column(int col, thrust::host_vector<int> &host_vec)
  {
    cudaMemcpy(host_vec.data(), vecs[col].data().get(), size * sizeof(int),
               cudaMemcpyDeviceToHost);
  }

  void print_row_before(int row_num)
  {
    for (int i = 0; i < row_num; i++)
    {
      for (int j = 0; j < N; j++)
      {
        std::cout << vecs[j][i] << " ";
      }
      std::cout << std::endl;
    }
  }

  void resize(size_t new_size)
  {
    for (int i = 0; i < N; i++)
    {
      vecs[i].resize(new_size);
    }
    size = new_size;
  }

  size_t get_size() { return vecs[0].size(); }

  rmm::device_vector<int *> raw_column_dptrs(int start_row = 0)
  {
    rmm::device_vector<int *> dptrs(N);
#pragma unroll
    for (int i = 0; i < N; i++)
    {
      dptrs[i] = vecs[i].data().get() + start_row;
    }
    return dptrs;
  }

  void lexical_reorder(OrderArray order)
  {
    std::array<rmm::device_vector<int>, N> vecs_tmp;
#pragma unroll
    for (int i = 0; i < N; i++)
    {
      vecs_tmp[i].swap(vecs[order[i]]);
    }
    vecs.swap(vecs_tmp);
    sort_ndvector(*this);
  }

  void swap(NDVector<N> &other)
  {
    vecs.swap(other.vecs);
    indices.swap(other.indices);
    size = other.size;
  }

  auto begin() {
    return zip_iterator(*this);
  }

  auto end() {
    return zip_iterator(*this) + size;
  }
};

template <int N>
void *init_ndvector_indices(NDVector<N> &ndvec)
{
  ndvec.indices.resize(ndvec.size);
  thrust::sequence(rmm::exec_policy(), ndvec.indices.begin(),
                   ndvec.indices.end());
  return &ndvec.indices;
}

// void sort_vector(rmm::device_vector<int> &vec) {
//   thrust::sort(vec.begin(), vec.end());
// }

// gather a NDVector from another NDVector based on indices
template <int N>
void gather_ndvector(NDVector<N> &ndvec, rmm::device_vector<int> &indices,
                     NDVector<N> &result)
{
  result.resize(ndvec.get_size());
  auto z_begin = zip_iterator(ndvec);
  auto z_end = z_begin + ndvec.size;
  auto result_begin = zip_iterator(result);
  thrust::gather(rmm::exec_policy(), indices.begin(), indices.end(), z_begin,
                 result_begin);
}

template <int N>
void merge_ndvector(NDVector<N> &ndvec, NDVector<N> &ndvec2,
                    NDVector<N> &result)
{
  result.resize(ndvec.size + ndvec2.size);
  auto z_begin = zip_iterator(ndvec);
  auto z_end = z_begin + ndvec.size;
  auto z2_begin = zip_iterator(ndvec2);
  auto z2_end = z2_begin + ndvec2.size;
  auto result_begin = zip_iterator(result);
  thrust::merge(rmm::exec_policy(), z_begin, z_end, z2_begin, z2_end,
                result_begin);

  result.size = ndvec.size + ndvec2.size;
}

template <int N>
void clear_ndvector(NDVector<N> &ndvec)
{
  for (int i = 0; i < N; i++)
  {
    ndvec.vecs[i].clear();
  }
  ndvec.size = 0;
}

// sort a N-dimensional vector based lexicographically
template <int N>
void sort_ndvector(NDVector<N> &ndvec)
{
  auto z_begin = zip_iterator(ndvec);
  // try tran
  auto z_end = z_begin + ndvec.size;
  thrust::sort(rmm::exec_policy(), z_begin, z_end);
}

// remove duplicates from a sorted vector
template <int N>
void unique_ndvector(NDVector<N> &ndvec)
{
  auto z_begin = zip_iterator(ndvec);
  auto z_end = z_begin + ndvec.size;
  auto unique_z_end = thrust::unique(rmm::exec_policy(), z_begin, z_end);
  auto unique_size = unique_z_end - z_begin;
  ndvec.resize(unique_size);
}

// sort a ndvector indices based on the values of a column
template <int N>
void *sort_ndvector_indices(NDVector<N> &ndvec, int col)
{
  auto z_begin = zip_iterator(ndvec);
  auto z_end = z_begin + ndvec.size;
  init_ndvector_indices(ndvec);
  thrust::sort_by_key(rmm::exec_policy(), ndvec.vecs[col].begin(),
                      ndvec.vecs[col].end(), ndvec.indices.begin());
  return &ndvec.indices;
}

template <int N>
void search_ndvector(NDVector<N> &ndvec, NDVector<N> &search_ndvec,
                     rmm::device_vector<bool> &result)
{
  auto z_begin = zip_iterator(ndvec);
  auto z_end = z_begin + ndvec.get_size();
  auto search_begin = zip_iterator(search_ndvec);
  auto search_end = search_begin + search_ndvec.get_size();

  thrust::binary_search(rmm::exec_policy(), z_begin, z_end, search_begin,
                        search_end, result.begin());
}

template <int N>
__device__ void load_tuple(int **vec, int pos, int *tuple)
{
#pragma unroll
  for (int i = 0; i < N; i++)
  {
    tuple[i] = vec[i][pos];
  }
}

template <int N>
__device__ bool binary_find_shared_prefix_seq(int **vec, int *tuple, int size)
{
  int left = 0;
  int right = size - 1;
  while (left <= right)
  {
    int mid = left + (right - left) / 2;
    int src_tuple[N];
    load_tuple<N>(vec, mid, src_tuple);
    int cmp = compare_tuple<N>(src_tuple, tuple);
    if (cmp == 0)
    {
      return true;
    }
    else if (cmp < 0)
    {
      left = mid + 1;
    }
    else
    {
      right = mid - 1;
    }
  }
  return false;
}

// search lower bound of a tuple in a sorted unique vector
// if not found, return size
template <int N>
__device__ int binary_find(int **vec, int *tuple, int size)
{
  int left = 0;
  int right = size - 1;
  while (left <= right)
  {
    int mid = left + (right - left) / 2;
    int src_tuple[N];
    load_tuple<N>(vec, mid, src_tuple);
    int cmp = compare_tuple<N>(src_tuple, tuple);
    if (cmp == 0)
    {
      return mid;
    }
    else if (cmp < 0)
    {
      left = mid + 1;
    }
    else
    {
      right = mid - 1;
    }
  }
  return size;
}

// search but using cuda kernel
template <int N>
__global__ void search_ndvector_kernel(int **vec, int **search_vec,
                                       bool *result, int size,
                                       int search_size, bool clean_dup = false)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < search_size; i += stride)
  {
    // construct currnt search tuple
    int search_tuple[N];
    load_tuple<N>(search_vec, i, search_tuple);
    // printf("search_tuple: %d %d\n", search_tuple[0], search_tuple[1]);
    // binary search base on lexicographical order
    bool found = binary_find_shared_prefix_seq<N>(vec, search_tuple, size);
    if (clean_dup && found)
    {
#pragma unroll
      for (int ar = 0; ar < N; ar++)
      {
        search_vec[ar][i] = EMPTY_ENTRY;
      }
    }
  }
}

template <int N>
void search_ndvector_kernel_wrapper(NDVector<N> &ndvec,
                                    NDVector<N> &search_ndvec,
                                    rmm::device_vector<bool> &result)
{
  auto vec_dptrs_d = ndvec.raw_column_dptrs();
  int **vec_dptrs = vec_dptrs_d.data().get();
  auto search_ndvec_dptrs_d = search_ndvec.raw_column_dptrs();
  int **search_vec_dptrs = search_ndvec_dptrs_d.data().get();

  bool *result_dptr = result.data().get();

  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
  int block_size = 256;
  int grid_size = sm_count * 32;
  search_ndvector_kernel<N><<<grid_size, block_size>>>(
      vec_dptrs, search_vec_dptrs, result_dptr, ndvec.get_size(), search_ndvec.get_size());
}

template <int N>
void remove_ndvector_by_stencil(NDVector<N> &ndvec,
                                rmm::device_vector<bool> &stencil)
{
  auto z_begin = zip_iterator(ndvec);
  auto z_end = z_begin + ndvec.size;
  auto new_end = thrust::remove_if(rmm::exec_policy(), z_begin, z_end,
                                   stencil.begin(), thrust::identity());
  auto new_size = new_end - z_begin;
  ndvec.resize(new_size);
}

template <int N, int M>
void histogram_ndvector(
    NDVector<N> &ndvec, NDVector<M> &hist_val,
    rmm::device_vector<int> &hist_cnt)
{
  auto z_begin = zip_head_iterator<N, M>(ndvec);
  auto z_end = z_begin + ndvec.get_size();
  // compute the sparse histogram bucket size
  size_t num_buckets = thrust::inner_product(
      rmm::exec_policy(),
      z_begin, z_end - 1,
      z_begin + 1, 1,
      thrust::plus<int>(),
      TupleNotEqualTo<M>());
  hist_cnt.resize(num_buckets);
  hist_val.resize(num_buckets);
  // compute the histogram
  thrust::reduce_by_key(
      rmm::exec_policy(),
      z_begin, z_end,
      thrust::constant_iterator<int>(1),
      hist_val.begin(),
      hist_cnt.begin());
  // compute the prefix sum for cnt
  thrust::exclusive_scan(rmm::exec_policy(), hist_cnt.begin(), hist_cnt.end(),
                         hist_cnt.begin());
}


// copy a NDVector to another NDVector
template <int N>
void copy_ndvector(NDVector<N> &ndvec, NDVector<N> &result)
{
  result.resize(ndvec.get_size());
  auto z_begin = zip_iterator(ndvec);
  auto z_end = z_begin + ndvec.get_size();
  auto result_begin = zip_iterator(result);
  thrust::copy(rmm::exec_policy(), z_begin, z_end, result_begin);
}

// indexed vector
template <int N, int J>
class IndexedNDVector
{
public:
  NDVector<N> ndvec;
  NDVector<J> indexed_col;
  rmm::device_vector<int> indexed_start;

  // you can move a NDVector to the constructor
  IndexedNDVector(NDVector<N> &other, std::array<int, N> index)
  {
    // move the NDVector to the indexed vector
    copy_ndvector(other, ndvec);
    ndvec.lexical_reorder(index);
    // sort the indexed_col
    sort_ndvector(indexed_col);
    // compute the histogram
    histogram_ndvector<N, J>(ndvec, indexed_col, indexed_start);
  }
};


// copy a IndexedNDVector to another IndexedNDVector
template <int N, int J>
void copy_indexed_ndvector(IndexedNDVector<N, J> &indexed_ndvec,
                           IndexedNDVector<N, J> &result)
{
  result.ndvec.resize(indexed_ndvec.ndvec.get_size());
  result.indexed_col.resize(indexed_ndvec.indexed_col.get_size());
  result.indexed_start.resize(indexed_ndvec.indexed_start.size());
  copy_ndvector(indexed_ndvec.ndvec, result.ndvec);
  copy_ndvector(indexed_ndvec.indexed_col, result.indexed_col);
  copy_ndvector(indexed_ndvec.indexed_start, result.indexed_start);
}


#define UNINITIALIZED INT32_MAX
#define FINISHED 0
#define MAX_PROBING_COUNT 8
#define RAW_PTR data().get()
// vec1 has N columns, vec2 has M columns
// join on the first K columns
// maybe consider JIT this kernel?
template <int N, int M, int K, int BLOCK_SIZE>
__global__ void equi_join_sub_kernel(
    int **vec1, int **vec2,
    int **vec2_indexed_col, int *vec2_indexed_start,
    int *prev_pos_vec1,
    int *prev_bucket_vec2, int *prev_offset_vec2,
    int **result, int size1, int size2)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // id in thread block
  int bid = threadIdx.x;
  // id in grid
  int gid = blockIdx.x;
  // block size
  const int block_size = blockDim.x;
  // cur probing pos of all thread stored in shared memory
  __shared__ int cur_1_tmp_pos[BLOCK_SIZE];
  cur_1_tmp_pos[bid] = UNINITIALIZED;
  __shared__ int cur_2_tmp_bucket[BLOCK_SIZE];
  cur_2_tmp_bucket[bid] = UNINITIALIZED;
  __shared__ int cur_2_tmp_offset[BLOCK_SIZE];
  cur_2_tmp_offset[bid] = UNINITIALIZED;
  // assume the join result is the second column of vec1 and vec2
  __shared__ int result_tmp[MAX_PROBING_COUNT][BLOCK_SIZE][2];
  // initialize the probing pos
  for (int i = 0; i < MAX_PROBING_COUNT; i++)
  {
    result_tmp[i][bid][0] = UNINITIALIZED;
    result_tmp[i][bid][1] = UNINITIALIZED;
  }

  int *vec1_pos = &(cur_1_tmp_pos[bid]);
  *vec1_pos = prev_pos_vec1[idx];
  int *vec2_offset = &(cur_2_tmp_offset[bid]);
  *vec2_offset = prev_offset_vec2[idx];
  int *vec2_bucket = &(cur_2_tmp_bucket[bid]);
  *vec2_bucket = prev_bucket_vec2[idx];
  int cur_thread_found_count = 0;
  __shared__ int block_thread_found_count;
  // initialize the block_thread_found_count
  if (bid == 0)
  {
    block_thread_found_count = 0;
  }
  __syncthreads();
  // a shared bitmask to indicate if the thread has found at least one tuple
  // processing i th bucket of vec1
  for (int i = *vec1_pos; i < size1; i += stride)
  {
    int tuple1[N];
    load_tuple<N>(vec1, i, tuple1);
    int tuple2[M];
    // check if current probing a bucket in vec2
    if (*vec2_bucket == UNINITIALIZED)
    {
      // if not, binary search in vec2, find the bucket for vec2,
      // probing from the start of the bucket
      int pos_bucket_idx_2 = binary_find<K>(vec2_indexed_col, tuple1, size2);
      // set probing pos
      *vec2_bucket = vec2_indexed_start[pos_bucket_idx_2];
      load_tuple<M>(vec2, *vec2_bucket, tuple2);
      *vec2_offset = 1;
    }
    else
    {
      int cur_bucket_start2 = vec2_indexed_start[*vec2_bucket];
      int cur_bucket_size2 = vec2_indexed_start[*vec2_bucket + 1] - vec2_indexed_start[*vec2_bucket];
      if (*vec2_offset >= cur_bucket_size2)
      {
        // if all tuple the current bucket is probed, move to the next bucket
        *vec2_bucket = UNINITIALIZED;
        // Do we need reset the offset?
        // *vec2_offset = UNINITIALIZED;
        continue;
      }
      // probing the k th tuple in vec2 at bucket *vec2_bucket
      load_tuple<M>(vec2, cur_bucket_start2 + *vec2_offset, tuple2);
      *vec2_offset = *vec2_offset + 1;
    }
    // write result to tmp shared memory
    // load tuple from vec2
    // >>>>>>>>>>>>>>>>>>> use templated functor to specify the column?
    result_tmp[cur_thread_found_count][bid][0] = tuple1[1];
    result_tmp[cur_thread_found_count][bid][1] = tuple2[1];
    // <<<<<<<<<<<<<<<<<<
    cur_thread_found_count++;
    if (cur_thread_found_count == 1)
    {
      // if the first tuple is found, update the global counter
      atomicAdd(&block_thread_found_count, 1);
    }
    // update global counter atomically
    // check if the thread next to it has found at least one tuple
    // if so, break waiting
    if (cur_thread_found_count > MAX_PROBING_COUNT || block_thread_found_count >= block_size)
    {
      goto PROBING_END;
    }
  }
  // all of vec1 are probed, reset the bucket and offset
  *vec1_pos = FINISHED;

  // tag for probing end, for goto
PROBING_END:
  // sync threads in the block, make sure all threads have finished probing
  __syncthreads();

  // write back to global memory
  for (int i = 0; i < cur_thread_found_count; i++)
  {
    // >>>>>>>>>>>>>>>>>>> use templated functor to specify the column?
    result[0][cur_thread_found_count * idx + i] = result_tmp[i][bid][0];
    result[1][cur_thread_found_count * idx + i] = result_tmp[i][bid][1];
    // <<<<<<<<<<<<<<<<<<
  }
  // save the probing pos for next kernel call
  prev_pos_vec1[idx] = *vec1_pos;
  prev_offset_vec2[idx] = *vec2_offset;
  prev_bucket_vec2[idx] = *vec2_bucket;
}

template <int N, int M, int K, int R>
void compute_equi_join(NDVector<N> &ndvec1, IndexedNDVector<M, K> &indexed_ndvec2,
                       NDVector<R> &result)
{
  auto ndvec2_dptrs = indexed_ndvec2.ndvec.raw_column_dptrs();
  result.resize(ndvec1.get_size());
  // number of SMs
  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
  int block_size = 256;
  int grid_size = sm_count * 32;
  auto sub_join_size = grid_size * block_size * 8;
  // create a Tmp NDVector to store the result
  NDVector<R> tmp_result(sub_join_size);

  int new_tuple_size = 0;
  // create tmp vector to store the tmp probing pos
  rmm::device_vector<int> prev_pos_vec1(sub_join_size);
  rmm::device_vector<int> prev_offset_vec2(sub_join_size);
  rmm::device_vector<int> prev_bucket_vec2(sub_join_size);

  bool finished = false;
  while (finished)
  {
    // resize the result vector
    int prev_size = new_tuple_size;
    new_tuple_size = new_tuple_size + sub_join_size;
    result.resize(new_tuple_size);
    // call the kernel
    equi_join_sub_kernel<N, M, K, 256><<<grid_size, block_size>>>(
        ndvec1.raw_column_dptrs().data().get(),
        ndvec2_dptrs.data().get(),
        indexed_ndvec2.indexed_col.raw_column_dptrs().data().get(),
        indexed_ndvec2.indexed_start.data().get(),
        prev_pos_vec1.data().get(),
        prev_bucket_vec2.data().get(), prev_offset_vec2.data().get(),
        tmp_result.raw_column_dptrs(prev_size).data().get(),
        ndvec1.get_size(),
        indexed_ndvec2.ndvec.get_size());

    // Do we need to sync here?
    cudaStreamSynchronize(0);
    // if all probing bucket has value FINISHED, then break
    int bks_tt = thrust::reduce(
        rmm::exec_policy(), prev_pos_vec1.begin(),
        prev_pos_vec1.end(), 0, thrust::plus<int>());
    if (bks_tt == 0)
    {
      break;
    }
  }

  // sort the result
  sort_ndvector(result);
  // remove duplicates
  unique_ndvector(result);
  // remove the last empty entry
  result.resize(result.get_size() - 1);
}
