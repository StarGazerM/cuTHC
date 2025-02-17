
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
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#include <thrust/zip_function.h>
#include <thrust/remove.h>
#include <thrust/merge.h>
#include <thrust/gather.h>

#include "cuthc.h"

// a N-dimensional vector, each dimension is a device vector
template <int N>
class NDVector
{

public:
  size_t dim;
  size_t size;
  std::array<rmm::device_vector<int>, N> vecs;
  rmm::device_vector<int> indices;

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

  size_t get_size()
  {
    return vecs[0].size();
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

void sort_vector(rmm::device_vector<int> &vec)
{
  thrust::sort(vec.begin(), vec.end());
}

#include <utility> // for index_sequence

template <int N>
auto zip_iterator(NDVector<N> &ndvec)
{
  return thrust::make_zip_iterator(
      [&]<size_t... Is>(std::index_sequence<Is...>)
      {
        return thrust::make_tuple(ndvec.vecs[Is].begin()...);
      }(std::make_index_sequence<N>{}));
}

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
void merge_ndvector(NDVector<N> &ndvec, NDVector<N> &ndvec2, NDVector<N> &result)
{
  result.resize(ndvec.size + ndvec2.size);
  auto z_begin = zip_iterator(ndvec);
  auto z_end = z_begin + ndvec.size;
  auto z2_begin = zip_iterator(ndvec2);
  auto z2_end = z2_begin + ndvec2.size;
  auto result_begin = zip_iterator(result);
  thrust::merge(rmm::exec_policy(), z_begin, z_end, z2_begin, z2_end, result_begin);

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
  auto z_end = z_begin + ndvec.size;
  auto search_begin = zip_iterator(search_ndvec);
  auto search_end = search_begin + search_ndvec.size;

  thrust::binary_search(rmm::exec_policy(), z_begin, z_end, search_begin,
                        search_end, result.begin());
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

// search for a value in a sorted vector
void search_vector(rmm::device_vector<int> &vec,
                   rmm::device_vector<int> &search_vec,
                   rmm::device_vector<bool> &result)
{
  thrust::binary_search(thrust::device, vec.begin(), vec.end(),
                        search_vec.begin(), search_vec.end(), result.begin());
}

#define DECLARE_CUTHC_MK_NDVECTOR(DIM)                 \
  void *cuthc_mk_ndvector_##DIM##N(int N, size_t size) \
  {                                                    \
    auto vec = new NDVector<DIM>(size);                \
    return vec;                                        \
  }

#define DECLARE_CUTHC_GATHER_NDVECTOR(DIM)                                    \
  void cuthc_gather_ndvector_##DIM##N(void *ptr, void *indices, void *result) \
  {                                                                           \
    auto vec = static_cast<NDVector<DIM> *>(ptr);                             \
    auto indices_vec = static_cast<rmm::device_vector<int> *>(indices);       \
    auto result_vec = static_cast<NDVector<DIM> *>(result);                   \
    gather_ndvector(*vec, *indices_vec, *result_vec);                          \
  }

#define DECLARE_CUTHC_FREE_NDVECTOR(DIM)          \
  void cuthc_free_ndvector_##DIM##N(void *ptr)    \
  {                                               \
    auto vec = static_cast<NDVector<DIM> *>(ptr); \
    delete vec;                                   \
  }

#define DECLARE_CUTHC_RESIZE_NDVECTOR(DIM)                    \
  void cuthc_resize_ndvector_##DIM##N(void *ptr, size_t size) \
  {                                                           \
    auto vec = static_cast<NDVector<DIM> *>(ptr);             \
    for (int i = 0; i < DIM; i++)                             \
    {                                                         \
      vec->vecs[i].resize(size);                              \
    }                                                         \
    vec->size = size;                                         \
  }

// int* cuthc_raw_ptr_ndvector_##N(void* ptr, size_t col);
#define DECLARE_CUTHC_RAW_PTR_NDVECTOR(DIM)                   \
  int *cuthc_raw_ptr_ndvector_##DIM##N(void *ptr, size_t col) \
  {                                                           \
    auto vec = static_cast<NDVector<DIM> *>(ptr);             \
    return vec->vecs[col].data().get();                       \
  }

// size_t cuthc_size_ndvector_##N(void* ptr);
#define DECLARE_CUTHC_SIZE_NDVECTOR(DIM)          \
  size_t cuthc_size_ndvector_##DIM##N(void *ptr)  \
  {                                               \
    auto vec = static_cast<NDVector<DIM> *>(ptr); \
    return vec->size;                             \
  }

// void cuthc_sort_ndvector_##N(void* ptr);
#define DECLARE_CUTHC_SORT_NDVECTOR(DIM)          \
  void cuthc_sort_ndvector_##DIM##N(void *ptr)    \
  {                                               \
    auto vec = static_cast<NDVector<DIM> *>(ptr); \
    sort_ndvector(*vec);                          \
  }

// void cuthc_search_ndvector_##N(void* ptr, void *input, bool* result);
#define DECLARE_CUTHC_SEARCH_NDVECTOR(DIM)                                  \
  void cuthc_search_ndvector_##DIM##N(void *ptr, void *input, void *result) \
  {                                                                         \
    auto vec = static_cast<NDVector<DIM> *>(ptr);                           \
    auto search_vec = static_cast<NDVector<DIM> *>(input);                  \
    auto result_vec = static_cast<rmm::device_vector<bool> *>(result);      \
    search_ndvector(*vec, *search_vec, *result_vec);                        \
  }

#define DECLARE_CUTHC_UNIQUE_NDVECTOR(N)        \
  void cuthc_unique_ndvector_##N(void *ptr)     \
  {                                             \
    auto vec = static_cast<NDVector<N> *>(ptr); \
    unique_ndvector(*vec);                      \
  }

#define DECLARE_REMOVE_NDVECTOR(N)                                       \
  void cuthc_remove_ndvector_##N(void *ptr, void *stencil)               \
  {                                                                      \
    auto vec = static_cast<NDVector<N> *>(ptr);                          \
    auto stencil_vec = static_cast<rmm::device_vector<bool> *>(stencil); \
    remove_ndvector_by_stencil(*vec, *stencil_vec);                      \
  }

#define DECLARE_MERGE_NDVECTOR(N)                                    \
  void cuthc_merge_ndvector_##N(void *ptr, void *ptr2, void *result) \
  {                                                                  \
    auto vec = static_cast<NDVector<N> *>(ptr);                      \
    auto vec2 = static_cast<NDVector<N> *>(ptr2);                    \
    auto result_vec = static_cast<NDVector<N> *>(result);            \
    merge_ndvector(*vec, *vec2, *result_vec);                        \
  }

#define DECLARE_CLEAR_NDVECTOR(N)               \
  void cuthc_clear_ndvector_##N(void *ptr)      \
  {                                             \
    auto vec = static_cast<NDVector<N> *>(ptr); \
    clear_ndvector(*vec);                       \
  }

#define DECLARE_INIT_NDVECTOR_INDICES(N)           \
  void *cuthc_init_ndvector_indices_##N(void *ptr) \
  {                                                \
    auto vec = static_cast<NDVector<N> *>(ptr);    \
    return init_ndvector_indices(*vec);            \
  }

#define DECLARE_SORT_NDVECTOR_INDICES(N)                   \
  void cuthc_sort_ndvector_indices_##N(void *ptr, int col) \
  {                                                        \
    auto vec = static_cast<NDVector<N> *>(ptr);            \
    sort_ndvector_indices(*vec, col);                      \
  }

#define CUTHC_MK_NDVECTOR(DIM)        \
  DECLARE_CUTHC_MK_NDVECTOR(DIM)      \
  DECLARE_CUTHC_FREE_NDVECTOR(DIM)    \
  DECLARE_CUTHC_RESIZE_NDVECTOR(DIM)  \
  DECLARE_CUTHC_RAW_PTR_NDVECTOR(DIM) \
  DECLARE_CUTHC_SIZE_NDVECTOR(DIM)    \
  DECLARE_CUTHC_SORT_NDVECTOR(DIM)    \
  DECLARE_CUTHC_SEARCH_NDVECTOR(DIM)  \
  DECLARE_CUTHC_UNIQUE_NDVECTOR(DIM)  \
  DECLARE_CUTHC_GATHER_NDVECTOR(DIM)  \
  DECLARE_REMOVE_NDVECTOR(DIM)        \
  DECLARE_MERGE_NDVECTOR(DIM)         \
  DECLARE_CLEAR_NDVECTOR(DIM)         \
  DECLARE_INIT_NDVECTOR_INDICES(DIM)  \
  DECLARE_SORT_NDVECTOR_INDICES(DIM)

CUTHC_MK_NDVECTOR(1)
CUTHC_MK_NDVECTOR(2)
CUTHC_MK_NDVECTOR(3)
CUTHC_MK_NDVECTOR(4)
CUTHC_MK_NDVECTOR(5)
CUTHC_MK_NDVECTOR(6)
CUTHC_MK_NDVECTOR(7)
CUTHC_MK_NDVECTOR(8)
