
#include <cstddef>
#include <cstdio>
#include <iostream>

#include "ndvector.hpp"

#include "cuthc.h"


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

// CUTHC_MK_NDVECTOR(1)
CUTHC_MK_NDVECTOR(2)
// CUTHC_MK_NDVECTOR(3)
// CUTHC_MK_NDVECTOR(4)
// CUTHC_MK_NDVECTOR(5)
// CUTHC_MK_NDVECTOR(6)
// CUTHC_MK_NDVECTOR(7)
// CUTHC_MK_NDVECTOR(8)
