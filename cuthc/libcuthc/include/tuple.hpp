
#pragma once

#include <utility> // for index_sequence
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

#define EMPTY_ENTRY INT32_MAX

template <int N>
__device__ int compare_tuple(int *a, int *b)
{
#pragma unroll
  for (int i = 0; i < N; i++)
  {
    if (a[i] < b[i])
    {
      return -1;
    }
    else if (a[i] > b[i])
    {
      return 1;
    }
  }
  return 0;
}

// template <int N>
// using ThrustTupleType = decltype([&]<size_t... Is>(
//                                      std::index_sequence<Is...>)
//                                  { return thrust::make_tuple(ndvec.vecs[Is].begin()...); }(std::make_index_sequence<N>{}));

// comparator functor for zipped tuple has size N
template <int N>
struct TupleComparator
{
  __device__ __host__ bool operator()(const auto &lhs, const auto &rhs)
  {
    return comp_impl(lhs, rhs, std::make_index_sequence<N>{});
  }

  template <size_t... Is>
  __device__ __host__ bool comp_impl(const auto &lhs, const auto &rhs, std::index_sequence<Is...>)
  {
    bool less = false;
    bool result = false;
    ((less = thrust::get<Is>(lhs) < thrust::get<Is>(rhs),
      result = result || (less && (... && (thrust::get<Is>(lhs) == thrust::get<Is>(rhs)))),
      result = result || less),
     ...);
    return result;
  }
};

template <int N>
struct TupleNotEqualTo
{
  __device__ __host__ bool operator()(const auto &lhs, const auto &rhs)
  {
    return neq_impl(lhs, rhs, std::make_index_sequence<N>{});
  }

  template <size_t... Is>
  __device__ __host__ bool neq_impl(const auto &lhs, const auto &rhs, std::index_sequence<Is...>)
  {
    bool neq = false;
    bool result = false;
    ((neq = thrust::get<Is>(lhs) != thrust::get<Is>(rhs),
      result = result || neq),
     ...);
    return result;
  }
};
