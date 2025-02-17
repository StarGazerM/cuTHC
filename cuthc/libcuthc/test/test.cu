

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

#include <chrono>
// #include "cuthc.h"

// default size of random vectors as 1GB
#define DEFAULT_SIZE 512 * 1024 * 1024
#define DEFAULT_SEARCH_SIZE 1024 * 1024
#define ARITY 4

// a N-dimensional vector, each dimension is a device vector
template <int N> class NDVector {

public:
  size_t size;
  std::array<rmm::device_vector<int>, N> vecs;

  NDVector(size_t size) : size(size) {
    for (int i = 0; i < N; i++) {
      vecs[i].resize(size);
    }
  }

  void host_copy_column(int col, thrust::host_vector<int> &host_vec) {
    cudaMemcpy(host_vec.data(), vecs[col].data().get(), size * sizeof(int),
               cudaMemcpyDeviceToHost);
  }

  void print_row_before(int row_num) {
    for (int i = 0; i < row_num; i++) {
      for (int j = 0; j < N; j++) {
        std::cout << vecs[j][i] << " ";
      }
      std::cout << std::endl;
    }
  }
};

// generate a random in32_t device vector
void random_int_device_vector(size_t size, rmm::device_vector<int> &vec) {
  vec.resize(size);
  thrust::host_vector<int> host_vec(size);
  for (size_t i = 0; i < size; i++) {
    host_vec[i] = rand() % DEFAULT_SEARCH_SIZE;
  }
  cudaMemcpy(vec.data().get(), host_vec.data(), size * sizeof(int),
             cudaMemcpyHostToDevice);
}

void sort_vector(rmm::device_vector<int> &vec) {
  thrust::sort(vec.begin(), vec.end());
}


#include <utility> // for index_sequence

template <int N> auto zip_iterator(NDVector<N> &ndvec) {
  return thrust::make_zip_iterator(
      [&]<size_t... Is>(std::index_sequence<Is...>) {
        return thrust::make_tuple(ndvec.vecs[Is].begin()...);
      }(std::make_index_sequence<N>{}));
}

// sort a N-dimensional vector based lexicographically
template <int N> void sort_ndvector(NDVector<N> &ndvec) {
  // rmm::device_vector<int> indices(ndvec.size);
  // rmm::device_vector<int> temp(ndvec.size);
  // // initialize indices
  // thrust::sequence(thrust::device, indices.begin(), indices.end());
  // for (int i = N - 1; i >= 0; i--) {
  //   // sort the i-th dimension
  //   thrust::gather(thrust::device, ndvec.vecs[i].begin(), ndvec.vecs[i].end(),
  //                  indices.begin(), temp.begin());
  //   thrust::sort_by_key(thrust::device, temp.begin(), temp.end(),
  //                       indices.begin());
  // }
  // for (int i = 0; i < N; i++) {
  //   // sort the i-th dimension
  //   thrust::gather(thrust::device, indices.begin(), indices.end(),
  //                  ndvec.vecs[i].begin(), temp.begin());
  //   // swap the sorted vector back to the original vector
  //   ndvec.vecs[i].swap(temp);
  // }

  auto z_begin = zip_iterator(ndvec);
  auto z_end = z_begin + ndvec.size;
  thrust::sort(rmm::exec_policy(), z_begin, z_end);
}


template <int N>
void search_ndvector(NDVector<N> &ndvec, NDVector<N> &search_ndvec,
                     rmm::device_vector<bool> &result) {
  auto z_begin = zip_iterator(ndvec);
  auto z_end = z_begin + ndvec.size;
  auto search_begin = zip_iterator(search_ndvec);
  auto search_end = search_begin + search_ndvec.size;

  thrust::binary_search(rmm::exec_policy(), z_begin, z_end, search_begin,
                        search_end, result.begin());
}

// search for a value in a sorted vector
void search_vector(rmm::device_vector<int> &vec,
                   rmm::device_vector<int> &search_vec,
                   rmm::device_vector<bool> &result) {
  thrust::binary_search(thrust::device, vec.begin(), vec.end(),
                        search_vec.begin(), search_vec.end(), result.begin());
}

int main() {

  NDVector<ARITY> ndvec(DEFAULT_SIZE);
  NDVector<ARITY> search_ndvec(DEFAULT_SEARCH_SIZE);
  for (int i = 0; i < ARITY; i++) {
    random_int_device_vector(DEFAULT_SIZE, ndvec.vecs[i]);
  }
  ndvec.print_row_before(100);
  auto start = std::chrono::high_resolution_clock::now();
  sort_ndvector(ndvec);
  // ndvec.print_row_before(100);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  printf("sort time: %ld ms\n", duration.count());
  for (int i = 0; i < ARITY; i++) {
    random_int_device_vector(DEFAULT_SEARCH_SIZE, search_ndvec.vecs[i]);
  }
  sort_ndvector(search_ndvec);

  rmm::device_vector<bool> result(DEFAULT_SEARCH_SIZE);
  start = std::chrono::high_resolution_clock::now();
  search_ndvector(ndvec, search_ndvec, result);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  printf("search time: %ld ms\n", duration.count());
  // print first 100 rows
  
  return 0;
}
