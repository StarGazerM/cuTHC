

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

#include "../include/ndvector.hpp"
#include "../include/read.cuh"

#include <chrono>
#include "../include/cuthc.h"

// default size of random vectors as 1GB
#define DEFAULT_SIZE 512 * 1024 * 1024
#define DEFAULT_SEARCH_SIZE 1024 * 1024
#define ARITY 3

// generate a random in32_t device vector
void random_int_device_vector(size_t size, rmm::device_vector<int> &vec)
{
  vec.resize(size);
  thrust::host_vector<int> host_vec(size);
  for (size_t i = 0; i < size; i++)
  {
    host_vec[i] = rand() % DEFAULT_SEARCH_SIZE;
  }
  cudaMemcpy(vec.data().get(), host_vec.data(), size * sizeof(int),
             cudaMemcpyHostToDevice);
}

void random_int_device_vector(size_t size, rmm::device_vector<int> &vec, int val)
{
  vec.resize(size);
  thrust::host_vector<int> host_vec(size);
  for (size_t i = 0; i < size; i++)
  {
    host_vec[i] = rand() % val;
  }
  cudaMemcpy(vec.data().get(), host_vec.data(), size * sizeof(int),
             cudaMemcpyHostToDevice);
}

void test_binary_find(NDVector<ARITY> &vec, NDVector<ARITY> &search_vec)
{
  rmm::device_vector<bool> result;
  result.resize(search_vec.get_size());
  auto total_time = 0;
  for (int i = 0; i < 10; i++)
  {
    auto start = std::chrono::high_resolution_clock::now();
    search_ndvector(vec, search_vec, result);
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    thrust::fill(result.begin(), result.end(), false);
    total_time += duration.count();
    // std::cout << "Binary search took " << duration.count() << " milliseconds" << std::endl;
  }
  std::cout << "Average binary search time: " << total_time / 10 << " milliseconds" << std::endl;
}

void test_find_kernal_print() {
  NDVector<2> ndvec(20);
  NDVector<2> search_ndvec(10);
  for (int i = 0; i < 2; i++)
  {
    random_int_device_vector(20, ndvec.vecs[i], 10);
  }
  for (int i = 0; i < 2; i++)
  {
    random_int_device_vector(10, search_ndvec.vecs[i], 10);
  }
  
  sort_ndvector(ndvec);
  sort_ndvector(search_ndvec);
  rmm::device_vector<bool> result;
  result.resize(search_ndvec.get_size());
  search_ndvector_kernel_wrapper(ndvec, search_ndvec, result);
  cudaDeviceSynchronize();
  // print
  ndvec.print_row_before(20);
  
  std::cout << "wwwwwwwwww" << std::endl;
  search_ndvec.print_row_before(10);
  std::cout << "wwwwwwwwww" << std::endl;
  for (int i = 0; i < search_ndvec.get_size(); i++)
  {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;
}

void test_binary_find_kernal(NDVector<ARITY> &vec, NDVector<ARITY> &search_vec)
{
  rmm::device_vector<bool> result;
  result.resize(search_vec.get_size());
  thrust::fill(result.begin(), result.end(), false);
  auto total_time = 0;
  for (int i = 0; i < 10; i++)
  {
    auto start = std::chrono::high_resolution_clock::now();
    cudaDeviceSynchronize();
    search_ndvector_kernel_wrapper(vec, search_vec, result);
    // synchronize
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    // std::cout << "Binary search took " << duration.count() << " milliseconds" << std::endl;
    total_time += duration.count();
  }
  std::cout << "Average binary search time: " << total_time / 10 << " milliseconds" << std::endl;
}

void test_join_ndev() {
  char *filename = "data.txt";
  NDVector<2> edge;
  read_data<2>(filename, edge);

  NDVector<2> path;

  copy_ndvector(edge, path);
  path.lexical_reorder({1, 0});
  IndexedNDVector<2, 1> indexed_edge(edge, {0, 1});

  NDVector<2> path_new;
  compute_equi_join<2,2,1,2>(path, indexed_edge, path_new);
  
  // print join size
  // 
}

int main()
{
  NDVector<ARITY> ndvec(DEFAULT_SIZE);
  NDVector<ARITY> search_ndvec(DEFAULT_SEARCH_SIZE);
  // generate random vectors
  for (int i = 0; i < ARITY; i++)
  {
    random_int_device_vector(DEFAULT_SIZE, ndvec.vecs[i]);
  }

  for (int i = 0; i < ARITY; i++)
  {
    random_int_device_vector(DEFAULT_SEARCH_SIZE, search_ndvec.vecs[i]);
  }
  auto sort_time = 0;
  auto before = std::chrono::high_resolution_clock::now();
  sort_ndvector(ndvec);
  sort_ndvector(search_ndvec);
  auto after = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(after - before);
  std::cout << "Sorting took " << duration.count() << " milliseconds" << std::endl;

  test_binary_find(ndvec, search_ndvec);
  test_binary_find_kernal(ndvec, search_ndvec);

  // test_find_kernal_print();

  return 0;
}
