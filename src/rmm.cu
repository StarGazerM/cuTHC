
#include "rmm/mr/device/per_device_resource.hpp"
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <thrust/host_vector.h>
#include <rmm/exec_policy.hpp>
#include <rmm/device_vector.hpp>

#include "cuthc.h"

rmm::mr::cuda_memory_resource cuda_mr{};
rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr, 4 * 256 * 1024};

void cuthc_rmm_pool_init() {
  rmm::mr::set_current_device_resource(&pool_mr);
}

void* cuthc_mk_int_pinned_vec(size_t size) {
    auto host_vec = new thrust::host_vector<int>(size);
    return host_vec;
}

void cuthc_free_int_pinned_vec(void* ptr) {
    auto host_vec = static_cast<thrust::host_vector<int>*>(ptr);
    delete host_vec;
}

void cuthc_resize_int_pinned_vec(void* ptr, size_t size) {
    auto host_vec = static_cast<thrust::host_vector<int>*>(ptr);
    host_vec->resize(size);
}

int* cuthc_raw_ptr_int_pinned_vec(void* ptr) {
    auto host_vec = static_cast<thrust::host_vector<int>*>(ptr);
    return host_vec->data();
}

size_t cuthc_size_int_pinned_vec(void* ptr) {
    auto host_vec = static_cast<thrust::host_vector<int>*>(ptr);
    return host_vec->size();
}

void cuthc_set_int_pinned_vec(void* ptr, size_t index, int value) {
    auto host_vec = static_cast<thrust::host_vector<int>*>(ptr);
    (*host_vec)[index] = value;
}

int cuthc_get_int_pinned_vec(void* ptr, size_t index) {
    auto host_vec = static_cast<thrust::host_vector<int>*>(ptr);
    return (*host_vec)[index];
}

void* cuthc_mk_int_device_vec(size_t size) {
    auto device_vec = new rmm::device_vector<int>(size);
    return device_vec;
}

void cuthc_free_int_device_vec(void* ptr) {
    auto device_vec = static_cast<rmm::device_vector<int>*>(ptr);
    delete device_vec;
}

void cuthc_resize_int_device_vec(void* ptr, size_t size) {
    auto device_vec = static_cast<rmm::device_vector<int>*>(ptr);
    device_vec->resize(size);
}

int* cuthc_raw_ptr_int_device_vec(void* ptr) {
    auto device_vec = static_cast<rmm::device_vector<int>*>(ptr);
    return device_vec->data().get();
}

size_t cuthc_size_int_device_vec(void* ptr) {
    auto device_vec = static_cast<rmm::device_vector<int>*>(ptr);
    return device_vec->size();
}

void cuthc_set_int_device_vec(void* ptr, size_t index, int value) {
    auto device_vec = static_cast<rmm::device_vector<int>*>(ptr);
    (*device_vec)[index] = value;
}

int cuthc_get_int_device_vec(void* ptr, size_t index) {
    auto device_vec = static_cast<rmm::device_vector<int>*>(ptr);
    return (*device_vec)[index];
}


void* cuthc_mk_bool_device_vec(size_t size) {
    auto device_vec = new rmm::device_vector<bool>(size);
    return device_vec;
}
void cuthc_free_bool_device_vec(void* ptr) {
    auto device_vec = static_cast<rmm::device_vector<bool>*>(ptr);
    delete device_vec;
}
void cuthc_resize_bool_device_vec(void* ptr, size_t size) {
    auto device_vec = static_cast<rmm::device_vector<bool>*>(ptr);
    device_vec->resize(size);
}
bool* cuthc_raw_ptr_bool_device_vec(void* ptr) {
    auto device_vec = static_cast<rmm::device_vector<bool>*>(ptr);
    return device_vec->data().get();
}
size_t cuthc_size_bool_device_vec(void* ptr) {
    auto device_vec = static_cast<rmm::device_vector<bool>*>(ptr);
    return device_vec->size();
}
void cuthc_set_bool_device_vec(void* ptr, size_t pos, bool value) {
    auto device_vec = static_cast<rmm::device_vector<bool>*>(ptr);
    (*device_vec)[pos] = value;
}
bool cuthc_get_bool_device_vec(void* ptr, size_t pos) {
    auto device_vec = static_cast<rmm::device_vector<bool>*>(ptr);
    return (*device_vec)[pos];
}

void cuthc_set_bool_device_vec_all(void* ptr, bool value) {
    auto device_vec = static_cast<rmm::device_vector<bool>*>(ptr);
    thrust::fill(rmm::exec_policy(), device_vec->begin(), device_vec->end(), value);
}
