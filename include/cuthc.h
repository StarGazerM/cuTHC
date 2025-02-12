

#pragma once

extern "C" {

    // initialize the RMM allocator
    void cuthc_rmm_pool_init();

    // create a pinned memory using host vector
    void* cuthc_mk_int_pinned_vec(size_t size);
    void cuthc_free_int_pinned_vec(void* ptr);
    // resize a pinned memory using host vector
    void cuthc_resize_int_pinned_vec(void* ptr, size_t size);
    int* cuthc_raw_ptr_int_pinned_vec(void* ptr);
    size_t cuthc_size_int_pinned_vec(void* ptr);
    void cuthc_set_int_pinned_vec(void* ptr, size_t pos, int value);
    int cuthc_get_int_pinned_vec(void* ptr, size_t pos);

    // device vector
    void* cuthc_mk_int_device_vec(size_t size);
    void cuthc_free_int_device_vec(void* ptr);
    void cuthc_resize_int_device_vec(void* ptr, size_t size);
    int* cuthc_raw_ptr_int_device_vec(void* ptr);
    size_t cuthc_size_int_device_vec(void* ptr);
    void cuthc_set_int_device_vec(void* ptr, size_t pos, int value);
    int cuthc_get_int_device_vec(void* ptr, size_t pos);
}
