

#pragma once

#define DECL_NDVECTOR(N) \
    void* cuthc_mk_ndvector_##N(int n, size_t size); \
    void cuthc_free_ndvector_##N(void* ptr); \
    void cuthc_resize_ndvector_##N(void* ptr, size_t size); \
    int* cuthc_raw_ptr_ndvector_##N(void* ptr, size_t* col); \
    size_t cuthc_size_ndvector_##N(void* ptr); \
    void cuthc_sort_ndvector_##N(void* ptr); \
    void cuthc_search_ndvector_##N(void* ptr, void *input, void* result); \
    void cuthc_remove_ndvector_##N(void* ptr, void* stencil, void* result); \
    void cuthc_unique_ndvector_##N(void* ptr); \
    void cuthc_merge_ndvector_##N(void* ptr1, void* ptr2, void* result); \
    void cuthc_clear_ndvector_##N(void* ptr);

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

    // device bool vector
    void* cuthc_mk_bool_device_vec(size_t size);
    void cuthc_free_bool_device_vec(void* ptr);
    void cuthc_resize_bool_device_vec(void* ptr, size_t size);
    bool* cuthc_raw_ptr_bool_device_vec(void* ptr);
    size_t cuthc_size_bool_device_vec(void* ptr);
    void cuthc_set_bool_device_vec(void* ptr, size_t pos, bool value);
    bool cuthc_get_bool_device_vec(void* ptr, size_t pos);
    void cuthc_set_bool_device_vec_all(void* ptr, bool value);
    int cuthc_num_sm();

    // ndarray api
    DECL_NDVECTOR(1)
    DECL_NDVECTOR(2)
    DECL_NDVECTOR(3)
    DECL_NDVECTOR(4)
    DECL_NDVECTOR(5)
    DECL_NDVECTOR(6)
    DECL_NDVECTOR(7)
    DECL_NDVECTOR(8)

}
