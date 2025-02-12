use std::ops::Index;



#[link(name = "cuthc", kind = "dylib")]
extern "C" {

    fn cuthc_rmm_pool_init();

    // create a pinned memory using host vector
    // void* cuthc_mk_int_pinned_vec(size_t size);
    // void cuthc_free_int_pinned_vec(void* ptr);
    // // resize a pinned memory using host vector
    // void cuthc_resize_int_pinned_vec(void* ptr, size_t size);
    // int* cuthc_raw_ptr_int_pinned_vec(void* ptr);
    // size_t cuthc_size_int_pinned_vec(void* ptr);
    // void cuthc_set_int_pinned_vec(void* ptr, size_t pos, int value);

    // rust wrapper for above functions
    // create a pinned memory using host vector
    fn cuthc_mk_int_pinned_vec(size: usize) -> *mut std::ffi::c_void;
    fn cuthc_free_int_pinned_vec(ptr: *mut std::ffi::c_void);
    // resize a pinned memory using host vector
    fn cuthc_resize_int_pinned_vec(ptr: *mut std::ffi::c_void, size: usize);
    fn cuthc_raw_ptr_int_pinned_vec(ptr: *mut std::ffi::c_void) -> *mut i32;
    fn cuthc_size_int_pinned_vec(ptr: *mut std::ffi::c_void) -> usize;
    fn cuthc_set_int_pinned_vec(ptr: *mut std::ffi::c_void, pos: usize, value: i32);
    fn cuthc_get_int_pinned_vec(ptr: *mut std::ffi::c_void, pos: usize) -> i32;
    

    // // device vector
    // void* cuthc_mk_int_device_vec(size_t size);
    // void cuthc_free_int_device_vec(void* ptr);
    // void cuthc_resize_int_device_vec(void* ptr, size_t size);
    // int* cuthc_raw_ptr_int_device_vec(void* ptr);
    // size_t cuthc_size_int_device_vec(void* ptr);
    fn cuthc_mk_int_device_vec(size: usize) -> *mut std::ffi::c_void;
    fn cuthc_free_int_device_vec(ptr: *mut std::ffi::c_void);
    fn cuthc_resize_int_device_vec(ptr: *mut std::ffi::c_void, size: usize);
    fn cuthc_raw_ptr_int_device_vec(ptr: *mut std::ffi::c_void) -> *mut i32;
    fn cuthc_size_int_device_vec(ptr: *mut std::ffi::c_void) -> usize;
    fn cuthc_set_int_device_vec(ptr: *mut std::ffi::c_void, pos: usize, value: i32);
    fn cuthc_get_int_device_vec(ptr: *mut std::ffi::c_void, pos: usize) -> i32;
}

// use cudarc::runtime::sys::CUdeviceptr;

pub fn memory_pool_init() {
    unsafe {
        cuthc_rmm_pool_init();
    }
}

pub struct PinnedVec {
    raw_vec_ptr: *mut std::ffi::c_void,
    raw_data_ptr: *mut i32,

    internal: Vec<i32>,
}

impl PinnedVec {
    pub fn new(size: usize) -> Self {
        let raw_vec_ptr = unsafe { cuthc_mk_int_pinned_vec(size) };
        let raw_data_ptr = unsafe { cuthc_raw_ptr_int_pinned_vec(raw_vec_ptr) };
        Self {
            raw_vec_ptr,
            raw_data_ptr,
            internal: Vec::with_capacity(size),
        }
    }

    pub fn resize(&mut self, size: usize) {
        unsafe {
            cuthc_resize_int_pinned_vec(self.raw_vec_ptr, size);
        }
    }

    pub fn size(&self) -> usize {
        unsafe {
            cuthc_size_int_pinned_vec(self.raw_vec_ptr)
        }
    }

    pub fn raw_data(&self) -> *mut i32 {
        self.raw_data_ptr
    }

    pub fn as_slice(&self) -> &[i32] {
        unsafe {
            std::slice::from_raw_parts(self.raw_data_ptr, self.size())
        }
    }

    // modify the data
    pub fn set(&mut self, pos: usize, value: i32) {
        unsafe {
            cuthc_set_int_pinned_vec(self.raw_vec_ptr, pos, value);
        }
    }

    pub fn get(&self, pos: usize) -> i32 {
        unsafe {
            cuthc_get_int_pinned_vec(self.raw_vec_ptr, pos)
        }
    }
}

// impl Drop for PinnedVec {
//     fn drop(&mut self) {
//         unsafe {
//             cuthc_free_int_pinned_vec(self.raw_vec_ptr);
//         }
//     }
// }

pub struct DeviceVec {
    raw_vec_ptr: *mut std::ffi::c_void,
    raw_data_ptr: *mut i32,
}

impl DeviceVec {
    pub fn new(size: usize) -> Self {
        let raw_vec_ptr = unsafe { cuthc_mk_int_device_vec(size) };
        let raw_data_ptr = unsafe { cuthc_raw_ptr_int_device_vec(raw_vec_ptr) };
        Self {
            raw_vec_ptr,
            raw_data_ptr,
        }
    }

    pub fn resize(&mut self, size: usize) {
        unsafe {
            cuthc_resize_int_device_vec(self.raw_vec_ptr, size);
        }
    }

    pub fn size(&self) -> usize {
        unsafe {
            cuthc_size_int_device_vec(self.raw_vec_ptr)
        }
    }

    pub fn raw_data(&self) -> *mut i32 {
        self.raw_data_ptr
    }

    pub fn set(&mut self, pos: usize, value: i32) {
        unsafe {
            cuthc_set_int_device_vec(self.raw_vec_ptr, pos, value);
        }
    }

    pub fn get(&self, pos: usize) -> i32 {
        unsafe {
            cuthc_get_int_device_vec(self.raw_vec_ptr, pos)
        }
    }
}

impl Drop for DeviceVec {
    fn drop(&mut self) {
        unsafe {
            cuthc_free_int_device_vec(self.raw_vec_ptr);
        }
    }
}
