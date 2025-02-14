#[link(name = "cuthc", kind = "dylib")]
extern "C" {

    fn cuthc_mk_ndvector_1(arity: i32, size: usize) -> *mut std::ffi::c_void;
    fn cuthc_free_ndvector_1(ptr: *mut std::ffi::c_void);
    fn cuthc_resize_ndvector_1(ptr: *mut std::ffi::c_void, size: usize);
    fn cuthc_raw_ptr_ndvector_1(ptr: *mut std::ffi::c_void, col: *const usize) -> *mut i32;
    fn cuthc_size_ndvector_1(ptr: *mut std::ffi::c_void) -> usize;
    fn cuthc_sort_ndvector_1(ptr: *mut std::ffi::c_void);
    fn cuthc_search_ndvector_1(
        ptr: *mut std::ffi::c_void,
        input: *mut std::ffi::c_void,
        result: *mut std::ffi::c_void,
    );
    fn cuthc_unique_ndvector_1(ptr: *mut std::ffi::c_void);
    fn cuthc_remove_ndvector_1(ptr: *mut std::ffi::c_void, stencil: *mut std::ffi::c_void);

    fn cuthc_mk_ndvector_2(arity: i32, size: usize) -> *mut std::ffi::c_void;
    fn cuthc_free_ndvector_2(ptr: *mut std::ffi::c_void);
    fn cuthc_resize_ndvector_2(ptr: *mut std::ffi::c_void, size: usize);
    fn cuthc_raw_ptr_ndvector_2(ptr: *mut std::ffi::c_void, col: *const usize) -> *mut i32;
    fn cuthc_size_ndvector_2(ptr: *mut std::ffi::c_void) -> usize;
    fn cuthc_sort_ndvector_2(ptr: *mut std::ffi::c_void);
    fn cuthc_search_ndvector_2(
        ptr: *mut std::ffi::c_void,
        input: *mut std::ffi::c_void,
        result: *mut std::ffi::c_void,
    );
    fn cuthc_unique_ndvector_2(ptr: *mut std::ffi::c_void);
    fn cuthc_remove_ndvector_2(ptr: *mut std::ffi::c_void, stencil: *mut std::ffi::c_void);

    fn cuthc_mk_ndvector_3(arity: i32, size: usize) -> *mut std::ffi::c_void;
    fn cuthc_free_ndvector_3(ptr: *mut std::ffi::c_void);
    fn cuthc_resize_ndvector_3(ptr: *mut std::ffi::c_void, size: usize);
    fn cuthc_raw_ptr_ndvector_3(ptr: *mut std::ffi::c_void, col: *const usize) -> *mut i32;
    fn cuthc_size_ndvector_3(ptr: *mut std::ffi::c_void) -> usize;
    fn cuthc_sort_ndvector_3(ptr: *mut std::ffi::c_void);
    fn cuthc_search_ndvector_3(
        ptr: *mut std::ffi::c_void,
        input: *mut std::ffi::c_void,
        result: *mut std::ffi::c_void,
    );
    fn cuthc_unique_ndvector_3(ptr: *mut std::ffi::c_void);
    fn cuthc_remove_ndvector_3(ptr: *mut std::ffi::c_void, stencil: *mut std::ffi::c_void);

    fn cuthc_mk_ndvector_4(arity: i32, size: usize) -> *mut std::ffi::c_void;
    fn cuthc_free_ndvector_4(ptr: *mut std::ffi::c_void);
    fn cuthc_resize_ndvector_4(ptr: *mut std::ffi::c_void, size: usize);
    fn cuthc_raw_ptr_ndvector_4(ptr: *mut std::ffi::c_void, col: *const usize) -> *mut i32;
    fn cuthc_size_ndvector_4(ptr: *mut std::ffi::c_void) -> usize;
    fn cuthc_sort_ndvector_4(ptr: *mut std::ffi::c_void);
    fn cuthc_search_ndvector_4(
        ptr: *mut std::ffi::c_void,
        input: *mut std::ffi::c_void,
        result: *mut std::ffi::c_void,
    );
    fn cuthc_unique_ndvector_4(ptr: *mut std::ffi::c_void);
    fn cuthc_remove_ndvector_4(ptr: *mut std::ffi::c_void, stencil: *mut std::ffi::c_void);

    fn cuthc_mk_ndvector_5(arity: i32, size: usize) -> *mut std::ffi::c_void;
    fn cuthc_free_ndvector_5(ptr: *mut std::ffi::c_void);
    fn cuthc_resize_ndvector_5(ptr: *mut std::ffi::c_void, size: usize);
    fn cuthc_raw_ptr_ndvector_5(ptr: *mut std::ffi::c_void, col: *const usize) -> *mut i32;
    fn cuthc_size_ndvector_5(ptr: *mut std::ffi::c_void) -> usize;
    fn cuthc_sort_ndvector_5(ptr: *mut std::ffi::c_void);
    fn cuthc_search_ndvector_5(
        ptr: *mut std::ffi::c_void,
        input: *mut std::ffi::c_void,
        result: *mut std::ffi::c_void,
    );
    fn cuthc_unique_ndvector_5(ptr: *mut std::ffi::c_void);
    fn cuthc_remove_ndvector_5(ptr: *mut std::ffi::c_void, stencil: *mut std::ffi::c_void);

    fn cuthc_mk_ndvector_6(arity: i32, size: usize) -> *mut std::ffi::c_void;
    fn cuthc_free_ndvector_6(ptr: *mut std::ffi::c_void);
    fn cuthc_resize_ndvector_6(ptr: *mut std::ffi::c_void, size: usize);
    fn cuthc_raw_ptr_ndvector_6(ptr: *mut std::ffi::c_void, col: *const usize) -> *mut i32;
    fn cuthc_size_ndvector_6(ptr: *mut std::ffi::c_void) -> usize;
    fn cuthc_sort_ndvector_6(ptr: *mut std::ffi::c_void);
    fn cuthc_search_ndvector_6(
        ptr: *mut std::ffi::c_void,
        input: *mut std::ffi::c_void,
        result: *mut std::ffi::c_void,
    );
    fn cuthc_unique_ndvector_6(ptr: *mut std::ffi::c_void);
    fn cuthc_remove_ndvector_6(ptr: *mut std::ffi::c_void, stencil: *mut std::ffi::c_void);

    fn cuthc_mk_ndvector_7(arity: i32, size: usize) -> *mut std::ffi::c_void;
    fn cuthc_free_ndvector_7(ptr: *mut std::ffi::c_void);
    fn cuthc_resize_ndvector_7(ptr: *mut std::ffi::c_void, size: usize);
    fn cuthc_raw_ptr_ndvector_7(ptr: *mut std::ffi::c_void, col: *const usize) -> *mut i32;
    fn cuthc_size_ndvector_7(ptr: *mut std::ffi::c_void) -> usize;
    fn cuthc_sort_ndvector_7(ptr: *mut std::ffi::c_void);
    fn cuthc_search_ndvector_7(
        ptr: *mut std::ffi::c_void,
        input: *mut std::ffi::c_void,
        result: *mut std::ffi::c_void,
    );
    fn cuthc_unique_ndvector_7(ptr: *mut std::ffi::c_void);
    fn cuthc_remove_ndvector_7(ptr: *mut std::ffi::c_void, stencil: *mut std::ffi::c_void);

    fn cuthc_mk_ndvector_8(arity: i32, size: usize) -> *mut std::ffi::c_void;
    fn cuthc_free_ndvector_8(ptr: *mut std::ffi::c_void);
    fn cuthc_resize_ndvector_8(ptr: *mut std::ffi::c_void, size: usize);
    fn cuthc_raw_ptr_ndvector_8(ptr: *mut std::ffi::c_void, col: *const usize) -> *mut i32;
    fn cuthc_size_ndvector_8(ptr: *mut std::ffi::c_void) -> usize;
    fn cuthc_sort_ndvector_8(ptr: *mut std::ffi::c_void);
    fn cuthc_search_ndvector_8(
        ptr: *mut std::ffi::c_void,
        input: *mut std::ffi::c_void,
        result: *mut std::ffi::c_void,
    );
    fn cuthc_unique_ndvector_8(ptr: *mut std::ffi::c_void);
    fn cuthc_remove_ndvector_8(ptr: *mut std::ffi::c_void, stencil: *mut std::ffi::c_void);

}

pub struct NdVector<const DIM: usize> {
    raw_vec_ptr: *mut std::ffi::c_void,
    arity: usize,
    size: usize,
}

impl<const DIM: usize> NdVector<DIM> {
    pub fn new(arity: usize, size: usize) -> Self {
        match DIM {
            1 => {
                let raw_vec_ptr = unsafe { cuthc_mk_ndvector_1(arity as i32, size) };
                Self {
                    raw_vec_ptr,
                    arity,
                    size,
                }
            }
            2 => {
                let raw_vec_ptr = unsafe { cuthc_mk_ndvector_2(arity as i32, size) };
                Self {
                    raw_vec_ptr,
                    arity,
                    size,
                }
            }
            3 => {
                let raw_vec_ptr = unsafe { cuthc_mk_ndvector_3(arity as i32, size) };
                Self {
                    raw_vec_ptr,
                    arity,
                    size,
                }
            }
            4 => {
                let raw_vec_ptr = unsafe { cuthc_mk_ndvector_4(arity as i32, size) };
                Self {
                    raw_vec_ptr,
                    arity,
                    size,
                }
            }
            5 => {
                let raw_vec_ptr = unsafe { cuthc_mk_ndvector_5(arity as i32, size) };
                Self {
                    raw_vec_ptr,
                    arity,
                    size,
                }
            }
            6 => {
                let raw_vec_ptr = unsafe { cuthc_mk_ndvector_6(arity as i32, size) };
                Self {
                    raw_vec_ptr,
                    arity,
                    size,
                }
            }
            7 => {
                let raw_vec_ptr = unsafe { cuthc_mk_ndvector_7(arity as i32, size) };
                Self {
                    raw_vec_ptr,
                    arity,
                    size,
                }
            }
            8 => {
                let raw_vec_ptr = unsafe { cuthc_mk_ndvector_8(arity as i32, size) };
                Self {
                    raw_vec_ptr,
                    arity,
                    size,
                }
            }
            _ => panic!("Unsupported dimension"),
        }
    }

    pub fn resize(&mut self, size: usize) {
        match DIM {
            1 => unsafe { cuthc_resize_ndvector_1(self.raw_vec_ptr, size) },
            2 => unsafe { cuthc_resize_ndvector_2(self.raw_vec_ptr, size) },
            3 => unsafe { cuthc_resize_ndvector_3(self.raw_vec_ptr, size) },
            4 => unsafe { cuthc_resize_ndvector_4(self.raw_vec_ptr, size) },
            5 => unsafe { cuthc_resize_ndvector_5(self.raw_vec_ptr, size) },
            6 => unsafe { cuthc_resize_ndvector_6(self.raw_vec_ptr, size) },
            7 => unsafe { cuthc_resize_ndvector_7(self.raw_vec_ptr, size) },
            8 => unsafe { cuthc_resize_ndvector_8(self.raw_vec_ptr, size) },
            _ => panic!("Unsupported dimension"),
        }
        self.size = size;
    }

    pub fn size(&self) -> usize {
        match DIM {
            1 => unsafe { cuthc_size_ndvector_1(self.raw_vec_ptr) },
            2 => unsafe { cuthc_size_ndvector_2(self.raw_vec_ptr) },
            3 => unsafe { cuthc_size_ndvector_3(self.raw_vec_ptr) },
            4 => unsafe { cuthc_size_ndvector_4(self.raw_vec_ptr) },
            5 => unsafe { cuthc_size_ndvector_5(self.raw_vec_ptr) },
            6 => unsafe { cuthc_size_ndvector_6(self.raw_vec_ptr) },
            7 => unsafe { cuthc_size_ndvector_7(self.raw_vec_ptr) },
            8 => unsafe { cuthc_size_ndvector_8(self.raw_vec_ptr) },
            _ => panic!("Unsupported dimension"),
        }
    }

    pub fn raw_data(&self, col: &[usize]) -> *mut i32 {
        match DIM {
            1 => unsafe { cuthc_raw_ptr_ndvector_1(self.raw_vec_ptr, col.as_ptr()) },
            2 => unsafe { cuthc_raw_ptr_ndvector_2(self.raw_vec_ptr, col.as_ptr()) },
            3 => unsafe { cuthc_raw_ptr_ndvector_3(self.raw_vec_ptr, col.as_ptr()) },
            4 => unsafe { cuthc_raw_ptr_ndvector_4(self.raw_vec_ptr, col.as_ptr()) },
            5 => unsafe { cuthc_raw_ptr_ndvector_5(self.raw_vec_ptr, col.as_ptr()) },
            6 => unsafe { cuthc_raw_ptr_ndvector_6(self.raw_vec_ptr, col.as_ptr()) },
            7 => unsafe { cuthc_raw_ptr_ndvector_7(self.raw_vec_ptr, col.as_ptr()) },
            8 => unsafe { cuthc_raw_ptr_ndvector_8(self.raw_vec_ptr, col.as_ptr()) },
            _ => panic!("Unsupported dimension"),
        }
    }

    pub fn sort(&mut self) {
        match DIM {
            1 => unsafe { cuthc_sort_ndvector_1(self.raw_vec_ptr) },
            2 => unsafe { cuthc_sort_ndvector_2(self.raw_vec_ptr) },
            3 => unsafe { cuthc_sort_ndvector_3(self.raw_vec_ptr) },
            4 => unsafe { cuthc_sort_ndvector_4(self.raw_vec_ptr) },
            5 => unsafe { cuthc_sort_ndvector_5(self.raw_vec_ptr) },
            6 => unsafe { cuthc_sort_ndvector_6(self.raw_vec_ptr) },
            7 => unsafe { cuthc_sort_ndvector_7(self.raw_vec_ptr) },
            8 => unsafe { cuthc_sort_ndvector_8(self.raw_vec_ptr) },
            _ => panic!("Unsupported dimension"),
        }
    }

    pub fn search(&mut self, input: &mut NdVector<DIM>) {
        match DIM {
            1 => unsafe {
                cuthc_search_ndvector_1(self.raw_vec_ptr, input.raw_vec_ptr, input.raw_vec_ptr)
            },
            2 => unsafe {
                cuthc_search_ndvector_2(self.raw_vec_ptr, input.raw_vec_ptr, input.raw_vec_ptr)
            },
            3 => unsafe {
                cuthc_search_ndvector_3(self.raw_vec_ptr, input.raw_vec_ptr, input.raw_vec_ptr)
            },
            4 => unsafe {
                cuthc_search_ndvector_4(self.raw_vec_ptr, input.raw_vec_ptr, input.raw_vec_ptr)
            },
            5 => unsafe {
                cuthc_search_ndvector_5(self.raw_vec_ptr, input.raw_vec_ptr, input.raw_vec_ptr)
            },
            6 => unsafe {
                cuthc_search_ndvector_6(self.raw_vec_ptr, input.raw_vec_ptr, input.raw_vec_ptr)
            },
            7 => unsafe {
                cuthc_search_ndvector_7(self.raw_vec_ptr, input.raw_vec_ptr, input.raw_vec_ptr)
            },
            8 => unsafe {
                cuthc_search_ndvector_8(self.raw_vec_ptr, input.raw_vec_ptr, input.raw_vec_ptr)
            },
            _ => panic!("Unsupported dimension"),
        }
    }

    pub fn unique(&mut self) {
        match DIM {
            1 => unsafe { cuthc_unique_ndvector_1(self.raw_vec_ptr) },
            2 => unsafe { cuthc_unique_ndvector_2(self.raw_vec_ptr) },
            3 => unsafe { cuthc_unique_ndvector_3(self.raw_vec_ptr) },
            4 => unsafe { cuthc_unique_ndvector_4(self.raw_vec_ptr) },
            5 => unsafe { cuthc_unique_ndvector_5(self.raw_vec_ptr) },
            6 => unsafe { cuthc_unique_ndvector_6(self.raw_vec_ptr) },
            7 => unsafe { cuthc_unique_ndvector_7(self.raw_vec_ptr) },
            8 => unsafe { cuthc_unique_ndvector_8(self.raw_vec_ptr) },
            _ => panic!("Unsupported dimension"),
        }
    }

    pub fn remove(&mut self, stencil: &mut NdVector<DIM>) {
        match DIM {
            1 => unsafe { cuthc_remove_ndvector_1(self.raw_vec_ptr, stencil.raw_vec_ptr) },
            2 => unsafe { cuthc_remove_ndvector_2(self.raw_vec_ptr, stencil.raw_vec_ptr) },
            3 => unsafe { cuthc_remove_ndvector_3(self.raw_vec_ptr, stencil.raw_vec_ptr) },
            4 => unsafe { cuthc_remove_ndvector_4(self.raw_vec_ptr, stencil.raw_vec_ptr) },
            5 => unsafe { cuthc_remove_ndvector_5(self.raw_vec_ptr, stencil.raw_vec_ptr) },
            6 => unsafe { cuthc_remove_ndvector_6(self.raw_vec_ptr, stencil.raw_vec_ptr) },
            7 => unsafe { cuthc_remove_ndvector_7(self.raw_vec_ptr, stencil.raw_vec_ptr) },
            8 => unsafe { cuthc_remove_ndvector_8(self.raw_vec_ptr, stencil.raw_vec_ptr) },
            _ => panic!("Unsupported dimension"),
        }
    }
}

impl<const DIM: usize> Drop for NdVector<DIM> {
    fn drop(&mut self) {
        match DIM {
            1 => unsafe { cuthc_free_ndvector_1(self.raw_vec_ptr) },
            2 => unsafe { cuthc_free_ndvector_2(self.raw_vec_ptr) },
            3 => unsafe { cuthc_free_ndvector_3(self.raw_vec_ptr) },
            4 => unsafe { cuthc_free_ndvector_4(self.raw_vec_ptr) },
            5 => unsafe { cuthc_free_ndvector_5(self.raw_vec_ptr) },
            6 => unsafe { cuthc_free_ndvector_6(self.raw_vec_ptr) },
            7 => unsafe { cuthc_free_ndvector_7(self.raw_vec_ptr) },
            8 => unsafe { cuthc_free_ndvector_8(self.raw_vec_ptr) },
            _ => panic!("Unsupported dimension"),
        }
    }
}
