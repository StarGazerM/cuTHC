use cuthc_macro::{gen_cuthc_nvec_call_n, gen_cuthc_nvec_n};

#[link(name = "cuthc", kind = "dylib")]
extern "C" {
    gen_cuthc_nvec_n!{8=>
        fn cuthc_mk_ndvector(arity: i32, size: usize) -> *mut std::ffi::c_void;
        fn cuthc_free_ndvector(ptr: *mut std::ffi::c_void);
        fn cuthc_resize_ndvector(ptr: *mut std::ffi::c_void, size: usize);
        fn cuthc_raw_ptr_ndvector(ptr: *mut std::ffi::c_void, col: usize) -> *mut i32;
        fn cuthc_size_ndvector(ptr: *mut std::ffi::c_void) -> usize;
        fn cuthc_sort_ndvector(ptr: *mut std::ffi::c_void);
        fn cuthc_search_ndvector(
            ptr: *mut std::ffi::c_void,
            input: *mut std::ffi::c_void,
            result: *mut std::ffi::c_void,
        );
        fn cuthc_unique_ndvector(ptr: *mut std::ffi::c_void);
        fn cuthc_remove_ndvector(ptr: *mut std::ffi::c_void, stencil: *mut std::ffi::c_void);
        fn cuthc_merge_ndvector(
            ptr: *mut std:: ffi::c_void,
            input: *mut std::ffi::c_void,
            result: *mut std::ffi::c_void,
        );
        fn cuthc_init_ndvector_indices(ptr: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
        fn cuthc_sort_ndvector_indices(ptr: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
        fn cuthc_gather_ndvector(
            ptr: *mut std::ffi::c_void,
            indices: *mut std::ffi::c_void,
            result: *mut std::ffi::c_void,
        );
    }
}


pub struct NdVector<const DIM: usize> {
    raw_vec_ptr: *mut std::ffi::c_void,
    arity: usize,
    size: usize,
    indices: *mut std::ffi::c_void,
}

impl<const DIM: usize> NdVector<DIM> {
    pub fn new(arity: usize, size: usize) -> Self {
        gen_cuthc_nvec_call_n!{8=>
            let raw_vec_ptr = cuthc_mk_ndvector(arity as i32, size) ;
            Self {
                raw_vec_ptr,
                arity,
                size,
                indices: std::ptr::null_mut(),
            }
        }
    }

    pub fn resize(&mut self, size: usize) {
        self.size = size;
        gen_cuthc_nvec_call_n!{8=>
            cuthc_resize_ndvector(self.raw_vec_ptr, size);
        }
        
    }

    pub fn size(&self) -> usize {
        gen_cuthc_nvec_call_n!{8=>
            cuthc_size_ndvector(self.raw_vec_ptr)
        }
    }

    pub fn raw_data(&self, col: usize) -> *mut i32 {
        gen_cuthc_nvec_call_n!{8=>
            cuthc_raw_ptr_ndvector(self.raw_vec_ptr, col)
        }
    }

    pub fn sort(&mut self) {
        gen_cuthc_nvec_call_n!{8=>
            cuthc_sort_ndvector(self.raw_vec_ptr)
        }
    }

    pub fn search(&mut self, input: &mut NdVector<DIM>) {
        gen_cuthc_nvec_call_n!{8=>
            cuthc_search_ndvector(self.raw_vec_ptr, input.raw_vec_ptr, input.raw_vec_ptr)
        }
    }

    pub fn unique(&mut self) {
        gen_cuthc_nvec_call_n!{8=>
            cuthc_unique_ndvector(self.raw_vec_ptr)
        }
    }

    pub fn remove(&mut self, stencil: &mut NdVector<DIM>) {
        gen_cuthc_nvec_call_n!{8=>
            cuthc_remove_ndvector(self.raw_vec_ptr, stencil.raw_vec_ptr)
        }
    }

    pub fn clear(&mut self) {
        self.resize(0);
    }

    pub fn indexed(&self) -> bool {
        !self.indices.is_null()
    }

    pub fn init_indices(&mut self) {
        gen_cuthc_nvec_call_n! {8=>
            let ind_ptr = cuthc_init_ndvector_indices(self.raw_vec_ptr);
            self.indices = ind_ptr;
        }
    }

    pub fn sort_indices(&mut self) {
        gen_cuthc_nvec_call_n! {8=>
            let ind_ptr = cuthc_sort_ndvector_indices(self.indices);
            self.indices = ind_ptr;
        }
    }

    pub fn gather(&mut self, indices: *mut std::ffi::c_void, result: &mut NdVector<DIM>) {
        gen_cuthc_nvec_call_n! {8=>
            cuthc_gather_ndvector(self.raw_vec_ptr, indices, result.raw_vec_ptr)
        }
    }

}

pub fn merge<const DIM: usize>(
    input: &mut NdVector<DIM>,
    result: &mut NdVector<DIM>,
) -> NdVector<DIM> {
    let new_vec = NdVector::new(input.arity, input.size + result.size);
    gen_cuthc_nvec_call_n!{8=>
        cuthc_merge_ndvector(new_vec.raw_vec_ptr, input.raw_vec_ptr, result.raw_vec_ptr)
    }
    new_vec
}

impl<const DIM: usize> Drop for NdVector<DIM> {
    fn drop(&mut self) {
        gen_cuthc_nvec_call_n!{8=>
            cuthc_free_ndvector(self.raw_vec_ptr)
        }
    }
}
