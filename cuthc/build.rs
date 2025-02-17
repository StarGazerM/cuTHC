
extern crate cmake;
use cmake::Config;


fn main() {
    let dst = Config::new("libcuthc")
        .build();
    println!("cargo:rustc-link-lib=dylib=cuthc");
    println!("cargo:rustc-link-search=native={}/build", dst.display());
}
