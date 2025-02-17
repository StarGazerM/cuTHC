
fn main() {
    println!("cargo:rustc-link-search=native=../../cuTHC/build");
    println!("cargo:rustc-link-lib=dylib=cuthc");
}
