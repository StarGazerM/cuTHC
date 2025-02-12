
fn main() {
    println!("cargo:rustc-link-search=native=/home/stargazermiao/workspace/cuTHC/build");
    println!("cargo:rustc-link-lib=dylib=cuthc");
}
