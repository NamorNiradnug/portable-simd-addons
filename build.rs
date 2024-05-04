fn main() {
    #[cfg(feature = "vectorclass_bench")]
    {
        println!("cargo:rerun-if-changed=benches/vclbench.cpp");
        cxx_build::bridge("benches/math.rs")
            .compiler("clang++")
            .file("benches/vclbench.cpp")
            .flag("-march=native")
            .flag("-O3")
            .flag("-fveclib=libmvec")
            .std("c++17")
            .compile("vclbench");
    }
}
