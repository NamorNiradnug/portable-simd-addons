fn main() {
    #[cfg(feature = "vectorclass_bench")]
    {
        println!("cargo:rerun-if-changed=benches/cpp/vclbench.cpp");
        cxx_build::bridge("benches/cpp/vclbench.rs")
            .compiler("clang++")
            .file("benches/cpp/vclbench.cpp")
            .flag("-march=native")
            .flag("-Ofast")
            .std("c++17")
            .compile("vclbench");
    }

    #[cfg(feature = "libmvec_bench")]
    {
        println!("cargo:rerun-if-changed=benches/cpp/libmvecbench.cpp");
        cxx_build::bridge("benches/cpp/libmvecbench.rs")
            .compiler("g++")
            .file("benches/cpp/libmvecbench.cpp")
            .flag("-march=native")
            .flag("-Ofast")
            .std("c++17")
            .compile("libmvecbench");
    }
}
