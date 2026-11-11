# Installed public-header consumer

Run this separately from the library's source build, inside a validation
allocation with its MPI and library dependencies available:

```bash
cmake --build /path/to/build --target ddla_lib --parallel 4
/path/to/LibDDLA/tests/install_consumer/check_install.sh \
  /path/to/build /tmp/ddla-installed-consumer-new
```

The scratch directory must not exist. The script installs the existing build
into `original`, renames that prefix to `relocated` so `original` is unavailable,
and configures a separate CXX-only project through `find_package(LibDDLA)`.
If optional bindings were configured, build their installable targets first.
The check does not build the library, modify the configured install prefix or
change canonical installations. It leaves scratch evidence for inspection.

Expected output includes `installed public-header consumer PASS
DDLA_HAS_GPU=0` for CPU-only or `=1` for dual builds and a passing one-rank CTest.
Only CPU computations execute. Dual builds additionally resolve a GPU `scal`
specialization without creating a GPU handle. No private headers or vendor
macros are used. `installed-target.txt` reports the imported library path,
includes, link providers, compile definitions and C++ standard requirement;
`compile_commands.json` records the standalone consumer's actual compile line.
Source/build/original-prefix include paths and private connector definitions
are rejected during configuration.

For CUDA provider inspection, inspect the library configure log or
`DDLA_CUDA_LIBRARY_PROVIDERS` in its `CMakeCache.txt`. A standard toolkit should
select `CUDA::cublas`, `CUDA::cublasLt`, `CUDA::curand`, `CUDA::cusolver` and
`CUDA::cudart`. Explicit library paths take precedence. If an imported target
is unavailable, normal library search plus the optional
`DDLA_CUDA_MATH_LIBRARY_DIR`/detected NVHPC hint supplies a concrete library
path. Concrete overrides/fallback dependencies retain their installation paths;
relocating LibDDLA does not relocate its external MPI/BLAS/GPU dependencies.
