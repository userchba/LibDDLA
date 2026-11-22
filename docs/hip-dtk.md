# HIP / DTK compatibility notes

## POTRF and the unused triangle

On df_dcu gfx906 DCUs, DTK hipSOLVER POTRF can modify entries in the strict
opposite triangle of the matrix. Treat that triangle as unspecified on return;
consume the Cholesky factor only in the triangle selected by `uplo`.
LibDDLA calls the native solver directly and does not allocate an extra matrix
or copy the unused triangle to preserve it.

The following direct hipSOLVER checks ran on November 21, 2026. Each version
was loaded in a separate process with its own DTK runtime and solver libraries.
The checks covered float, double, complex float and complex double, both upper
and lower factors, orders 33, 64, 100 and 257, and leading-dimension padding of
zero or four rows. The unused input triangle contained either finite sentinels
or NaNs: 128 cases per version.

| DTK version | Selected factor / info / padding | Unused triangle modified |
| --- | --- | --- |
| 22.10.1 | 128/128 passed | Not observed |
| 23.10.1 | 128/128 passed | Not observed |
| 24.04.3 | 128/128 passed | Observed for float and double |
| 25.04.3 | 128/128 passed | Observed for float and double |
| 26.04 | 128/128 passed | Observed for float and double |

These are results for the installed builds and tested shapes, not guarantees
for every installation, size or future DTK version. No unused-triangle writes
were observed for complex types in this set. Correct factors with NaNs in the
unused input triangle distinguish this behavior from contamination of the
selected factor.

LibDDLA's POTRF regression checks `info`, reconstructs the matrix from the
selected factor, rejects non-finite residuals, and checks leading-dimension
padding bit for bit. It does not require the strict opposite output triangle
to retain its input bytes. The unused input remains poisoned, so reading it
into the numerical result is still an error.

This is a DTK compatibility limitation, not a reason to accept a wrong selected
factor or writes outside the matrix. It is separate from the extreme-input
scaling used by HIP bidiagonalization and Hermitian imaginary-diagonal handling.

The native solver interface describes the output as the selected triangular
factor: see [rocSOLVER POTRF](https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html#rocsolver-type-potrf).
The cross-version observations above come from actual DTK runs, not an inference
about DTK from upstream ROCm documentation.

## HERK/HER2K and ignored imaginary diagonals

On df_dcu gfx906 with DTK 25.04.3 (checked November 22, 2026), native hipBLAS
HERK/HER2K can propagate a NaN from an input imaginary diagonal of `C` into
the real selected output. A one-batch strided-batched hipBLAS call showed the
same limitation. This is a selected-output NaN, rather than an unused-triangle
write like the POTRF behavior above.

LibDDLA keeps the native HERK/HER2K calls. For portable HIP use, initialize
Hermitian input diagonals to finite values, preferably with zero imaginary
parts. LibDDLA does not add a diagonal-cleaning kernel or memory operation to
these calls. For updates that read `C`, the HIP structured rank-update fixture
uses a finite nonzero imaginary diagonal to check ordinary finite-input behavior. CPU and CUDA
fixtures retain the ignored-imaginary-diagonal NaN control. The selected
result must still be finite, its diagonal real, and the opposite triangle,
inputs, parent entries and leading-dimension padding intact.

Both [hipBLAS](https://rocm.docs.amd.com/projects/hipBLAS/en/develop/reference/hipblas-api-functions.html)
and [cuBLAS](https://docs.nvidia.com/cuda/archive/11.8.0/cublas/index.html#cublas-t-herk)
describe the imaginary input diagonal as unused or assumed zero. The
observation above is a limitation of the tested DTK build; it does not establish
behavior of other DTK versions. CUDA was not rerun in this df_dcu check.

## Native QR/QL at extreme finite scales

On the same DTK 25.04.3 / gfx906 stack, native hipSOLVER QR can produce
incorrect reflector scalars or non-finite factors for finite huge, tiny,
subnormal or mixed-scale inputs. The QL wrapper uses a reversed QR and
inherits this limitation. The CPU LAPACK control passed the tested stress
inputs. Separate native DTK NRM2 probes returned infinity or zero even when
the true norm was finite and representable, consistent with an overflow or
underflow problem; those probes do not prove that NRM2 alone causes every
QR failure.

Ordinary well-scaled distributed square cases of order 1024 and 4096 passed
independent residual and Q-adjoint consistency checks for all four scalar
types on a 2-by-2 grid. This does not establish support for arbitrary extreme
scales. LibDDLA keeps the native QR/QL path to preserve its performance;
it does not add the slower robust replacement or automatic matrix scaling.
The finite-extreme regression cases still expose this known DTK limitation.
CUDA was not rerun in this comparison.
