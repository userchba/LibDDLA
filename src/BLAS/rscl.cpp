#include "blas_impl.h"
#include "ddla_stream.h"
#include "helpers.h"
#include "instantiate.h"
#include "require_backend.h"

#include <cmath>
#include <cfloat>
#include <limits>
#include <stdexcept>

extern "C" {
void srscl_(const int*, const float*, float*, const int*);
void drscl_(const int*, const double*, double*, const int*);
void csrscl_(const int*, const float*, std::complex<float>*, const int*);
void zdrscl_(const int*, const double*, std::complex<double>*, const int*);
void sladiv_(const float*, const float*, const float*, const float*, float*, float*);
void dladiv_(const double*, const double*, const double*, const double*, double*, double*);
}

namespace ddla::blas_impl {
namespace {

template <typename T, typename R>
T cpu_divide(T x, R a)
{
    return x / a;
}

template <typename R>
std::complex<R> cpu_divide(std::complex<R> x, std::complex<R> a)
{
    const R xr = x.real(), xi = x.imag(), ar = a.real(), ai = a.imag();
    if (ai == R(0)) return {xr / ar, xi / ar};
    if (ar == R(0)) return {xi / ai, -xr / ai};
    R real, imaginary;
    if constexpr (std::is_same_v<R, float>)
        sladiv_(&xr, &xi, &ar, &ai, &real, &imaginary);
    else
        dladiv_(&xr, &xi, &ar, &ai, &real, &imaginary);
    return {real, imaginary};
}

template <typename T, typename R>
void cpu_rscl(int n, R a, T* x, int incx)
{
    if constexpr (std::is_same_v<T, float>) srscl_(&n, &a, x, &incx);
    else if constexpr (std::is_same_v<T, double>) drscl_(&n, &a, x, &incx);
    else if constexpr (std::is_same_v<R, float>) csrscl_(&n, &a, x, &incx);
    else zdrscl_(&n, &a, x, &incx);
}

template <typename T>
auto component_max(T value)
{
    if constexpr (std::is_arithmetic_v<T>) return std::abs(value);
    else return std::max(std::abs(value.real()), std::abs(value.imag()));
}

template <typename T, typename R>
bool scaling_range(T value, R lower, R upper)
{
    const auto inside = [=](R component) {
        const R magnitude = std::abs(component);
        return magnitude == R(0) || (magnitude >= lower && magnitude <= upper);
    };
    return inside(std::real(value)) && inside(std::imag(value));
}

#if DDLA_HAS_GPU
template <typename R>
__device__ R gpu_divide(R x, R a) { return x / a; }

template <typename R>
__device__ thrust::complex<R> gpu_divide(thrust::complex<R> x, R a)
{
    return {x.real() / a, x.imag() / a};
}

// Baudin/Smith division, adapted from LAPACK xLADIV (see LAPACK-LICENSE).
// No cuBLAS/hipBLAS entry provides a vector division with this range handling.
template <typename R>
__device__ R divide_component(R a, R b, R c, R d, R ratio, R inverse)
{
    if (ratio == R(0)) return (a + d * (b / c)) * inverse;
    const R product = b * ratio;
    return product != R(0) ? (a + product) * inverse
                           : a * inverse + (b * inverse) * ratio;
}

template <typename R>
__device__ thrust::complex<R> gpu_divide(thrust::complex<R> x, thrust::complex<R> a)
{
    if (a.imag() == R(0)) return gpu_divide(x, a.real());
    if (a.real() == R(0)) return {x.imag() / a.imag(), -x.real() / a.imag()};
    R ar = x.real(), ai = x.imag(), br = a.real(), bi = a.imag(), scale = R(1);
    const R numerator = fmax(fabs(ar), fabs(ai));
    const R denominator = fmax(fabs(br), fabs(bi));
    constexpr R maximum = std::is_same_v<R, float> ? FLT_MAX : DBL_MAX;
    constexpr R minimum = std::is_same_v<R, float> ? FLT_MIN : DBL_MIN;
    constexpr R epsilon = (std::is_same_v<R, float> ? FLT_EPSILON : DBL_EPSILON) / R(2);
    constexpr R boost = R(2) / (epsilon * epsilon);
    if (numerator >= maximum / R(2)) {
        ar *= R(.5); ai *= R(.5); scale *= R(2);
    }
    if (denominator >= maximum / R(2)) {
        br *= R(.5); bi *= R(.5); scale *= R(.5);
    }
    if (numerator <= minimum * R(2) / epsilon) {
        ar *= boost; ai *= boost; scale /= boost;
    }
    if (denominator <= minimum * R(2) / epsilon) {
        br *= boost; bi *= boost; scale *= boost;
    }
    const bool swapped = fabs(bi) > fabs(br);
    if (swapped) {
        const R old_ar = ar, old_br = br;
        ar = ai; ai = old_ar; br = bi; bi = old_br;
    }
    const R ratio = bi / br, inverse = R(1) / (br + bi * ratio);
    const R real = divide_component(ar, ai, br, bi, ratio, inverse);
    const R imaginary = divide_component(ai, -ar, br, bi, ratio, inverse);
    return {real * scale, (swapped ? -imaginary : imaginary) * scale};
}

template <typename T, typename A>
__global__ void divide_vector(int n, A a, T* x, int incx)
{
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < std::size_t(n); i += std::size_t(gridDim.x) * blockDim.x)
        x[i * incx] = gpu_divide(x[i * incx], a);
}

template <typename T, typename A>
__global__ void divide_rows(int m, int n, const A* diagonal, T* x, int lda)
{
    for (std::size_t p = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         p < std::size_t(m) * n; p += std::size_t(gridDim.x) * blockDim.x) {
        const int row = int(p % m), column = int(p / m);
        const auto index = row + std::size_t(column) * lda;
        x[index] = gpu_divide(x[index], diagonal[row]);
    }
}
#endif
} // namespace

template <DdlaBackend Backend, typename T, typename Alpha>
void rscl_run(const DdlaHandle_t& h, int n, const Alpha& divisor, T* x, int incx)
{
    const Alpha a = divisor;
    if (!n || a == Alpha(1)) return;
    if constexpr (Backend == DdlaBackend::CPU) {
        if constexpr (std::is_arithmetic_v<Alpha>) {
            // LAPACK's reciprocal scaling can round a finite quotient to Inf
            // at the range boundary. Divide directly for tiny divisors or
            // potentially overflowing RHS values instead of rescaling them.
            using R = Alpha;
            bool direct = std::abs(a) < std::numeric_limits<R>::min();
            if (!direct && std::abs(a) < R(1)) {
                const R bound = std::abs(a) * (std::numeric_limits<R>::max() / R(4));
                for (int i = 0; i < n && !direct; ++i)
                    direct = component_max(x[std::size_t(i) * incx]) > bound;
            }
            if (!direct) {
                cpu_rscl(n, a, x, incx);
                return;
            }
        } else {
            using R = typename real_type<T>::type;
            // Keep operands well inside the range where reciprocal scaling
            // can neither overflow products/sums nor underflow products.
            // Extreme values retain xLADIV; no unsafe reciprocal is formed.
            const R lower = std::ldexp(R(1), std::numeric_limits<R>::min_exponent / 4 + 2);
            const R upper = std::ldexp(R(1), std::numeric_limits<R>::max_exponent / 4 - 2);
            if (scaling_range(a, lower, upper)) {
                const T inverse = cpu_divide(T(1), a);
                bool ordinary = scaling_range(inverse, lower, upper);
                for (int i = 0; i < n && ordinary; ++i)
                    ordinary = scaling_range(x[std::size_t(i) * incx], lower, upper);
                if (ordinary) {
                    scal_run<Backend, T>(h, n, inverse, x, incx);
                    return;
                }
            }
        }
        detail::host_parallel_for(n, 65536, h->host_threads, [&](long long lo, long long hi) {
            for (auto i = lo; i < hi; ++i)
                x[std::size_t(i) * incx] = cpu_divide(x[std::size_t(i) * incx], a);
        });
    } else {
#if DDLA_HAS_GPU
        using DT = detail::DeviceScalarT<T>;
        using DA = detail::DeviceScalarT<Alpha>;
        divide_vector<DT, DA><<<unsigned(std::min((std::size_t(n) + 255) / 256,
                                               std::size_t(65535))), 256, 0,
                                static_cast<gpuStream_t>(h->stream)>>>(
            n, detail::to_device_scalar(a), reinterpret_cast<DT*>(x), incx);
        DDLA_CHECK(ddlaGetLastError(h));
#endif
    }
}

// Packed local diagonal: no global matrix or row-index array is replicated.
template <DdlaBackend Backend, typename T, typename Alpha>
void rscl_rows_run(const DdlaHandle_t& h, int m, int n, const Alpha* diagonal, T* x, int lda)
{
    if (!m || !n) return;
    if constexpr (Backend == DdlaBackend::CPU) {
        if constexpr (!std::is_arithmetic_v<Alpha>) {
            // Reuse one guarded scalar-vector solve per row rather than
            // evaluating the same complex reciprocal for every RHS column.
            for (int i = 0; i < m; ++i)
                rscl_run<Backend, T, Alpha>(h, n, diagonal[i], x + i, lda);
        } else {
            dgmm_columns(m, n, h->host_threads, [&](int lo, int hi) {
                for (int j = lo; j < hi; ++j)
                    for (int i = 0; i < m; ++i) {
                        const auto index = i + std::size_t(j) * lda;
                        x[index] = cpu_divide(x[index], diagonal[i]);
                    }
            });
        }
    } else {
#if DDLA_HAS_GPU
        using DT = detail::DeviceScalarT<T>;
        using DA = detail::DeviceScalarT<Alpha>;
        divide_rows<DT, DA><<<unsigned(std::min((std::size_t(m) * n + 255) / 256,
                                             std::size_t(65535))), 256, 0,
                              static_cast<gpuStream_t>(h->stream)>>>(
            m, n, reinterpret_cast<const DA*>(diagonal), reinterpret_cast<DT*>(x), lda);
        DDLA_CHECK(ddlaGetLastError(h));
#endif
    }
}
} // namespace ddla::blas_impl

namespace ddla {
template <DdlaBackend Backend, typename T, typename Alpha>
void rscl(const DdlaHandle_t& h, int n, const Alpha& a, T* x, int incx)
{
    static_assert(std::is_same_v<T, Alpha> ||
                  std::is_same_v<Alpha, typename real_type<T>::type>,
                  "rscl: incompatible divisor type");
    detail::require_backend<Backend>(h, "rscl");
    if (n < 0) throw std::invalid_argument("rscl: negative vector length");
    if (!n) return;
    if (!x || incx <= 0 || a == Alpha(0) || !std::isfinite(std::real(a)) ||
        !std::isfinite(std::imag(a)))
        throw std::invalid_argument("rscl: invalid vector, stride or divisor");
    if (std::size_t(n - 1) > (std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) /
                             sizeof(T) - 1) / std::size_t(incx))
        throw std::invalid_argument("rscl: vector span exceeds addressable storage");
    blas_impl::rscl_run<Backend, T, Alpha>(h, n, a, x, incx);
}

#define INSTANCE(B,T,A) \
    template void rscl<B,T,A>(const DdlaHandle_t&,int,const A&,T*,int); \
    template void blas_impl::rscl_run<B,T,A>(const DdlaHandle_t&,int,const A&,T*,int); \
    template void blas_impl::rscl_rows_run<B,T,A>(const DdlaHandle_t&,int,int,const A*,T*,int)
#define TYPES(B) \
    INSTANCE(B,float,float); INSTANCE(B,double,double); \
    INSTANCE(B,std::complex<float>,std::complex<float>); \
    INSTANCE(B,std::complex<double>,std::complex<double>); \
    INSTANCE(B,std::complex<float>,float); INSTANCE(B,std::complex<double>,double)
DDLA_FOR_EACH_BACKEND(TYPES)
#undef TYPES
#undef INSTANCE
} // namespace ddla
