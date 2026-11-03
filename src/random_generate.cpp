#include "random_generate_impl.h"
#include <ddla/ddla.h>
#include <ddla/ddla_handle_t.h>

#include <chrono>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "ddla_connector.h"
#include "ddla_stream.h"

namespace ddla {

namespace {

inline const char* random_generate_backend_name(DdlaBackend backend)
{
    return backend == DdlaBackend::CPU ? "CPU" : "GPU";
}

} // anonymous namespace

namespace random_impl {
template <DdlaBackend Backend, typename T>
void random_generate_check(const DdlaHandle_t& handle, T* data, const int64_t& lengthOfData)
{
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "GPU random_generate is not available in this LibDDLA build");

    if (handle == nullptr) {
        throw std::runtime_error("random_generate: null handle");
    }
    const DdlaBackend actual = ddlaGetBackend(handle);
    if (actual != Backend) {
        throw std::runtime_error(
            std::string("random_generate: template backend ") + random_generate_backend_name(Backend) +
            " does not match handle backend " + random_generate_backend_name(actual));
    }
    if (lengthOfData <= 0) return;
    if (data == nullptr) throw std::runtime_error("random_generate: null data");
}
template <DdlaBackend Backend, typename T>
void random_generate_run(const DdlaHandle_t& handle, PreparedRng& resource, T* data, const int64_t& lengthOfData)
{
    try {
        if(lengthOfData<=0)return;
        if constexpr (Backend == DdlaBackend::CPU) {
            auto& rng=resource.engine;
            if constexpr (std::is_same_v<T, float>) {
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                for (int64_t i = 0; i < lengthOfData; ++i) data[i] = dist(rng);
            } else if constexpr (std::is_same_v<T, double>) {
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                for (int64_t i = 0; i < lengthOfData; ++i) data[i] = dist(rng);
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                float* p = reinterpret_cast<float*>(data);
                for (int64_t i = 0; i < 2 * lengthOfData; ++i) p[i] = dist(rng);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                double* p = reinterpret_cast<double*>(data);
                for (int64_t i = 0; i < 2 * lengthOfData; ++i) p[i] = dist(rng);
            } else {
                static_assert(sizeof(T) == 0,
                              "random_generate supports only float, double, std::complex<float>, std::complex<double>");
            }
        } else {
#if DDLA_HAS_GPU
            const auto gen=resource.generator();
            if constexpr (std::is_same_v<T, float>) {
                GPURAND_CHECK(gpurandGenerateUniform(gen, data, lengthOfData));
            } else if constexpr (std::is_same_v<T, double>) {
                GPURAND_CHECK(gpurandGenerateUniformDouble(gen, data, lengthOfData));
            } else if constexpr (std::is_same_v<T, std::complex<float>>) {
                GPURAND_CHECK(gpurandGenerateUniform(gen, reinterpret_cast<float*>(data), lengthOfData * 2));
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                GPURAND_CHECK(gpurandGenerateUniformDouble(gen, reinterpret_cast<double*>(data), lengthOfData * 2));
            } else {
                static_assert(sizeof(T) == 0,
                              "random_generate supports only float, double, std::complex<float>, std::complex<double>");
            }
            // The generator must outlive the kernel it launched; this also makes the call return with the values written.
            DDLA_CHECK(ddlaSynchronize(handle));
#endif
        }
    } catch (...) {
        // Retire queued host reads before the caller reuses or releases scratch.
        if (ddlaGetBackend(handle) == DdlaBackend::GPU)
            try { (void)ddlaSynchronize(handle); } catch (...) {}
        throw;
    }
}
} // namespace random_impl
template <DdlaBackend Backend, typename T>
void random_generate(const DdlaHandle_t& handle, T* data, const int64_t& lengthOfData)
{
    random_impl::random_generate_check<Backend,T>(handle,data,lengthOfData);
    if(lengthOfData<=0)return;
    const auto seed=static_cast<std::uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
    random_impl::PreparedRng resource(handle,seed);
    random_impl::random_generate_run<Backend,T>(handle,resource,data,lengthOfData);
    resource.reset();
}

#define INSTANTIATE_RANDOM_GENERATE(BACKEND, TYPE)                           \
    template void random_generate<BACKEND, TYPE>(const DdlaHandle_t&, TYPE*, const int64_t&); \
    template void random_impl::random_generate_check<BACKEND,TYPE>(const DdlaHandle_t&,TYPE*,const int64_t&); \
    template void random_impl::random_generate_run<BACKEND,TYPE>(const DdlaHandle_t&,random_impl::PreparedRng&,TYPE*,const int64_t&)

INSTANTIATE_RANDOM_GENERATE(DdlaBackend::CPU, float);
INSTANTIATE_RANDOM_GENERATE(DdlaBackend::CPU, double);
INSTANTIATE_RANDOM_GENERATE(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_RANDOM_GENERATE(DdlaBackend::CPU, std::complex<double>);

#if DDLA_HAS_GPU
INSTANTIATE_RANDOM_GENERATE(DdlaBackend::GPU, float);
INSTANTIATE_RANDOM_GENERATE(DdlaBackend::GPU, double);
INSTANTIATE_RANDOM_GENERATE(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_RANDOM_GENERATE(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_RANDOM_GENERATE

} // namespace ddla
