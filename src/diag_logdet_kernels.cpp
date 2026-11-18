#include "diag_logdet_impl.h"
#include "ddla_connector.h"
#include "ddla_stream.h"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#if defined(DDLA_USE_CUDA)
#include <thrust/system/cuda/execution_policy.h>
#elif defined(DDLA_USE_HIP)
#include <thrust/system/hip/execution_policy.h>
#endif

namespace ddla { namespace detail {
template<class T>
DiagLogState diagonal_logsum_gpu(const DdlaHandle_t& h, const T* a, const DiagLogLayout& layout)
{
    const auto stream = static_cast<gpuStream_t>(h->stream);
#if defined(DDLA_USE_CUDA)
    auto policy = thrust::cuda::par.on(stream);
#else
    auto policy = thrust::hip::par.on(stream);
#endif
    const auto first = thrust::make_counting_iterator(0);
    return thrust::transform_reduce(policy, first, first + layout.count,
                                   DiagLogTerm<T>{a, layout}, DiagLogState{}, DiagLogJoin{});
}
#define INST(T) template DiagLogState diagonal_logsum_gpu<T>(const DdlaHandle_t&,const T*,const DiagLogLayout&);
INST(float) INST(double) INST(std::complex<float>) INST(std::complex<double>)
#undef INST
} }
