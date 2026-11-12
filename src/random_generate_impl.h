#ifndef DDLA_RANDOM_GENERATE_IMPL_H
#define DDLA_RANDOM_GENERATE_IMPL_H
#include <ddla/ddla.h>
#include "ddla_stream.h"
#include "ddla_connector.h"
#include <cstdint>
#include <random>

namespace ddla { namespace random_impl {
// Prepared resources are outside check/run. The CPU engine and GPU generator
// retain their sequence across reuse; neither retains ownership of a handle.
class PreparedRng {
#if DDLA_HAS_GPU
    gpurandGenerator_t generator_=nullptr;
#endif
public:
    std::mt19937_64 engine;
    PreparedRng(DdlaHandle_t h,std::uint64_t seed):engine(seed)
    {
#if DDLA_HAS_GPU
        if(h->backend==DdlaBackend::GPU) {
            GPURAND_CHECK(gpurandCreateGenerator(&generator_,GPURAND_RNG_PSEUDO_DEFAULT));
            try {
                GPURAND_CHECK(gpurandSetStream(generator_,static_cast<gpuStream_t>(h->stream)));
                GPURAND_CHECK(gpurandSetPseudoRandomGeneratorSeed(generator_,seed));
            }catch(...){(void)gpurandDestroyGenerator(generator_);generator_=nullptr;throw;}
        }
#else
        (void)h;
#endif
    }
    PreparedRng(const PreparedRng&)=delete;
    PreparedRng& operator=(const PreparedRng&)=delete;
    ~PreparedRng() noexcept {
#if DDLA_HAS_GPU
        if(generator_)(void)gpurandDestroyGenerator(generator_);
#endif
    }
#if DDLA_HAS_GPU
    gpurandGenerator_t generator() const noexcept{return generator_;}
#endif
    void reset() {
#if DDLA_HAS_GPU
        if(generator_){GPURAND_CHECK(gpurandDestroyGenerator(generator_));generator_=nullptr;}
#endif
    }
};
template<DdlaBackend B,class T> void random_generate_check(const DdlaHandle_t&,T*,const int64_t&);
template<DdlaBackend B,class T> void random_generate_run(const DdlaHandle_t&,PreparedRng&,T*,const int64_t&);
}}
#endif
