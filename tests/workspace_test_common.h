#ifndef DDLA_WORKSPACE_TEST_COMMON_H
#define DDLA_WORKSPACE_TEST_COMMON_H

#include "workspace_trace_api.h"
#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <complex>
#include <limits>

namespace workspace_test {
using Snapshot=workspace_trace_snapshot;
inline double finite_error(double error) noexcept
{return std::isfinite(error)?error:std::numeric_limits<double>::infinity();}
template<class T> double relative_error(const T& observed,const T& reference)
{
    const double error=static_cast<double>(std::abs(observed-reference));
    const double scale=static_cast<double>(std::abs(reference));
    return std::isfinite(error)&&std::isfinite(scale)?error/(1+scale):std::numeric_limits<double>::infinity();
}

namespace detail {
struct Api {
    using Begin=void(*)(int);
    using End=void(*)(workspace_trace_snapshot*);
    Begin begin=nullptr;
    End end=nullptr;
    Api()
    {
        // Resolve both symbols before begin enables any counters. memcpy
        // avoids assuming that data/function pointer casts are C++ portable.
        void* b=dlsym(RTLD_DEFAULT,"workspace_trace_begin");
        void* e=dlsym(RTLD_DEFAULT,"workspace_trace_end");
        static_assert(sizeof(b)==sizeof(begin),"dlsym function pointer ABI");
        std::memcpy(&begin,&b,sizeof(begin));std::memcpy(&end,&e,sizeof(end));
    }
};
inline Api& api(){static Api value;return value;}
}

inline bool available()
{const auto& api=detail::api();return api.begin && api.end;}

class Scope {
    bool active_=false;
public:
    explicit Scope(bool query=false)
    {
        auto& api=detail::api();
        if(!api.begin || !api.end) {
            const char* required=std::getenv("DDLA_WORKSPACE_TRACE_REQUIRED");
            if(required && std::strcmp(required,"1")==0)
                throw std::runtime_error("workspace diagnostics required: workspace_trace preload is missing");
            return;
        }
        api.begin(query?1:0);active_=true;
    }
    Scope(const Scope&)=delete;
    Scope& operator=(const Scope&)=delete;
    Snapshot finish() noexcept
    {
        Snapshot result{};
        if(active_){detail::api().end(&result);active_=false;}
        return result;
    }
    ~Scope() noexcept{if(active_)(void)finish();}
};
}
#endif
