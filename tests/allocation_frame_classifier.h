#ifndef DDLA_TEST_ALLOCATION_FRAME_CLASSIFIER_H
#define DDLA_TEST_ALLOCATION_FRAME_CLASSIFIER_H
#include "allocation_attribution_probe_api.h"
#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>

namespace allocation_test {
static inline const char* basename(const char* path)
{
    if(!path)return "";
    const char* last=std::strrchr(path,'/');
    return last?last+1:path;
}
template<std::size_t N>static inline void copy(char (&out)[N],const char* text)
{
    if(!text)text="";
    std::size_t i=0;
    for(;i+1<N&&text[i];++i)out[i]=text[i];
    out[i]=0;
}
static inline bool library(const char* file)
{
    return std::strncmp(file,"libddla.",8)==0||std::strncmp(file,"libddla-",8)==0;
}
static inline bool system_allocator(const char* file)
{
    return std::strstr(file,"libstdc++")||std::strstr(file,"libc++")||
        std::strstr(file,"libgcc_s")||std::strstr(file,"libunwind")||
        std::strncmp(file,"libc.so",7)==0||std::strncmp(file,"ld-linux",8)==0;
}
static inline bool verified_helper(const char* symbol)
{
    // Observed WEAK DEFAULT in BOTH the executable and DDLA in runtime84d5.
    // The executable can preempt this allocator-only template instantiation.
    // No arbitrary executable, vendor or general std::* frame is skipped.
    return symbol&&std::strcmp(symbol,"_ZNSt6vectorISt4byteSaIS0_EE17_M_default_appendEm")==0;
}
static inline bool known_interposer_allocator(const char* file,const char* symbol)
{
    const bool interposer=std::strcmp(file,"libpblas_alloc_failure.so")==0||
                          std::strcmp(file,"libworkspace_trace.so")==0;
    return interposer&&symbol&&
        (std::strncmp(symbol,"_Znw",4)==0||std::strncmp(symbol,"_Zna",4)==0);
}
static inline AllocationProbeOwner inspect(const char* own_path,const char* executable,
                                     AllocationAttribution* output=nullptr)
{
    AllocationAttribution result{};
    void* frames[40];const int count=backtrace(frames,40);
    bool first=true;
    for(int i=1;i<count;++i) {
        Dl_info info{};
        if(!dladdr(frames[i],&info)||!info.dli_fname)continue;
        if(own_path&&std::strcmp(info.dli_fname,own_path)==0)continue;
        const char* file=basename(info.dli_fname);
        if(system_allocator(file))continue;
        if(first) {
            copy(result.first_file,info.dli_fname);copy(result.first_symbol,info.dli_sname);
            result.legacy_library=library(file);first=false;
        }
        if(verified_helper(info.dli_sname)) {
            if(!result.peeled_helpers)copy(result.helper_file,info.dli_fname);
            ++result.peeled_helpers;continue;
        }
        if(known_interposer_allocator(file,info.dli_sname)) {
            ++result.peeled_interposers;continue;
        }
        copy(result.owner_file,info.dli_fname);copy(result.owner_symbol,info.dli_sname);
        if(library(file))result.owner=AllocationProbeOwner::Library;
        else if(executable&&*executable&&std::strcmp(file,basename(executable))==0)
            result.owner=AllocationProbeOwner::Fixture;
        else result.owner=AllocationProbeOwner::Other;
        // STOP at this semantic owner. In particular, never walk through a
        // vendor operation merely because a DDLA caller exists farther up.
        break;
    }
    if(output)*output=result;
    return result.owner;
}
}
#endif
