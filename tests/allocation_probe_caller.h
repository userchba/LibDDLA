#ifndef DDLA_ALLOCATION_PROBE_CALLER_H
#define DDLA_ALLOCATION_PROBE_CALLER_H

#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>

namespace allocation_probe {
// Like workspace_trace, attribute an allocation to its first non-allocator
// owner. In particular, never look through MPI/vendor frames for a DDLA frame.
// Start at the allocator's caller: the interposing wrappers themselves live
// in the executable and must not make a vendor allocation look test-owned.
inline bool managed_caller(void* caller)
{
    static char anchor;
    Dl_info self{};
    if(!dladdr(&anchor,&self)) return true;
    enum Owner { Managed, External, Runtime, Unknown };
    auto owner=[&](void* address) {
        Dl_info info{};
        if(!dladdr(address,&info) || !info.dli_fname) return Unknown;
        if(info.dli_fbase==self.dli_fbase) return Managed;
        const char* slash=std::strrchr(info.dli_fname,'/');
        const char* file=slash?slash+1:info.dli_fname;
        if(std::strstr(file,"libddla.") || std::strstr(file,"libddla-")) return Managed;
        if(std::strstr(file,"libstdc++") || std::strstr(file,"libc++") ||
           std::strstr(file,"libgcc_s") || std::strstr(file,"libunwind") ||
           std::strncmp(file,"libc.so",7)==0 || std::strncmp(file,"ld-linux",8)==0) return Runtime;
        return External;
    };
    const auto direct=owner(caller);
    if(direct!=Runtime) return direct!=External;
    // An out-of-line C++ allocator can sit between operator new and the owner.
    // Locate the exact caller boundary rather than assuming a stack depth or
    // depending on exported wrapper symbols / compiler inlining decisions.
    void* frames[32];
    const int count=backtrace(frames,32);
    bool past_caller=false;
    for(int i=0;i<count;++i) {
        if(!past_caller) { past_caller=frames[i]==caller; continue; }
        const auto current=owner(frames[i]);
        if(current!=Runtime) return current!=External;
    }
    // Missing unwind/symbol information must not silently weaken the probe.
    return true;
}
}

#endif
