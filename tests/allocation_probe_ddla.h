#ifndef DDLA_ALLOCATION_PROBE_DDLA_H
#define DDLA_ALLOCATION_PROBE_DDLA_H

#include <ddla/ddla_handle_t.h>
#include <cstring>
#include <dlfcn.h>
#include <new>
#include <stdexcept>

namespace allocation_probe {void inspect();}

// ddla::vector obtains CPU scratch through ddlaMalloc, not operator new.
// Observe that allocator too, without intercepting MPI's private malloc calls.
namespace ddla {
ddlaStatus_t ddlaMalloc(const DdlaHandle_t& handle,void** ptr,std::size_t bytes)
{
    using Allocate=ddlaStatus_t(*)(const DdlaHandle_t&,void**,std::size_t);
    static const Allocate real=[] {
        Allocate self=&ddlaMalloc,fn=nullptr;
        void* address=nullptr;
        static_assert(sizeof(address)==sizeof(self),"dlsym function pointer ABI");
        std::memcpy(&address,&self,sizeof(address));
        Dl_info symbol{};
        if(!dladdr(address,&symbol) || !symbol.dli_sname)
            throw std::runtime_error("cannot resolve ddlaMalloc probe");
        address=dlsym(RTLD_NEXT,symbol.dli_sname);
        std::memcpy(&fn,&address,sizeof(fn));
        if(!fn) throw std::runtime_error("cannot resolve original ddlaMalloc");
        return fn;
    }();
    try {allocation_probe::inspect();}
    catch(const std::bad_alloc&) {
        *ptr=nullptr;
        return ddlaStatus_t::DDLA_STATUS_ALLOC_FAILED;
    }
    return real(handle,ptr,bytes);
}
}

#endif
