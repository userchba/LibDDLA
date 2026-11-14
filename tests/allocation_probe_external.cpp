// A separate DSO makes attribution independent of the installed MPI transport.
// Volatile function pointers keep the allocation calls observable in release
// builds; the caller of this DSO is deliberately the monitored executable.
#include <cstdlib>
#include <new>

extern "C" int ddla_test_external_allocations()
{
    using Allocate=void*(*)(std::size_t);
    using AlignedAllocate=void*(*)(std::size_t,std::align_val_t);
    using PosixAllocate=int(*)(void**,std::size_t,std::size_t);
    try {
        Allocate volatile plain=static_cast<Allocate>(&::operator new);
        Allocate volatile array=static_cast<Allocate>(&::operator new[]);
        AlignedAllocate volatile aligned=static_cast<AlignedAllocate>(&::operator new);
        AlignedAllocate volatile aligned_array=static_cast<AlignedAllocate>(&::operator new[]);
        PosixAllocate volatile posix=&posix_memalign;
        void* p=plain(128); ::operator delete(p);
        p=array(128); ::operator delete[](p);
        p=aligned(128,std::align_val_t(64)); ::operator delete(p,std::align_val_t(64));
        p=aligned_array(128,std::align_val_t(64)); ::operator delete[](p,std::align_val_t(64));
        if(posix(&p,64,128)) return 0;
        std::free(p);
        return 1;
    } catch(const std::bad_alloc&) { return 0; }
}
