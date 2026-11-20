// Test-only LD_PRELOAD allocator failures. Never linked into LibDDLA.
#include <ddla/ddla_handle_t.h>
#include "staging_preparation_trace_api.h"
#include "allocation_frame_classifier.h"
#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>
#include <new>
#include <cstdint>
#include <unistd.h>

namespace {
std::atomic<int> armed{0}; // 1: ddlaMalloc (backend storage), 2: C++ host storage
std::atomic<unsigned> fired{0};
std::atomic<std::size_t> failed_bytes{0},target_bytes{0};
std::atomic<unsigned> target_ordinal{0};
std::atomic<bool> trace_enabled{false};
std::atomic<std::size_t> traced_allocations{0},slot_max[2]{};
std::atomic<unsigned> traced_messages{0};
std::atomic<unsigned> traced_native_calls{0};
void* observed_slots[2]{};
thread_local unsigned resolving = 0;
const char* own_path = nullptr;
char executable[4096]{};
std::atomic<bool> probe_enabled{false};
std::atomic<std::size_t> probe_target{0};
std::atomic_flag probe_lock=ATOMIC_FLAG_INIT;
AllocationProbeSnapshot probe{},raw_probe{};
struct RelativeFrame {
    std::uintptr_t offset=0;
    char file[192]{},symbol[192]{};
};
struct RelativeStack {
    std::uintptr_t owner_offset=0,helper_offset=0;
    unsigned count=0;
    RelativeFrame frames[6]{};
};
RelativeStack raw_sites[16]{};
struct MessageRecord {
    char primitive[24]{};
    unsigned slot=0,pointer_changed=0;
    std::size_t bytes=0,count=0,width=0;
    RelativeStack site{};
};
MessageRecord raw_messages[32]{};
unsigned raw_message_count=0;
std::uint64_t raw_message_total=0;
void* slot_getter_context=nullptr;
void (*slot_getter)(void*,void**,void**)=nullptr;
struct ResolutionGuard {
    ResolutionGuard() { ++resolving; }
    ~ResolutionGuard() { --resolving; }
};
template <typename F> F next_own(F self)
{
    ResolutionGuard guard;
    void* address = nullptr;
    static_assert(sizeof(address) == sizeof(self), "function pointer ABI");
    std::memcpy(&address, &self, sizeof(address));
    Dl_info info{};
    if (!dladdr(address, &info) || !info.dli_sname) return nullptr;
    address = dlsym(RTLD_NEXT, info.dli_sname);
    F result = nullptr;
    std::memcpy(&result, &address, sizeof(result));
    return result;
}
bool library_caller(AllocationAttribution* record=nullptr)
{
    if(resolving)return false;
    ResolutionGuard guard;
    return allocation_test::inspect(own_path,executable,record)==AllocationProbeOwner::Library;
}
// Additional DIAGNOSTIC stack only. The calibrated classifier above remains
// the sole ownership decision; these later frames never change its result.
RelativeStack relative_stack()
{
    ResolutionGuard guard;
    RelativeStack result{};void* frames[40];const int n=backtrace(frames,40);
    for(int i=1;i<n&&result.count<6;++i){
        Dl_info info{};
        if(!dladdr(frames[i],&info)||!info.dli_fname||!info.dli_fbase)continue;
        if(own_path&&std::strcmp(info.dli_fname,own_path)==0)continue;
        const auto* file=allocation_test::basename(info.dli_fname);
        if(allocation_test::system_allocator(file))continue;
        const auto offset=reinterpret_cast<std::uintptr_t>(frames[i])-reinterpret_cast<std::uintptr_t>(info.dli_fbase);
        if(allocation_test::verified_helper(info.dli_sname)){
            if(!result.helper_offset)result.helper_offset=offset;
            continue;
        }
        if(allocation_test::known_interposer_allocator(file,info.dli_sname))continue;
        if(!result.count)result.owner_offset=offset;
        auto& out=result.frames[result.count++];out.offset=offset;
        allocation_test::copy(out.file,info.dli_fname);allocation_test::copy(out.symbol,info.dli_sname);
    }
    return result;
}
void record_matching(std::size_t bytes,const AllocationAttribution& owner)
{
    while(probe_lock.test_and_set(std::memory_order_acquire)){}
    if(probe_enabled.load(std::memory_order_relaxed)&&bytes==probe_target.load(std::memory_order_relaxed)) {
        ++probe.matching;
        if(owner.owner==AllocationProbeOwner::Library) {
            ++probe.library;
            if(owner.peeled_helpers)++probe.library_with_helper;
            if(!owner.legacy_library)++probe.legacy_rejected_library;
        } else if(owner.owner==AllocationProbeOwner::Fixture)++probe.fixture;
        else if(owner.owner==AllocationProbeOwner::Other)++probe.other;
        else ++probe.unknown;
        if(probe.record_count<16)probe.records[probe.record_count++]={bytes,owner};
    }
    probe_lock.clear(std::memory_order_release);
}
void record_raw(std::size_t bytes,const AllocationAttribution& owner,const RelativeStack& site)
{
    while(probe_lock.test_and_set(std::memory_order_acquire)){}
    ++raw_probe.matching;++raw_probe.library;
    if(owner.peeled_helpers)++raw_probe.library_with_helper;
    if(!owner.legacy_library)++raw_probe.legacy_rejected_library;
    if(raw_probe.record_count<16){
        const unsigned index=raw_probe.record_count++;
        raw_probe.records[index]={bytes,owner};raw_sites[index]=site;
    }
    probe_lock.clear(std::memory_order_release);
}
bool fail(int kind, std::size_t bytes)
{
    if (resolving || !bytes || armed.load(std::memory_order_relaxed) != kind) return false;
    int expected = kind;
    if (!armed.compare_exchange_strong(expected, 0)) return false;
    // Disarm before constructing the exception or entering the caller's handler.
    failed_bytes.store(bytes);
    fired.fetch_add(1);
    char message[128];
    const int n = std::snprintf(message, sizeof(message),
        "PBLAS_ALLOCATION_FAULT kind=%d bytes=%zu\n", kind, bytes);
    if (n > 0) (void)write(STDERR_FILENO, message, std::size_t(n));
    return true;
}
void* allocate(std::size_t size, std::size_t alignment = 0)
{
    const int mode=armed.load(std::memory_order_relaxed);
    const bool matching=!resolving&&probe_enabled.load(std::memory_order_relaxed)&&size==probe_target.load(std::memory_order_relaxed);
    if(!resolving&&(mode==2||mode==3||trace_enabled.load(std::memory_order_relaxed)||matching)) {
        AllocationAttribution owner{};
        const bool tracing=trace_enabled.load(std::memory_order_relaxed);
        const bool managed=library_caller(matching||tracing?&owner:nullptr);
        if(matching)record_matching(size,owner);
        if(managed) {
            if(tracing){++traced_allocations;record_raw(size,owner,relative_stack());}
            if(mode==2&&fail(2,size))throw std::bad_alloc();
            if(mode==3&&size==target_bytes.load()&&target_ordinal.fetch_sub(1)==1&&fail(3,size))
                throw std::bad_alloc();
        }
    }
    for (;;) {
        void* result = nullptr;
        if (alignment) {
            if (posix_memalign(&result, alignment, size ? size : 1) != 0) result = nullptr;
        } else result = std::malloc(size ? size : 1);
        if (result) return result;
        const auto handler = std::get_new_handler();
        if (!handler) throw std::bad_alloc();
        handler();
    }
}
}

void* operator new(std::size_t n) { return allocate(n); }
void* operator new[](std::size_t n) { return allocate(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept { try { return allocate(n); } catch (...) { return nullptr; } }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { try { return allocate(n); } catch (...) { return nullptr; } }
void* operator new(std::size_t n, std::align_val_t a) { return allocate(n, std::size_t(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return allocate(n, std::size_t(a)); }
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept { try { return allocate(n, std::size_t(a)); } catch (...) { return nullptr; } }
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept { try { return allocate(n, std::size_t(a)); } catch (...) { return nullptr; } }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }

namespace ddla {
using MallocFn = ddlaStatus_t (*)(const DdlaHandle_t&, void**, std::size_t);
static MallocFn real_malloc()
{
    static MallocFn fn = next_own<MallocFn>(&ddlaMalloc);
    return fn;
}
ddlaStatus_t ddlaMalloc(const DdlaHandle_t& h, void** pointer, std::size_t bytes)
{
    if (fail(1, bytes)) {
        *pointer = nullptr;
        return ddlaStatus_t::DDLA_STATUS_ALLOC_FAILED;
    }
    const auto fn = real_malloc();
    return fn ? fn(h, pointer, bytes) : ddlaStatus_t::DDLA_STATUS_INTERNAL_ERROR;
}
}

extern "C" int pblas_alloc_failure_prewarm()
{
    ResolutionGuard guard;
    Dl_info info{};
    // Resolve stack walking and symbol lookup before enabling failures.
    void* frames[32];
    (void)backtrace(frames, 32);
    if (dladdr(frames[0], &info)) own_path = info.dli_fname;
    const auto length=readlink("/proc/self/exe",executable,sizeof(executable)-1);
    if(length>0)executable[length]=0;
    return ddla::real_malloc() && own_path ? 1 : 0;
}
extern "C" void pblas_alloc_failure_arm(int kind)
{
    fired.store(0);
    failed_bytes.store(0);
    armed.store(kind);
}
extern "C" void pblas_alloc_failure_disarm() { armed.store(0);probe_enabled.store(false); }
extern "C" unsigned pblas_alloc_failure_fired() { return fired.load(); }
extern "C" std::size_t pblas_alloc_failure_bytes() { return failed_bytes.load(); }

extern "C" void pblas_alloc_failure_arm_host_bytes(std::size_t bytes,unsigned ordinal)
{
    staging_allocation_probe_begin(bytes);
    fired.store(0);failed_bytes.store(0);target_bytes.store(bytes);target_ordinal.store(ordinal);
    armed.store(bytes && ordinal ? 3 : 0);
}
extern "C" void staging_preparation_trace_set_slot_getter(void* context,
    void (*getter)(void*,void**,void**))
{
    slot_getter_context=context;slot_getter=getter;
}
extern "C" void staging_preparation_trace_begin(void* slot0,void* slot1)
{
    trace_enabled.store(false);observed_slots[0]=slot0;observed_slots[1]=slot1;
    while(probe_lock.test_and_set(std::memory_order_acquire)){}
    raw_probe={};for(auto& site:raw_sites)site={};
    raw_message_count=0;raw_message_total=0;
    probe_lock.clear(std::memory_order_release);
    traced_allocations.store(0);slot_max[0].store(0);slot_max[1].store(0);traced_messages.store(0);traced_native_calls.store(0);
    trace_enabled.store(true);
}
extern "C" void staging_preparation_trace_end(StagingPreparationTrace* result)
{
    trace_enabled.store(false);slot_getter=nullptr;slot_getter_context=nullptr;
    *result={traced_allocations.load(),{slot_max[0].load(),slot_max[1].load()},traced_messages.load(),traced_native_calls.load()};
}
namespace {
void staged_message(const char* primitive,const void* send,std::size_t send_count,MPI_Datatype send_type,
                    void* recv,std::size_t recv_count,MPI_Datatype recv_type)
{
    if(!trace_enabled.load(std::memory_order_relaxed))return;
    ++traced_native_calls;
    bool pointer_changed[2]{};
    if(slot_getter){
        void* current[2]{};slot_getter(slot_getter_context,&current[0],&current[1]);
        for(int i=0;i<2;++i){pointer_changed[i]=current[i]!=observed_slots[i];observed_slots[i]=current[i];}
    }
    bool matched=false;
    const auto observe=[&](int slot,const void* pointer,std::size_t count,MPI_Datatype type) {
        if(pointer!=observed_slots[slot])return;
        int width=0;PMPI_Type_size(type,&width);
        const auto bytes=std::size_t(count)*std::size_t(width);
        auto previous=slot_max[slot].load();bool higher=false;
        while(previous<bytes){if(slot_max[slot].compare_exchange_weak(previous,bytes)){higher=true;break;}}
        if(higher){
            const auto site=relative_stack();
            while(probe_lock.test_and_set(std::memory_order_acquire)){}
            ++raw_message_total;
            if(raw_message_count<32){auto& record=raw_messages[raw_message_count++];record={};
                allocation_test::copy(record.primitive,primitive);record.slot=unsigned(slot);
                record.pointer_changed=pointer_changed[slot];record.bytes=bytes;record.count=count;record.width=std::size_t(width);record.site=site;}
            probe_lock.clear(std::memory_order_release);
        }
        matched=true;
    };
    observe(0,send,send_count,send_type);observe(1,recv,recv_count,recv_type);
    if(matched)++traced_messages;
}
}
extern "C" int MPI_Bcast(void* buffer,int count,MPI_Datatype type,int root,MPI_Comm comm)
{
    staged_message("MPI_Bcast",buffer,count,type,nullptr,0,type);
    return PMPI_Bcast(buffer,count,type,root,comm);
}
extern "C" int MPI_Sendrecv(const void* send,int sc,MPI_Datatype st,int dest,int stag,
    void* recv,int rc,MPI_Datatype rt,int source,int rtag,MPI_Comm comm,MPI_Status* status)
{
    staged_message("MPI_Sendrecv",send,sc,st,recv,rc,rt);
    return PMPI_Sendrecv(send,sc,st,dest,stag,recv,rc,rt,source,rtag,comm,status);
}
extern "C" int MPI_Alltoallv(const void* send,const int* sc,const int* sd,MPI_Datatype st,
    void* recv,const int* rc,const int* rd,MPI_Datatype rt,MPI_Comm comm)
{
    if(trace_enabled.load(std::memory_order_relaxed)) {
        int size=0;PMPI_Comm_size(comm,&size);
        staged_message("MPI_Alltoallv",send,size?sd[size-1]+sc[size-1]:0,st,recv,size?rd[size-1]+rc[size-1]:0,rt);
    }
    return PMPI_Alltoallv(send,sc,sd,st,recv,rc,rd,rt,comm);
}
extern "C" int MPI_Reduce(const void* send,void* recv,int count,MPI_Datatype type,MPI_Op op,int root,MPI_Comm comm)
{
    staged_message("MPI_Reduce",send,count,type,recv,count,type);
    return PMPI_Reduce(send,recv,count,type,op,root,comm);
}

// The spectral composites also use point-to-point panel transfers and native
// allreduces. Receive-only transfers use staging slot0, just like sends.
extern "C" int MPI_Send(const void* buffer,int count,MPI_Datatype type,int dest,int tag,MPI_Comm comm)
{
    staged_message("MPI_Send",buffer,count,type,nullptr,0,type);
    return PMPI_Send(buffer,count,type,dest,tag,comm);
}
extern "C" int MPI_Recv(void* buffer,int count,MPI_Datatype type,int source,int tag,MPI_Comm comm,MPI_Status* status)
{
    staged_message("MPI_Recv",buffer,count,type,nullptr,0,type);
    return PMPI_Recv(buffer,count,type,source,tag,comm,status);
}
extern "C" int MPI_Allreduce(const void* send,void* recv,int count,MPI_Datatype type,MPI_Op op,MPI_Comm comm)
{
    staged_message("MPI_Allreduce",send,count,type,recv,count,type);
    return PMPI_Allreduce(send,recv,count,type,op,comm);
}
extern "C" int MPI_Allgather(const void* send,int sc,MPI_Datatype st,void* recv,int rc,MPI_Datatype rt,MPI_Comm comm)
{
    int size=0;PMPI_Comm_size(comm,&size);
    staged_message("MPI_Allgather",send,sc,st,recv,std::size_t(rc)*size,rt);
    return PMPI_Allgather(send,sc,st,recv,rc,rt,comm);
}

extern "C" void staging_allocation_probe_begin(std::size_t bytes)
{
    probe_enabled.store(false);
    while(probe_lock.test_and_set(std::memory_order_acquire)){}
    probe={};probe_target.store(bytes);probe_lock.clear(std::memory_order_release);
    probe_enabled.store(bytes!=0,std::memory_order_release);
}
extern "C" void staging_allocation_probe_end(AllocationProbeSnapshot* result)
{
    probe_enabled.store(false,std::memory_order_release);
    while(probe_lock.test_and_set(std::memory_order_acquire)){}
    probe.remaining_ordinal=target_ordinal.load();
    if(result)*result=probe;
    probe_lock.clear(std::memory_order_release);
}
extern "C" void staging_allocation_probe_dump(int rank,const char* operation)
{
    AllocationProbeSnapshot result{};staging_allocation_probe_end(&result);
    char line[2048];
    int n=std::snprintf(line,sizeof(line),"ALLOCATION_ATTRIBUTION_SUMMARY operation=%s rank=%d matching=%llu library=%llu fixture=%llu other=%llu unknown=%llu library_with_helper=%llu legacy_rejected_library=%llu remaining_ordinal=%u\n",
        operation?operation:"",rank,(unsigned long long)result.matching,(unsigned long long)result.library,
        (unsigned long long)result.fixture,(unsigned long long)result.other,(unsigned long long)result.unknown,
        (unsigned long long)result.library_with_helper,(unsigned long long)result.legacy_rejected_library,result.remaining_ordinal);
    if(n>0)(void)write(STDERR_FILENO,line,std::min(std::size_t(n),sizeof(line)-1));
    for(unsigned i=0;i<result.record_count;++i) {
        const auto& r=result.records[i];const auto& a=r.attribution;
        n=std::snprintf(line,sizeof(line),"ALLOCATION_ATTRIBUTION_RECORD operation=%s rank=%d bytes=%zu owner=%u legacy_library=%u peeled_helpers=%u first_file=%s first_symbol=%s helper_file=%s owner_file=%s owner_symbol=%s\n",
            operation?operation:"",rank,r.bytes,unsigned(a.owner),a.legacy_library,a.peeled_helpers,
            a.first_file,a.first_symbol,a.helper_file,a.owner_file,a.owner_symbol);
        if(n>0)(void)write(STDERR_FILENO,line,std::min(std::size_t(n),sizeof(line)-1));
    }
}

extern "C" void staging_preparation_allocation_dump(int rank,const char* operation)
{
    AllocationProbeSnapshot result{};RelativeStack sites[16]{};MessageRecord messages[32]{};
    unsigned message_count=0;std::uint64_t message_total=0;
    while(probe_lock.test_and_set(std::memory_order_acquire)){}
    result=raw_probe;for(unsigned i=0;i<result.record_count;++i)sites[i]=raw_sites[i];
    message_count=raw_message_count;message_total=raw_message_total;
    for(unsigned i=0;i<message_count;++i)messages[i]=raw_messages[i];
    probe_lock.clear(std::memory_order_release);
    char line[2048];
    int n=std::snprintf(line,sizeof(line),"STAGING_RAW_ALLOCATION_SUMMARY operation=%s rank=%d total=%llu retained_records=%u\n",
        operation?operation:"",rank,(unsigned long long)result.matching,result.record_count);
    if(n>0)(void)write(STDERR_FILENO,line,std::min(std::size_t(n),sizeof(line)-1));
    for(unsigned i=0;i<result.record_count;++i){
        const auto& record=result.records[i];const auto& a=record.attribution;
        n=std::snprintf(line,sizeof(line),"STAGING_RAW_ALLOCATION_RECORD operation=%s rank=%d bytes=%zu owner=%u legacy_library=%u peeled_helpers=%u first_file=%s first_symbol=%s helper_file=%s owner_file=%s owner_symbol=%s owner_relativeOffset=0x%llx helper_relativeOffset=0x%llx\n",
            operation?operation:"",rank,record.bytes,unsigned(a.owner),a.legacy_library,a.peeled_helpers,
            a.first_file,a.first_symbol,a.helper_file,a.owner_file,a.owner_symbol,
            (unsigned long long)sites[i].owner_offset,(unsigned long long)sites[i].helper_offset);
        if(n>0)(void)write(STDERR_FILENO,line,std::min(std::size_t(n),sizeof(line)-1));
        for(unsigned f=0;f<sites[i].count;++f){const auto& frame=sites[i].frames[f];
            n=std::snprintf(line,sizeof(line),"STAGING_RAW_ALLOCATION_FRAME operation=%s rank=%d record=%u frame=%u relativeOffset=0x%llx file=%s symbol=%s\n",
                operation?operation:"",rank,i,f,(unsigned long long)frame.offset,frame.file,frame.symbol);
            if(n>0)(void)write(STDERR_FILENO,line,std::min(std::size_t(n),sizeof(line)-1));}
    }
    n=std::snprintf(line,sizeof(line),"STAGING_MPI_HIGHWATER_SUMMARY operation=%s rank=%d total=%llu retained_records=%u\n",
        operation?operation:"",rank,(unsigned long long)message_total,message_count);
    if(n>0)(void)write(STDERR_FILENO,line,std::min(std::size_t(n),sizeof(line)-1));
    for(unsigned i=0;i<message_count;++i){const auto& record=messages[i];
        n=std::snprintf(line,sizeof(line),"STAGING_MPI_HIGHWATER operation=%s rank=%d record=%u primitive=%s slot=%u bytes=%zu count=%zu type_width=%zu pointer_changed=%u owner_relativeOffset=0x%llx\n",
            operation?operation:"",rank,i,record.primitive,record.slot,record.bytes,record.count,record.width,record.pointer_changed,
            (unsigned long long)record.site.owner_offset);
        if(n>0)(void)write(STDERR_FILENO,line,std::min(std::size_t(n),sizeof(line)-1));
        for(unsigned f=0;f<record.site.count;++f){const auto& frame=record.site.frames[f];
            n=std::snprintf(line,sizeof(line),"STAGING_MPI_HIGHWATER_FRAME operation=%s rank=%d record=%u frame=%u relativeOffset=0x%llx file=%s symbol=%s\n",
                operation?operation:"",rank,i,f,(unsigned long long)frame.offset,frame.file,frame.symbol);
            if(n>0)(void)write(STDERR_FILENO,line,std::min(std::size_t(n),sizeof(line)-1));}
    }
}
