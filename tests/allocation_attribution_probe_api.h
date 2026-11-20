#ifndef DDLA_TEST_ALLOCATION_ATTRIBUTION_PROBE_API_H
#define DDLA_TEST_ALLOCATION_ATTRIBUTION_PROBE_API_H
#include <cstddef>
#include <cstdint>
// Test-only ABI. Fixed buffers keep attribution itself allocation-free.
enum class AllocationProbeOwner : unsigned { Unknown=0, Library=1, Fixture=2, Other=3 };
struct AllocationAttribution {
    AllocationProbeOwner owner;
    unsigned peeled_helpers;
    unsigned peeled_interposers;
    unsigned legacy_library;
    char first_file[192],first_symbol[192];
    char owner_file[192],owner_symbol[192];
    char helper_file[192];
};
struct AllocationProbeRecord {
    std::size_t bytes;
    AllocationAttribution attribution;
};
struct AllocationProbeSnapshot {
    std::uint64_t matching,library,fixture,other,unknown;
    std::uint64_t library_with_helper,legacy_rejected_library;
    unsigned remaining_ordinal,record_count;
    AllocationProbeRecord records[16];
};
extern "C" {
void staging_allocation_probe_begin(std::size_t target_bytes);
void staging_allocation_probe_end(AllocationProbeSnapshot* result);
void staging_allocation_probe_dump(int rank,const char* operation);
// Fixed-buffer wildcard capture of DDLA-owned allocations during a raw trace.
void staging_preparation_allocation_dump(int rank,const char* operation);
// Diagnostic getter must return current slot addresses without allocation or
// MPI (the fixture calls host_buffers<byte>(0,0)). Cleared at trace end.
void staging_preparation_trace_set_slot_getter(void* context,
    void (*getter)(void* context,void** slot0,void** slot1));
// The vendor diagnostic DSO forwards the REAL cuBLAS operation after this
// controlled allocation; it neither replaces numerical output nor mocks BLAS.
int staging_vendor_probe_prepare();
void staging_vendor_probe_arm(std::size_t target_bytes);
unsigned staging_vendor_probe_allocations();
unsigned staging_vendor_probe_forwards();
}
#endif
