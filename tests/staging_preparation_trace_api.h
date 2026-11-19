#ifndef DDLA_STAGING_PREPARATION_TRACE_API_H
#define DDLA_STAGING_PREPARATION_TRACE_API_H
#include <cstddef>
struct StagingPreparationTrace {
    std::size_t host_allocations;
    std::size_t slot_bytes[2];
    unsigned messages;
    unsigned native_calls; // all intercepted calls while tracing, including host metadata
};
extern "C" {
void pblas_alloc_failure_arm_host_bytes(std::size_t bytes,unsigned ordinal);
void staging_preparation_trace_begin(void* slot0,void* slot1);
void staging_preparation_trace_end(StagingPreparationTrace* result);
}
#endif
