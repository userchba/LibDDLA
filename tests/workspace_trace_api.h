#ifndef DDLA_WORKSPACE_TRACE_API_H
#define DDLA_WORKSPACE_TRACE_API_H

#include <stdint.h>

// Diagnostic-preload ABI only. This header introduces no production state.
typedef struct workspace_trace_snapshot {
    uint64_t ddla_alloc;
    uint64_t ddla_free;
    uint64_t pinned_alloc;
    uint64_t raw_gpu_alloc;
    uint64_t host_new;
    uint64_t vendor_query;
    uint64_t copies;
    uint64_t launches;
    uint64_t syncs;
    uint64_t collectives;
    uint64_t resource_setup;
} workspace_trace_snapshot;

#ifdef __cplusplus
extern "C" {
#endif
void workspace_trace_begin(int query);
void workspace_trace_end(workspace_trace_snapshot* result);
// Forward the first matching real ddlaMemcpy, then return INTERNAL_ERROR.
// Arm with a nonnull exact host source and byte count; disarm with nullptr.
void workspace_trace_fail_memcpy_after_upload(const void* source,uint64_t bytes);
uint64_t workspace_trace_memcpy_faults(void);
#ifdef __cplusplus
}
#endif
#endif
