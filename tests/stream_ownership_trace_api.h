#ifndef DDLA_STREAM_OWNERSHIP_TRACE_API_H
#define DDLA_STREAM_OWNERSHIP_TRACE_API_H
#include <stdint.h>

// Test-preload ABI; never linked into the production library.
struct stream_ownership_snapshot {
    uint64_t creates;
    uint64_t destroys;
    uint64_t synchronizes;
    uint64_t default_synchronizes;
    uint64_t default_destroys;
    uint64_t blas_binds;
    uint64_t solver_binds;
    uint64_t blas_faults;
    uint64_t solver_faults;
};
enum stream_ownership_fault { StreamNoFault, StreamBlasFault, StreamSolverFault };
extern "C" {
void stream_ownership_trace_reset();
void stream_ownership_trace_snapshot(stream_ownership_snapshot*);
void stream_ownership_trace_fail_once(int fault);
}
#endif
