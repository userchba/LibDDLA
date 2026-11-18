#ifndef DDLA_PGEADD_QUEUE_TRACE_API_H
#define DDLA_PGEADD_QUEUE_TRACE_API_H
#include <stdint.h>
typedef struct pgeadd_queue_snapshot {
    uint64_t uploads, records, waits, drains, wrong_queue, unretired, record_faults;
    uint64_t event_creates, event_destroys, stream_creates, stream_destroys, create_faults;
} pgeadd_queue_snapshot;
#ifdef __cplusplus
extern "C" {
#endif
// A null event observes the first event recorded on the selected queue.
void pgeadd_queue_trace_begin(const void* host, uint64_t bytes, void* queue, void* event, int fail_record);
void pgeadd_queue_trace_end(pgeadd_queue_snapshot*);
void pgeadd_queue_trace_fail_create(int ordinal);
void pgeadd_queue_trace_setup_snapshot(pgeadd_queue_snapshot*);
#ifdef __cplusplus
}
#endif
#endif
