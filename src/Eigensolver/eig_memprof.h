#ifndef DDLA_EIG_MEMPROF_H
#define DDLA_EIG_MEMPROF_H

// Per-stage memory profile of the eigensolver driver (DDLA_EIG_PROFILE=1), one PHEEVDMEM line per stage
// boundary and rank, next to the PHEEVDPROF timing line:
//
//   PHEEVDMEM myid=R n=N phase=<stage just finished> dev_peak_MB= dev_end_MB= dev_base_MB= rss_MB= hwm_MB= shm_MB=
//
//   dev_*  device memory IN USE on this rank's GPU (total - free from cudaMemGetInfo/hipMemGetInfo): that
//          counts everything -- the caller's matrices, the library's allocations, cuBLAS/NCCL workspaces and
//          the context itself -- not just ddlaMalloc.  dev_peak is the maximum seen during the stage
//          (a helper thread samples every ~0.5 ms, so buffers that live for microseconds can be missed;
//          the big ones this profile exists for live for the whole stage), dev_end the value at the boundary,
//          dev_base the value when the profile started (A and Z are already resident then).
//   rss/hwm/shm  host resident set, its high-water mark and the shared-memory part of it (the node-shared
//          chase window shows up in shm) from /proc/self/status, in MB.
//
// Off (the default) the object does nothing: no thread, no syscalls.  The profile adds no synchronization to
// the solver; the sampler only queries free memory.

#include "ddla_connector.h"
#include "ddla_stream.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <array>
#include <mutex>

namespace ddla {
namespace detail {

struct EigHostMem {
    long rss_kb = 0, hwm_kb = 0, shmem_kb = 0;
};

inline EigHostMem eig_read_host_mem()
{
    EigHostMem m;
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (f == nullptr) return m;
    char line[256];
    while (std::fgets(line, sizeof(line), f) != nullptr) {
        long v = 0;
        if (std::sscanf(line, "VmRSS: %ld kB", &v) == 1) m.rss_kb = v;
        else if (std::sscanf(line, "VmHWM: %ld kB", &v) == 1) m.hwm_kb = v;
        else if (std::sscanf(line, "RssShmem: %ld kB", &v) == 1) m.shmem_kb = v;
    }
    std::fclose(f);
    return m;
}

class EigMemProfile {
public:
    EigMemProfile(const DdlaHandle_t& handle, bool enabled, bool gpu)
        : handle_(handle), enabled_(enabled), gpu_(gpu && enabled)
    {
        if (!enabled_) return;
        status_=std::fopen("/proc/self/status","r");
        if(status_&&std::setvbuf(status_,status_buffer_.data(),_IOFBF,status_buffer_.size())){std::fclose(status_);status_=nullptr;}
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
        if(gpu_)try{sampler_=std::thread([this]{sample_loop();});}catch(...){if(status_){std::fclose(status_);status_=nullptr;}throw;}
#endif
        begin_run();
    }
    EigMemProfile(const EigMemProfile&) = delete;
    EigMemProfile& operator=(const EigMemProfile&) = delete;
    ~EigMemProfile(){stop();if(status_)std::fclose(status_);}

    // Persistent sampler preparation is outside numerical execution. A raw
    // caller may reset its marks/baseline for compatible repeated calls; the
    // public driver begins before workspace/resource allocation instead.
    void begin_run() {
        if(!enabled_)return;
        std::lock_guard<std::mutex> lock(sample_mutex_);
        active_.store(false,std::memory_order_release);generation_.fetch_add(1,std::memory_order_acq_rel);recs_count_=0;const auto used=gpu_?used_now():0;base_used_=used;cur_peak_.store(used);active_.store(true,std::memory_order_release);
    }
    bool running() const noexcept{return enabled_&&active_.load(std::memory_order_acquire);}
    void pause_run() noexcept{active_.store(false,std::memory_order_release);}
    void report_timing(int rank,int n,int band) const {
        if(!enabled_||recs_count_<2)return;char text[1024];int length=std::snprintf(text,sizeof(text),"PHEEVDPROF myid=%d n=%d nbw=%d",rank,n,band);
        for(std::size_t i=1;i<recs_count_&&length>0&&std::size_t(length)<sizeof(text);++i)
            length+=std::snprintf(text+length,sizeof(text)-std::size_t(length)," %s=%.3f",recs_[i].name,recs_[i].time-recs_[i-1].time);
        if(length>0&&std::size_t(length)<sizeof(text))std::snprintf(text+length,sizeof(text)-std::size_t(length)," total=%.3f",recs_[recs_count_-1].time-recs_[0].time);
        std::printf("%s\n",text);
    }

    /// A stage just finished: record its peak/end and start the next stage's peak at the current value.
    void phase(const char* name)
    {
        if (!enabled_||recs_count_==recs_.size()) return;
        Rec r;r.time=MPI_Wtime();
        r.name = name;
        if (gpu_) {
            const std::size_t u = used_now();
            r.dev_end = u;
            r.dev_peak = std::max(cur_peak_.exchange(u), u);
        }
        r.host=read_host_mem();
        recs_[recs_count_++]=r;
    }

    /// One line per recorded boundary (the first, "start", carries the baseline).
    void report(int myid, int n)
    {
        if (!enabled_) return;
        pause_run();
        constexpr double MB = 1.0 / (1024.0 * 1024.0);
        for(std::size_t i=0;i<recs_count_;++i){const auto& r=recs_[i];
            std::printf("PHEEVDMEM myid=%d n=%d phase=%s dev_peak_MB=%.0f dev_end_MB=%.0f dev_base_MB=%.0f "
                        "rss_MB=%.0f hwm_MB=%.0f shm_MB=%.0f\n",
                        myid, n, r.name, static_cast<double>(r.dev_peak) * MB,
                        static_cast<double>(r.dev_end) * MB, static_cast<double>(base_used_) * MB,
                        r.host.rss_kb / 1024.0, r.host.hwm_kb / 1024.0, r.host.shmem_kb / 1024.0);
        }
        std::fflush(stdout);
        recs_count_=0;
    }

private:
    EigHostMem read_host_mem(){
        EigHostMem result;if(!status_)return result;std::clearerr(status_);std::rewind(status_);char line[256];
        while(std::fgets(line,sizeof(line),status_)){
            if(!std::strncmp(line,"VmRSS:",6))result.rss_kb=std::strtol(line+6,nullptr,10);
            else if(!std::strncmp(line,"VmHWM:",6))result.hwm_kb=std::strtol(line+6,nullptr,10);
            else if(!std::strncmp(line,"RssShmem:",9))result.shmem_kb=std::strtol(line+9,nullptr,10);
        }return result;
    }
    struct Rec {
        const char* name=nullptr;double time=0;
        std::size_t dev_peak = 0, dev_end = 0;
        EigHostMem host;
    };

    void stop()
    {
        stop_.store(true);
        if (sampler_.joinable()) sampler_.join();
    }

    // used = total - free; a failed query reports the last good value (never throws: runs on a helper thread).
    // Builds without a GPU backend have nothing to query (gpu_ is never set there).
    std::size_t used_now()
    {
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
        std::size_t f = 0, t = 0;
        if (ddlaMemGetInfo(handle_, &f, &t) == ddlaStatus_t::DDLA_STATUS_SUCCESS)
            last_used_.store(t-f,std::memory_order_relaxed);
#endif
        return last_used_.load(std::memory_order_relaxed);
    }
    std::atomic<std::size_t> last_used_{0};

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
    void sample_loop()
    {
        // a new thread starts on device 0: select this rank's device first
        if (ddlaSetDevice(handle_) != ddlaStatus_t::DDLA_STATUS_SUCCESS) return;
        std::size_t seen = 0;
        while (!stop_.load()) {
            std::size_t f = 0, t = 0;
            const auto generation=generation_.load(std::memory_order_acquire);
            if(active_.load(std::memory_order_acquire)&&ddlaMemGetInfo(handle_, &f, &t) == ddlaStatus_t::DDLA_STATUS_SUCCESS) {
                seen = t - f;
                // A sample begun before reset must not update the next run's
                // peak, even when the reset baseline has the same value.
                std::lock_guard<std::mutex> lock(sample_mutex_);
                std::size_t cur = cur_peak_.load();
                while(active_.load(std::memory_order_acquire)&&generation_.load(std::memory_order_acquire)==generation&&seen>cur&&!cur_peak_.compare_exchange_weak(cur,seen)){}
            }
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }
#endif

    std::FILE* status_=nullptr;std::array<char,4096> status_buffer_{};
    DdlaHandle_t handle_;
    bool enabled_ = false;
    bool gpu_ = false;
    std::size_t base_used_ = 0;
    std::atomic<std::size_t> cur_peak_{0};
    std::atomic<bool> stop_{false},active_{false};std::atomic<std::size_t> generation_{0};
    std::mutex sample_mutex_;
    std::thread sampler_;
    std::array<Rec,32> recs_{};std::size_t recs_count_=0;
};

} // namespace detail
} // namespace ddla

#endif // DDLA_EIG_MEMPROF_H
