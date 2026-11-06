#ifndef DDLA_DC_THREADS_H
#define DDLA_DC_THREADS_H

// Host data parallelism for the secular-equation work of the distributed D&C (pbdsdc.cpp):
// std::thread fork/join with dynamic chunks, no build dependency beyond -pthread (already on the
// CXX line; precedent src/phb2st.cpp).  Workers run pure host loops: they never call MPI, CUDA,
// HIP or NCCL.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <memory>
#include <type_traits>
#include <utility>
#include <thread>
#include <vector>

#include "host_threads.h"

namespace ddla {
namespace detail {

// Prepared execution state. Construction and destruction may create/join
// workers; synchronous dispatch only borrows the caller's callback/context.
// No worker ever calls MPI or a backend runtime.
class DcHostTeam {
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable work_,finished_;
    std::atomic<std::size_t> next_{0};
    std::size_t generation_=0;
    int remaining_=0,active_=1,n_=0,grain_=1,chunks_=0;
    bool stopping_=false,fixed_roles_=false;
    void* context_=nullptr;
    void (*function_)(void*,int,int)=nullptr;
    std::exception_ptr failure_;

    void drain() noexcept {
        for(;;) {
            const auto c=next_.fetch_add(1,std::memory_order_relaxed);
            if(c>=static_cast<std::size_t>(chunks_))return;
            const int lo=static_cast<int>(c)*grain_;
            try {function_(context_,lo,lo+std::min(grain_,n_-lo));}
            catch(...) {std::lock_guard<std::mutex> lock(mutex_);if(!failure_)failure_=std::current_exception();}
        }
    }
    void worker(int index) noexcept {
        std::size_t seen=0;
        std::unique_lock<std::mutex> lock(mutex_);
        for(;;) {
            work_.wait(lock,[&]{return stopping_||seen!=generation_;});
            if(stopping_)return;
            seen=generation_;const bool active=index<active_;
            lock.unlock();
            if(active) {
                if(fixed_roles_) {
                    try{function_(context_,index,index+1);}catch(...){std::lock_guard<std::mutex> guard(mutex_);if(!failure_)failure_=std::current_exception();}
                } else drain();
            }
            lock.lock();
            if(--remaining_==0)finished_.notify_one();
        }
    }
    void stop() noexcept {
        {std::lock_guard<std::mutex> lock(mutex_);stopping_=true;}
        work_.notify_all();for(auto& worker:workers_)if(worker.joinable())worker.join();
    }
public:
    explicit DcHostTeam(int threads) {
        assert(threads>=1);workers_.reserve(std::size_t(threads-1));
        try {for(int i=1;i<threads;++i)workers_.emplace_back([this,i]{worker(i);});}
        catch(...) {stop();throw;}
    }
    ~DcHostTeam(){stop();}
    DcHostTeam(const DcHostTeam&)=delete;
    DcHostTeam& operator=(const DcHostTeam&)=delete;
    int size() const noexcept {return int(workers_.size())+1;}
    void run(int threads,int n,int grain,void* context,void (*function)(void*,int,int),bool fixed_roles=false) {
        if(n<=0)return;
        grain=std::max(1,grain);
        const int chunks=1+(n-1)/grain,active=std::min(threads,chunks);
        assert(threads>=1&&threads<=size());
        if(active<=1){function(context,0,n);return;}
        {
            std::lock_guard<std::mutex> lock(mutex_);
            assert(!remaining_);n_=n;grain_=grain;chunks_=chunks;active_=active;fixed_roles_=fixed_roles;
            next_.store(0,std::memory_order_relaxed);failure_=nullptr;
            context_=context;function_=function;remaining_=int(workers_.size());++generation_;
        }
        work_.notify_all();
        if(fixed_roles) {
            try{function(context,0,1);}catch(...){std::lock_guard<std::mutex> guard(mutex_);if(!failure_)failure_=std::current_exception();}
        } else drain();
        std::unique_lock<std::mutex> lock(mutex_);
        finished_.wait(lock,[&]{return remaining_==0;});
        context_=nullptr;function_=nullptr;
        auto failure=failure_;failure_=nullptr;lock.unlock();
        if(failure)std::rethrow_exception(failure);
    }
    // Stable logical roles: role zero executes on the caller (MPI-funneled
    // work stays there), and role i on the same prepared worker each time.
    // Barrier-using callbacks retain their own cooperative cancellation.
    void run_roles(int threads,void* context,void (*function)(void*,int)) {
        struct Callback {void* context;void (*function)(void*,int);};
        Callback callback{context,function};
        run(threads,threads,1,&callback,[](void* p,int role,int) {
            auto& callback=*static_cast<Callback*>(p);callback.function(callback.context,role);
        },true);
    }
    template<class F>void run_roles(int threads,F&& function) {
        using Callable=typename std::remove_reference<F>::type;
        run_roles(threads,const_cast<void*>(static_cast<const void*>(std::addressof(function))),
            [](void* p,int role){(*static_cast<Callable*>(p))(role);});
    }
    template<class F>void run(int threads,int n,int grain,F&& function) {
        using Callable=typename std::remove_reference<F>::type;
        run(threads,n,grain,const_cast<void*>(static_cast<const void*>(std::addressof(function))),
            [](void* context,int lo,int hi){(*static_cast<Callable*>(context))(lo,hi);});
    }
};

// Threads for one rank's host work: the allowed CPUs shared by the ranks of the node, at most 8.
// DDLA_SVD_DC_THREADS (>= 1) overrides.
inline int dc_host_threads(int ranks_on_node, int workers = 0)
{
    if (const char* env = std::getenv("DDLA_SVD_DC_THREADS")) {
        const int v = std::atoi(env);
        if (v >= 1) return std::min(v, 64);
    }
    if (workers < 1) workers = host_allowed_cpus() / std::max(1, ranks_on_node);
    return std::max(1, std::min(8, workers));
}

// f(begin, end) over [0, n) in dynamically scheduled chunks of `grain` elements, on up to `nthreads`
// threads INCLUDING the caller.  f runs concurrently: it may only touch disjoint output ranges.
template <typename F>
void dc_parallel_for(int nthreads, int n, int grain, F&& f)
{
    if (n <= 0) return;
    grain = std::max(1, grain);
    const int nchunks = (n + grain - 1) / grain;
    const int nt = std::min(nthreads, nchunks);
    if (nt <= 1) {
        f(0, n);
        return;
    }
    std::atomic<int> next(0);
    auto worker = [&]() {
        for (;;) {
            const int c = next.fetch_add(1, std::memory_order_relaxed);
            if (c >= nchunks) return;
            f(c * grain, std::min(n, (c + 1) * grain));
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(nt - 1));
    try {
        for (int t = 1; t < nt; ++t) pool.emplace_back(worker);
    } catch (...) {
        // fewer workers than planned: the calling thread drains the remaining chunks
    }
    worker();
    for (std::thread& t : pool) t.join();
}

// Transitional numerical helpers also serve the legacy owning solver. New
// borrowed entries always supply a nonnull prepared team.
template<class F>void dc_parallel_for(DcHostTeam* team,int threads,int n,int grain,F&& function)
{
    if(team)team->run(threads,n,grain,std::forward<F>(function));
    else dc_parallel_for(threads,n,grain,std::forward<F>(function));
}

} // namespace detail
} // namespace ddla

#endif // DDLA_DC_THREADS_H
