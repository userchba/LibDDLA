#include "api_grid_test_common.h"
#include "ddla_stream.h"
#include "mpi_datatype.h"
#include <cstring>
#include <dlfcn.h>
#include <tuple>

using namespace ddla;
using namespace api_grid_test;

#define DECLARE_Q(P, G, T) \
    void P##G##lq_(const int*, const int*, const int*, T*, const int*, const T*, T*, const int*, int*); \
    void P##G##rq_(const int*, const int*, const int*, T*, const int*, const T*, T*, const int*, int*)
extern "C" {
DECLARE_Q(s, org, float);
DECLARE_Q(d, org, double);
DECLARE_Q(c, ung, std::complex<float>);
DECLARE_Q(z, ung, std::complex<double>);
}
#undef DECLARE_Q

namespace {
#define NATIVE_Q(P, G, T) \
    void native_q(bool rq, int n, int k, T* a, const T* tau, T* work, int size, int& info) \
    { \
        if (rq) P##G##rq_(&n, &n, &k, a, &n, tau, work, &size, &info); \
        else P##G##lq_(&n, &n, &k, a, &n, tau, work, &size, &info); \
    }
NATIVE_Q(s, org, float)
NATIVE_Q(d, org, double)
NATIVE_Q(c, ung, std::complex<float>)
NATIVE_Q(z, ung, std::complex<double>)
#undef NATIVE_Q

template<class T> T conjugate(T x)
{
    if constexpr (std::is_arithmetic_v<T>) return x;
    else return std::conj(x);
}
template<class T> T sample(int i, int j)
{
    using R = typename real_type<T>::type;
    const R a = R(((i + 3) * (j + 7) * 13 % 47) - 23) / R(31);
    if constexpr (std::is_arithmetic_v<T>) return a + (i == j ? R(2) : R(0));
    else return T(a + (i == j ? R(2) : R(0)), R(((i + 5) * (j + 2) * 7 % 37) - 18) / R(29));
}
auto topology(const DdlaHandle_t& h)
{
    return std::make_tuple(h->nprows_, h->npcols_, h->myprow_, h->mypcol_, h->major,
        h->comm, h->row_comm, h->col_comm, h->nccl_comm, h->nccl_row_comm, h->nccl_col_comm,
        h->stream, h->blasH, h->solverH, h->host_threads);
}
template<class F> F symbol(const char* name)
{
    void* p = dlsym(RTLD_DEFAULT, name);
    F result = nullptr;
    static_assert(sizeof(p) == sizeof(result), "dlsym ABI");
    std::memcpy(&result, &p, sizeof(p));
    if (!result) throw std::runtime_error(std::string("Missing test hook: ") + name);
    return result;
}

template<class T>
void oracle(const DdlaHandle_t& h, bool rq, int m, int n, const std::vector<T>& a, const std::vector<T>& tau)
{
    RootSection root(h);
    if (root.root) {
        const int k = std::min(m, n);
        std::vector<T> q(std::size_t(n) * n, T(0));
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < n; ++j)
                q[(rq ? n - k + i : i) + std::size_t(j) * n] = a[(rq ? m - k + i : i) + std::size_t(j) * m];
        T optimal{};
        int info = 0;
        native_q(rq, n, k, q.data(), tau.data(), &optimal, -1, info);
        if (info) throw std::runtime_error("LAPACK Q query failed");
        const int count = std::max(1, int(std::real(optimal)));
        std::vector<T> work(count);
        native_q(rq, n, k, q.data(), tau.data(), work.data(), count, info);
        if (info) throw std::runtime_error("LAPACK Q generation failed");
        double residual = 0, orthogonality = 0;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                T sum = 0;
                for (int l = 0; l < n; ++l)
                    if (rq ? l >= n - m + i : l <= i) sum += a[i + std::size_t(l) * m] * q[l + std::size_t(j) * n];
                const double error = double(std::abs(sum - sample<T>(i, j))) / (1 + double(std::abs(sample<T>(i, j))));
                residual = std::max(residual, std::isfinite(error) ? error : std::numeric_limits<double>::infinity());
            }
            for (int i = 0; i < n; ++i) {
                T sum = 0;
                for (int l = 0; l < n; ++l) sum += q[i + std::size_t(l) * n] * conjugate(q[j + std::size_t(l) * n]);
                const double error = double(std::abs(sum - T(i == j)));
                orthogonality = std::max(orthogonality, std::isfinite(error) ? error : std::numeric_limits<double>::infinity());
            }
        }
        const double tol = sizeof(typename real_type<T>::type) == 4 ? 3e-4 : 2e-11;
        root.check(rq ? "RQ reconstruction" : "LQ reconstruction", residual, tol);
        root.check("row-factor Q orthogonality", orthogonality, tol);
    }
    root.finish();
}

template<class T> void local_case(const DdlaHandle_t& h, bool rq, int m, int n)
{
    const T guard = T(-12345);
    const int lda = m + 3, k = std::min(m, n);
    std::vector<T> a(4 + std::size_t(lda) * (n + 2) + 4, guard), tau(k + 8, guard);
    for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) a[4 + i + std::size_t(j) * lda] = sample<T>(i,j);
    DeviceBuffer<T> matrix(h, a.size()), factors(h, tau.size());
    upload(h, matrix.ptr, a); upload(h, factors.ptr, tau);
    if (rq) gerqf(h, m, n, matrix.ptr + 4, lda, factors.ptr + 4);
    else gelqf(h, m, n, matrix.ptr + 4, lda, factors.ptr + 4);
    auto result = download(h, matrix.ptr, a.size()); tau = download(h, factors.ptr, tau.size());
    double bad = 0;
    std::vector<T> full(std::size_t(m) * n);
    for (std::size_t v = 0; v < a.size(); ++v) {
        const auto t = v >= 4 ? v - 4 : std::size_t(lda) * (n+2);
        const int i = int(t % lda), j = int(t / lda);
        if (i < m && j < n) full[i + std::size_t(j) * m] = result[v];
        else if (result[v] != a[v]) bad = 1;
    }
    for (int i = 0; i < 4; ++i) if (tau[i] != guard || tau[k + 4 + i] != guard) bad = 1;
    require_close(h, "local row-factor padding/guards", bad, 0);
    std::vector<T> compact(tau.begin() + 4, tau.begin() + 4 + k);
    oracle(h, rq, m, n, full, compact);
}

template<class T> void distributed_case(const DdlaHandle_t& h, bool rq, int m, int n,
                                       bool shifted, int fault_skip = -1, bool memory = false)
{
    const T guard = T(-12345);
    const bool large_block = std::getenv("DDLA_ROW_LARGE_BLOCK") != nullptr;
    const int mb = large_block ? std::max(m, 1) : memory ? 16 : 3;
    const int nb = memory ? 16 : 4;
    const int rsrc = shifted ? h->nprows_ - 1 : 0, csrc = shifted ? h->npcols_ - 1 : 0;
    int d[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(d, h, m + 3, n + 4, mb, nb, rsrc, csrc));
    d[DDLA_LLD_] += 2;
    const int ml = num_loc(m, mb, h->myprow_, rsrc, h->nprows_);
    const int nl = num_loc(n, nb, h->mypcol_, csrc, h->npcols_);
    auto values = make_local<T>(h, d, [&](int i, int j) { return i < m && j < n ? sample<T>(i,j) : guard; });
    for (int j = 0; j < ddla_test::n_loc(h,d); ++j)
        for (int i = ddla_test::m_loc(h,d); i < d[DDLA_LLD_]; ++i) values[i + std::size_t(j)*d[DDLA_LLD_]] = guard;
    std::vector<T> input(4 + values.size() + 4, guard);std::copy(values.begin(), values.end(), input.begin() + 4);
    DeviceBuffer<T> matrix(h, input.size());upload(h, matrix.ptr, input);check_ddla_sync(h);
    std::vector<T> tau(ml + 8, guard);
    const int k = std::min(m,n), first = rq ? m-k : 0;
    const int lo = num_loc(first, mb, h->myprow_, rsrc, h->nprows_);
    const int hi = num_loc(first+k, mb, h->myprow_, rsrc, h->nprows_);
    T* const local_tau = hi > lo ? tau.data() + 4 : nullptr;
    const auto saved = topology(h);
    int info = -1;
    auto call = [&] {
        T* data = ml > 0 && nl > 0 ? matrix.ptr + 4 : nullptr;
        if (rq) pgerqf(h,m,n,data,d,local_tau,info);
        else pgelqf(h,m,n,data,d,local_tau,info);
    };
    if (fault_skip >= 0) {
        auto warm = symbol<int(*)()>("pblas_alloc_failure_prewarm");
        auto arm = symbol<void(*)(int,unsigned)>("row_factor_fail_after");
        auto disarm = symbol<void(*)()>("pblas_alloc_failure_disarm");
        auto fired = symbol<unsigned(*)()>("pblas_alloc_failure_fired");
        if (!warm()) throw std::runtime_error("Allocation hook prewarm failed");
        const int target = h->nprocs - 1;
        if (h->myid == target) arm(1, unsigned(fault_skip));
        bool caught = false;
        try { call(); } catch (const std::exception&) { caught = true; }
        disarm();
        require_close(h,"row-factor allocation failure is collective",caught?0:1,0);
        require_close(h,"row-factor allocation hook fired",fired() == unsigned(h->myid==target)?0:1,0);
        require_close(h,"row-factor topology restored after failure",saved==topology(h)?0:1,0);
        const auto unchanged = download(h,matrix.ptr,input.size());
        require_close(h,"row-factor failure preserves input",std::memcmp(unchanged.data(),input.data(),input.size()*sizeof(T))?1:0,0);
    }
    if (memory) symbol<void(*)()>("row_factor_memory_begin")();
    call();
    require_close(h,"row-factor success and restored topology",info==0 && saved==topology(h)?0:1,0);
    if (memory) {
        auto peak = symbol<std::size_t(*)()>("row_factor_memory_peak")();
        auto largest = symbol<std::size_t(*)()>("row_factor_memory_largest")();
        symbol<void(*)()>("row_factor_memory_end")();
        std::cout << "ROW_FACTOR_MEMORY rq=" << rq << " rank=" << h->myid << " ranks=" << h->nprocs
                  << " grid=" << h->nprows_ << 'x' << h->npcols_ << " major=" << h->major << " mb=" << mb
                  << " local_matrix_bytes=" << std::size_t(ml)*nl*sizeof(T) << " managed_peak=" << peak << " largest=" << largest << '\n';
        return;
    }
    const auto out = download(h,matrix.ptr,input.size());
    double bad = 0;
    std::vector<T> full(std::size_t(m)*n,T(0)), compact(k,T(0));
    for (std::size_t v=0; v<out.size(); ++v) {
        if (v<4 || v>=values.size()+4) { if(out[v]!=guard)bad=1;continue; }
        const auto t=v-4;const int li=int(t%d[DDLA_LLD_]),lj=int(t/d[DDLA_LLD_]);
        const int i=li<ddla_test::m_loc(h,d)?indx_l2g_r(d,h,li):m+3,j=indx_l2g_c(d,h,lj);
        if(i<m && j<n)full[i+std::size_t(j)*m]=out[v];
        else if(out[v]!=input[v])bad=1;
    }
    for(int t=0;t<int(tau.size());++t)if((t<lo+4 || t>=hi+4) && tau[t]!=guard)bad=1;
    require_close(h,"distributed row-factor parent/padding/tau guards",bad,0);
    for(int t=0;t<k;++t)if(h->mypcol_==0 && indxg2p(first+t,mb,rsrc,h->nprows_)==h->myprow_)
        compact[t]=tau[4+num_loc(first+t,mb,h->myprow_,rsrc,h->nprows_)];
    if(!full.empty())MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,full.data(),int(full.size()),detail::mpi_datatype<T>(),MPI_SUM,h->comm));
    if(k)MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,compact.data(),k,detail::mpi_datatype<T>(),MPI_SUM,h->comm));
    if(k)oracle(h,rq,m,n,full,compact);
}

template<class T> void cases(const DdlaHandle_t& h)
{
    for(bool rq:{false,true}) {
        for(auto shape:{std::pair<int,int>{5,17},{17,5},{9,9},{1,7},{7,1}}) {
            local_case<T>(h,rq,shape.first,shape.second);
            for(bool shifted:{false,true})distributed_case<T>(h,rq,shape.first,shape.second,shifted);
        }
        setenv("DDLA_ROW_LARGE_BLOCK", "1", 1);
        distributed_case<T>(h,rq,73,81,true);
        unsetenv("DDLA_ROW_LARGE_BLOCK");
        distributed_case<T>(h,rq,0,7,false);
        distributed_case<T>(h,rq,7,0,false);
        if(rq)gerqf<T>(h,0,7,nullptr,0,nullptr);else gelqf<T>(h,7,0,nullptr,0,nullptr);
        expect_invalid(h,"local row-factor negative size",[&]{if(rq)gerqf<T>(h,-1,2,nullptr,1,nullptr);else gelqf<T>(h,2,-1,nullptr,1,nullptr);});
    }
}
} // namespace

int main(int argc,char** argv)
{
    ddla_test_mpi_init(&argc,&argv);
    int rank=0,size=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    const bool fault=argc>1 && std::string(argv[1])=="--fault";
    const bool memory=argc>1 && std::string(argv[1])=="--memory";
    const int argument=argc>2?std::atoi(argv[2]):0;
    try {
        for(int rows=1;rows<=size;++rows)if(size%rows==0)for(char major:{'R','C'}) {
            DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,ddla_test::execution_backend));
            if(std::getenv("DDLA_ROW_HOST_STAGED") && h->backend==DdlaBackend::GPU)
                DDLA_CHECK(ddlaSetCommunication(h,DdlaCommunication::HostStagedMPI));
            DDLA_CHECK(ddlaSet(h,MPI_COMM_WORLD,rows,size/rows,major));
            if(fault) {
                setenv("DDLA_QR_PANEL","inplace",1);
                for(bool rq:{false,true})distributed_case<double>(h,rq,17,13,true,argument);
            } else if(memory) {
                setenv("DDLA_QR_PANEL","gather",1);
                for(bool rq:{false,true})distributed_case<double>(h,rq,argument,argument,false,-1,true);
            } else {
                for(const char* panel:{"gather","inplace"}) {
                    setenv("DDLA_QR_PANEL",panel,1);
                    cases<float>(h);cases<double>(h);cases<std::complex<float>>(h);cases<std::complex<double>>(h);
                }
            }
            DDLA_CHECK(ddlaDestroy(h));
        }
        if(rank==0)std::cout<<"ROW_FACTORIZATIONS_PASS ranks="<<size<<" fault="<<fault<<" memory="<<memory<<std::endl;
        MPI_Finalize();return 0;
    } catch(const std::exception& e) {
        std::cerr<<"row-factor test rank "<<rank<<": "<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);return 1;
    }
}
