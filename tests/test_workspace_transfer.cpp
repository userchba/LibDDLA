#include "api_grid_test_common.h"
#include "matrix_transfer.h"
#include "comm_traits.h"
#include "row_share.h"
#include "workspace_test_common.h"

#include <array>
#include <cstring>

using namespace api_grid_test;

namespace {
template <typename T> T value(int r,int c,int salt=0)
{ return T(1+r+31*c+salt); }
template <typename T> bool same(const std::vector<T>& a,const std::vector<T>& b)
{ return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0); }
template <typename Snapshot> bool no_allocation(const Snapshot& s)
{ return s.ddla_alloc==0 && s.ddla_free==0 && s.pinned_alloc==0 && s.raw_gpu_alloc==0 && s.host_new==0 && s.vendor_query==0; }
template <typename Snapshot> bool pure_query(const Snapshot& s)
{ return no_allocation(s) && s.copies==0 && s.launches==0 && s.syncs==0 && s.collectives==0 && s.resource_setup==0; }
// A tunnel build stages GPU messages in the handle's host buffers, which grow
// on first use (comm_traits.h): there the first run of a case may allocate.
template <DdlaBackend B,typename Snapshot> bool run_allocates_nothing(const Snapshot& s,int pass)
{
#if defined(DDLA_USE_GPU_CPU_TUNNEL)
    if(B==DdlaBackend::GPU && pass==0)return true;
#endif
    (void)pass;return no_allocation(s);
}

// Exactly the queried bytes of both kinds, surrounded by guards. The same
// addresses serve every execution in a case.
class GuardedScratch {
    static constexpr std::size_t guard=256;
    DdlaHandle_t h_;
    std::size_t backend_bytes_,host_bytes_;
    ddla::vector<char> backend_;
    std::vector<char> host_;
public:
    GuardedScratch(DdlaHandle_t h,std::size_t backend_bytes,std::size_t host_bytes)
        :h_(h),backend_bytes_(backend_bytes),host_bytes_(host_bytes),
         backend_(h,backend_bytes+2*guard),host_(host_bytes+2*guard,char(0x5a))
    { DDLA_CHECK(ddlaMemset(h,backend_.data(),0x5a,backend_.size())); }
    void* backend(){return backend_.data()+guard;}
    void* host(){return host_.data()+guard;}
    void verify()
    {
        std::array<char,2*guard> backend{};
        DDLA_CHECK(ddlaMemcpy(h_,backend.data(),backend_.data(),guard));
        DDLA_CHECK(ddlaMemcpy(h_,backend.data()+guard,backend_.data()+guard+backend_bytes_,guard));
        DDLA_CHECK(ddlaSynchronize(h_));
        bool okay=true;
        for(std::size_t i=0;i<guard;++i)
            okay=okay && host_[i]==char(0x5a) && host_[guard+host_bytes_+i]==char(0x5a)
                && backend[i]==char(0x5a) && backend[guard+i]==char(0x5a);
        require_close(h_,"transfer scratch exact-size guards",okay?0:1,0);
    }
};

template <DdlaBackend B,typename T>
void transfer_case(DdlaHandle_t h,int M,int N,int ia,int ja,int m,int n,int root)
{
    int pr,pc,r,c;ddlaGetGridDims(h,pr,pc);ddlaGetGridCoords(h,r,c);
    int desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(desc,h,M,N,3,5,pr-1,pc-1));desc[DDLA_LLD_]+=3;
    const T sentinel=T(-731);
    auto original=make_local<T>(h,desc,[](int i,int j){return value<T>(i,j);});
    for(int j=0;j<ddla_test::n_loc(h,desc);++j)
        for(int i=ddla_test::m_loc(h,desc);i<desc[DDLA_LLD_];++i)original[i+std::size_t(j)*desc[DDLA_LLD_]]=sentinel;
    DeviceBuffer<T> a(h,original.size());upload(h,a.ptr,original);
    const int ld=m+4;
    std::vector<T> host_window(h->myid==root?std::size_t(ld)*n:0,sentinel);
    DeviceBuffer<T> window(h,host_window.size());upload(h,window.ptr,host_window);
    matrix_impl::gather_check<B,T>(h,m,n,a.ptr,ia,ja,desc,root,window.ptr,ld);
    MPI_Barrier(h->comm);
    bool pure=true;
    if(h->myid==0) {
        // Only this rank queries: an accidental collective here deadlocks the
        // test. No data pointers are part of the sizing interface.
        workspace_test::Scope scope(true);
        std::size_t backend_bytes=0,host_bytes=0;
        matrix_impl::transfer_workspace_size<T>(h,m,n,ia,ja,desc,root,&backend_bytes,&host_bytes);
        pure=pure_query(scope.finish());
    }
    MPI_Barrier(h->comm);
    require_close(h,"one-rank transfer query is pure",pure?0:1,0);
    std::size_t backend_bytes=0,host_bytes=0;
    matrix_impl::transfer_workspace_size<T>(h,m,n,ia,ja,desc,root,&backend_bytes,&host_bytes);
    GuardedScratch scratch(h,backend_bytes,host_bytes);
    for(int pass=0;pass<2;++pass) {
        upload(h,a.ptr,original);
        workspace_test::Scope gather_scope;
        matrix_impl::gather_run<B,T>(h,m,n,a.ptr,ia,ja,desc,root,window.ptr,ld,
                                     scratch.backend(),backend_bytes,scratch.host(),host_bytes);
        const auto gather_snapshot=gather_scope.finish();
        require_close(h,"gather run performs no allocation/query",run_allocates_nothing<B>(gather_snapshot,pass)?0:1,0);
        check_ddla_sync(h);
        auto expected=host_window;
        if(h->myid==root)for(int j=0;j<n;++j)for(int i=0;i<m;++i)
            expected[i+std::size_t(j)*ld]=value<T>(ia-1+i,ja-1+j);
        require_close(h,"gather run order and root padding",same(download(h,window.ptr,expected.size()),expected)?0:1,0);
        if(h->myid==root)for(int j=0;j<n;++j)for(int i=0;i<m;++i)
            host_window[i+std::size_t(j)*ld]=value<T>(i,j,1000+pass);
        upload(h,window.ptr,host_window);
        workspace_test::Scope scatter_scope;
        matrix_impl::scatter_run<B,T>(h,m,n,a.ptr,ia,ja,desc,root,window.ptr,ld,true,
                                      scratch.backend(),backend_bytes,scratch.host(),host_bytes);
        const auto scatter_snapshot=scatter_scope.finish();
        require_close(h,"scatter run performs no allocation/query",run_allocates_nothing<B>(scatter_snapshot,pass)?0:1,0);
        check_ddla_sync(h);
        auto expected_a=original;
        for(int j=0;j<ddla_test::n_loc(h,desc);++j)for(int i=0;i<ddla_test::m_loc(h,desc);++i) {
            const int gr=indx_l2g_r(desc,h,i),gc=indx_l2g_c(desc,h,j);
            if(gr>=ia-1 && gr<ia-1+m && gc>=ja-1 && gc<ja-1+n)
                expected_a[i+std::size_t(j)*desc[DDLA_LLD_]]=value<T>(gr-ia+1,gc-ja+1,1000+pass);
        }
        require_close(h,"scatter run selection and LLD padding",same(download(h,a.ptr,expected_a.size()),expected_a)?0:1,0);
        scratch.verify();
    }
}

template <DdlaBackend B,typename T>
void transport_case(DdlaHandle_t h,char axis,char op)
{
    constexpr int M=19,N=17,mb=3,nb=5;
    const int rs=h->nprows_-1,cs=h->npcols_-1;
    int desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(desc,h,M,N,mb,nb,rs,cs));desc[DDLA_LLD_]+=3;
    auto source=make_local<T>(h,desc,[](int i,int j){return value<T>(i,j);});
    DeviceBuffer<T> input(h,source.size());upload(h,input.ptr,source);
    const int m=axis=='R'?2:M,n=axis=='C'?2:N;
    const int source_length=axis=='R'?num_loc(n,nb,h->mypcol_,cs,h->npcols_)
                                          :num_loc(m,mb,h->myprow_,rs,h->nprows_);
    const int target_length=op=='N'?source_length:
        (axis=='R'?num_loc(n,nb,h->myprow_,cs,h->nprows_)
                  :num_loc(m,mb,h->mypcol_,rs,h->npcols_));
    const std::size_t count=std::size_t(axis=='R'?m:n)*std::max(source_length,target_length);
    DeviceBuffer<T> output(h,count);
    transport_impl::transport_check<B>(h,axis,op,m,n,desc);
    MPI_Barrier(h->comm);
    bool pure=true;
    if(h->myid==0) {
        workspace_test::Scope scope(true);
        std::size_t backend_bytes=0,host_bytes=0;
        transport_impl::transport_workspace_size<T>(h,axis,op,m,n,desc,&backend_bytes,&host_bytes);
        pure=pure_query(scope.finish());
    }
    MPI_Barrier(h->comm);
    require_close(h,"one-rank transport query is pure",pure?0:1,0);
    std::size_t backend_bytes=0,host_bytes=0;
    transport_impl::transport_workspace_size<T>(h,axis,op,m,n,desc,&backend_bytes,&host_bytes);
    GuardedScratch scratch(h,backend_bytes,host_bytes);
    for(int pass=0;pass<2;++pass) {
        workspace_test::Scope scope;
        transport_impl::transport_run<B,T>(h,axis,op,m,n,input.ptr,0,0,desc,output.ptr,
                                           scratch.backend(),backend_bytes,scratch.host(),host_bytes);
        const auto snapshot=scope.finish();
        require_close(h,"transport run performs no allocation/query",run_allocates_nothing<B>(snapshot,pass)?0:1,0);
        check_ddla_sync(h);
        const auto got=download(h,output.ptr,std::size_t(axis=='R'?m:n)*target_length);
        double err=0;
        for(int j=0;j<target_length;++j) {
            const int global=axis=='R'
                ?indxl2g(j,nb,op=='N'?h->mypcol_:h->myprow_,cs,op=='N'?h->npcols_:h->nprows_)
                :indxl2g(j,mb,op=='N'?h->myprow_:h->mypcol_,rs,op=='N'?h->nprows_:h->npcols_);
            for(int i=0;i<(axis=='R'?m:n);++i) {
                const T expected=axis=='R'?value<T>(i,global):value<T>(global,i);
                const std::size_t offset=axis=='C' && op=='N'
                    ?std::size_t(j)+std::size_t(i)*target_length
                    :std::size_t(i)+std::size_t(j)*(axis=='R'?m:n);
                err=std::max(err,double(std::abs(got[offset]-expected)));
            }
        }
        require_close(h,"transport run exact rectangular redistribution",err,0);
        scratch.verify();
    }
}

template <DdlaBackend B,typename T>
void convert_case(DdlaHandle_t h)
{
    using R=typename real_type<T>::type;
    constexpr int rows=3,cols=5,ldd=rows+3;
    for(bool transpose:{false,true})for(bool zero:{false,true}) {
        const int lds=(transpose?cols:rows)+2,source_cols=transpose?rows:cols;
        std::vector<R> source(std::size_t(lds)*source_cols,R(-53));
        for(int c=0;c<source_cols;++c)for(int r=0;r<(transpose?cols:rows);++r)
            source[r+std::size_t(c)*lds]=value<R>(r,c);
        std::vector<T> expected(std::size_t(ldd)*cols,T(-719));
        DeviceBuffer<R> input(h,source.size());upload(h,input.ptr,source);
        DeviceBuffer<T> output(h,expected.size());upload(h,output.ptr,expected);
        for(int c=0;c<cols;++c)for(int r=0;r<rows;++r)
            expected[r+std::size_t(c)*ldd]=zero?T(0):T(value<R>(transpose?c:r,transpose?r:c));
        workspace_test::Scope scope;
        matrix_impl::convert_rows<B,T>(h,zero?nullptr:input.ptr,lds,output.ptr,ldd,rows,cols,transpose);
        const auto snapshot=scope.finish();
        require_close(h,"local conversion has no allocation/query/collective/sync",
            no_allocation(snapshot)&&snapshot.resource_setup==0&&snapshot.collectives==0&&snapshot.syncs==0?0:1,0);
        check_ddla_sync(h);
        require_close(h,"local conversion transpose/widen/zero/padding",
            same(download(h,output.ptr,expected.size()),expected)?0:1,0);
        require_close(h,"local conversion preserves source",
            same(download(h,input.ptr,source.size()),source)?0:1,0);
    }
    workspace_test::Scope scope;
    matrix_impl::convert_rows<B,T>(h,nullptr,1,nullptr,1,0,cols,false);
    matrix_impl::convert_rows<B,T>(h,nullptr,1,nullptr,1,rows,0,true);
    require_close(h,"empty local conversion is pure",pure_query(scope.finish())?0:1,0);
}

template <DdlaBackend B,typename T>
void rows_case(DdlaHandle_t h,int k,bool transpose)
{
    using R=typename real_type<T>::type;
    const int m=k+3,n=k+2,M=m+2,N=n+3;
    const detail::RowShare share(k,h->nprocs);
    const int ld=std::max(1,share.count(h->myid))+2;
    std::vector<R> source(std::size_t(ld)*k,R(-53));
    for(int j=0;j<k;++j)for(int i=0;i<share.count(h->myid);++i)
        source[i+std::size_t(j)*ld]=value<R>(share.lo(h->myid)+i,j);
    DeviceBuffer<R> input(h,source.size());upload(h,input.ptr,source);
    int desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(desc,h,M,N,2,3,h->nprows_-1,h->npcols_-1));desc[DDLA_LLD_]+=4;
    const auto count=local_size(h,desc);
    std::vector<T> original(count,T(-719));
    DeviceBuffer<T> output(h,count);upload(h,output.ptr,original);
    matrix_impl::scatter_rows_check<B,T>(h,k,input.ptr,ld,transpose,m,n,output.ptr,desc);
    require_close(h,"row-scatter check leaves output unchanged",same(download(h,output.ptr,count),original)?0:1,0);
    MPI_Barrier(h->comm);
    bool pure=true;
    if(h->myid==0) {
        workspace_test::Scope scope(true);
        std::size_t backend_bytes=0,host_bytes=0;
        matrix_impl::scatter_rows_workspace_size<T>(h,k,&backend_bytes,&host_bytes);
        pure=pure_query(scope.finish());
    }
    MPI_Barrier(h->comm);
    require_close(h,"one-rank row-scatter query is pure",pure?0:1,0);
    std::size_t backend_bytes=0,host_bytes=0;
    matrix_impl::scatter_rows_workspace_size<T>(h,k,&backend_bytes,&host_bytes);
    GuardedScratch scratch(h,backend_bytes,host_bytes);
    for(int pass=0;pass<2;++pass) {
        upload(h,output.ptr,original);
        workspace_test::Scope scope;
        matrix_impl::scatter_rows_run<B,T>(h,k,input.ptr,ld,transpose,m,n,output.ptr,desc,
                                           scratch.backend(),backend_bytes,scratch.host(),host_bytes);
        const auto snapshot=scope.finish();
        require_close(h,"row-scatter run performs no allocation/query",run_allocates_nothing<B>(snapshot,pass)?0:1,0);
        auto expected=original;
        for(int j=0;j<ddla_test::n_loc(h,desc);++j)for(int i=0;i<ddla_test::m_loc(h,desc);++i) {
            const int gr=indx_l2g_r(desc,h,i),gc=indx_l2g_c(desc,h,j);
            if(gr<m && gc<n)expected[i+std::size_t(j)*desc[DDLA_LLD_]]=
                gr<k && gc<k ? T(value<R>(transpose?gc:gr,transpose?gr:gc)) : T(0);
        }
        require_close(h,"row-scatter run transpose/widen/extension/padding",same(download(h,output.ptr,count),expected)?0:1,0);
        scratch.verify();
    }
}

template <DdlaBackend B> void shared_scratch_case(DdlaHandle_t h)
{
    if(h->nprocs<=1)return; // These one-rank paths use no scratch.
    using T=std::complex<double>;
    constexpr int k=5,m=7,n=6;
    const detail::RowShare share(k,h->nprocs);
    const int ld=std::max(1,share.count(h->myid));
    std::vector<double> slab(std::size_t(ld)*k);
    for(int j=0;j<k;++j)for(int i=0;i<share.count(h->myid);++i)
        slab[i+std::size_t(j)*ld]=value<double>(share.lo(h->myid)+i,j);
    DeviceBuffer<double> source_rows(h,slab.size());upload(h,source_rows.ptr,slab);
    int row_desc[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(row_desc,h,m+2,n+2,2,3,h->nprows_-1,h->npcols_-1));
    row_desc[DDLA_LLD_]+=3;
    std::vector<T> initial(local_size(h,row_desc),T(-811,23));
    DeviceBuffer<T> rows(h,initial.size());
    auto expected_rows=initial;
    for(int j=0;j<ddla_test::n_loc(h,row_desc);++j)for(int i=0;i<ddla_test::m_loc(h,row_desc);++i) {
        const int gr=indx_l2g_r(row_desc,h,i),gc=indx_l2g_c(row_desc,h,j);
        if(gr<m && gc<n)expected_rows[i+std::size_t(j)*row_desc[DDLA_LLD_]]=
            gr<k && gc<k ? T(value<double>(gr,gc)) : T(0);
    }
    constexpr int M=19,N=17,mb=3,nb=5;
    int float_desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(float_desc,h,M,N,mb,nb,0,0));
    auto host_float=make_local<float>(h,float_desc,[](int i,int j){return value<float>(i,j,71);});
    DeviceBuffer<float> floats(h,host_float.size());upload(h,floats.ptr,host_float);
    const int source_length=num_loc(M,mb,h->myprow_,0,h->nprows_);
    const int target_length=num_loc(M,mb,h->mypcol_,0,h->npcols_);
    DeviceBuffer<float> transported(h,std::size_t(2)*std::max(source_length,target_length));
    DeviceBuffer<float> gathered(h,h->myid==0?std::size_t(M)*N:0);
    std::vector<float> expected_gather(h->myid==0?std::size_t(M)*N:0);
    if(h->myid==0)for(int j=0;j<N;++j)for(int i=0;i<M;++i)
        expected_gather[i+std::size_t(j)*M]=value<float>(i,j,71);
    matrix_impl::scatter_rows_check<B,T>(h,k,source_rows.ptr,ld,false,m,n,rows.ptr,row_desc);
    transport_impl::transport_check<B>(h,'C','T',M,2,float_desc);
    matrix_impl::gather_check<B,float>(h,M,N,floats.ptr,1,1,float_desc,0,gathered.ptr,M);
    // One scratch, the largest of the three, serves them all.
    std::size_t row_backend=0,row_host=0,transport_backend=0,transport_host=0,gather_backend=0,gather_host=0;
    matrix_impl::scatter_rows_workspace_size<T>(h,k,&row_backend,&row_host);
    transport_impl::transport_workspace_size<float>(h,'C','T',M,2,float_desc,&transport_backend,&transport_host);
    matrix_impl::transfer_workspace_size<float>(h,M,N,1,1,float_desc,0,&gather_backend,&gather_host);
    GuardedScratch scratch(h,std::max({row_backend,transport_backend,gather_backend}),
                           std::max({row_host,transport_host,gather_host}));
    // Float transport/packing and metadata deliberately occupy the same bytes
    // between two complex row conversions.
    for(int pass=0;pass<2;++pass) {
        upload(h,rows.ptr,initial);
        workspace_test::Scope row_scope;
        matrix_impl::scatter_rows_run<B,T>(h,k,source_rows.ptr,ld,false,m,n,rows.ptr,row_desc,
                                           scratch.backend(),row_backend,scratch.host(),row_host);
        const auto row_snapshot=row_scope.finish();
        require_close(h,"shared scratch complex row run has no allocation",run_allocates_nothing<B>(row_snapshot,pass)?0:1,0);
        require_close(h,"shared scratch restores complex panels and padding",
            same(download(h,rows.ptr,expected_rows.size()),expected_rows)?0:1,0);
        workspace_test::Scope transport_scope;
        transport_impl::transport_run<B,float>(h,'C','T',M,2,floats.ptr,0,0,float_desc,transported.ptr,
                                               scratch.backend(),transport_backend,scratch.host(),transport_host);
        const auto transport_snapshot=transport_scope.finish();
        require_close(h,"shared scratch float transport has no allocation",run_allocates_nothing<B>(transport_snapshot,pass)?0:1,0);
        check_ddla_sync(h);
        const auto got=download(h,transported.ptr,std::size_t(2)*target_length);
        double err=0;
        for(int j=0;j<target_length;++j)for(int i=0;i<2;++i) {
            const int gr=indxl2g(j,mb,h->mypcol_,0,h->npcols_);
            err=std::max(err,double(std::abs(got[i+std::size_t(j)*2]-value<float>(gr,i,71))));
        }
        require_close(h,"shared scratch float transport values",err,0);
        workspace_test::Scope gather_scope;
        matrix_impl::gather_run<B,float>(h,M,N,floats.ptr,1,1,float_desc,0,gathered.ptr,M,
                                         scratch.backend(),gather_backend,scratch.host(),gather_host);
        const auto gather_snapshot=gather_scope.finish();
        require_close(h,"shared scratch float gather has no allocation",run_allocates_nothing<B>(gather_snapshot,pass)?0:1,0);
        check_ddla_sync(h);
        require_close(h,"shared scratch float gather values",
            same(download(h,gathered.ptr,expected_gather.size()),expected_gather)?0:1,0);
        scratch.verify();
    }
}

template <DdlaBackend B> void run(MPI_Comm comm,int pr,int pc,char major)
{
    DdlaHandle_t h=nullptr;ddlaInit(h,B);ddlaSet(h,comm,pr,pc,major);
    shared_scratch_case<B>(h);
#define CASES(T) \
    convert_case<B,T>(h); \
    transfer_case<B,T>(h,19,17,2,3,15,13,h->nprocs-1); \
    transfer_case<B,T>(h,2,1,1,1,2,1,0); \
    rows_case<B,T>(h,7,false); rows_case<B,T>(h,7,true); \
    rows_case<B,T>(h,1,false); rows_case<B,T>(h,0,true); \
    transport_case<B,T>(h,'R','N'); transport_case<B,T>(h,'C','N'); \
    transport_case<B,T>(h,'R','T'); transport_case<B,T>(h,'C','T')
    CASES(float);CASES(double);CASES(std::complex<float>);CASES(std::complex<double>);
#undef CASES
    if(h->myid==0)std::cout<<"workspace transfer PASS backend="<<int(B)<<" grid="<<pr<<"x"<<pc<<" major="<<major<<std::endl;
    ddlaDestroy(h);
}
}

int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);int size;MPI_Comm_size(MPI_COMM_WORLD,&size);
    TestOptions options;std::string error;
    if(!parse_options(argc,argv,size,options,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}
    for(auto grid:options.grids)for(char major:{'R','C'}) {
#if DDLA_HAS_CPU
        run<DdlaBackend::CPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#endif
#if DDLA_HAS_GPU
        run<DdlaBackend::GPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#endif
    }
    MPI_Finalize();return 0;
}
