#include "api_grid_test_common.h"
#include "PBLAS/ptrtrm_impl.h"
#include <ddla/ddla_vector.h>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>

using namespace ddla;
namespace {

template <typename T> T scalar(double re, double im = 0) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) return T(re);
    else return T(re, im);
}
template <typename T> T value(int i, int j, int salt) {
    return scalar<T>(((i*17+j*11+salt*7)%31-15)/16.0,
                     ((i*7+j*19+salt*13)%29-14)/16.0);
}
template <typename T> std::complex<double> wide(T v) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) return {double(v), 0};
    else return {double(v.real()), double(v.imag())};
}

// Same harness as the ptrmm grid test: padded descriptors exercise the
// leading-block contract, NaN poison guards every element the routine must
// not read or write, and the reference is an independent host loop.
template <typename T> struct Matrix {
    DdlaHandle_t h;
    int d[DDLA_DLEN_], nr, nc, pr, pc, r, c;
    std::vector<T> host;
    ddla::vector<T> data;
    Matrix(DdlaHandle_t handle, int m, int n, int mb, int nb, int rs, int cs)
        : h(handle), data(h) {
        ddlaGetGridDims(h, pr, pc); ddlaGetGridCoords(h, r, c);
        DDLA_CHECK(ddlaDescInit(d, h, m+3, n+5, mb, nb, rs, cs,
                               num_loc(m+3, mb, r, rs, pr)+3));
        nr = num_loc(m+3, mb, r, rs, pr); nc = num_loc(n+5, nb, c, cs, pc);
        host.assign(std::max(std::size_t(1), std::size_t(d[DDLA_LLD_])*nc), scalar<T>(-987.25, 654.5));
        data.resize(host.size());
    }
    int gi(int i) const { return indxl2g(i, d[DDLA_MB_], r, d[DDLA_RSRC_], pr); }
    int gj(int j) const { return indxl2g(j, d[DDLA_NB_], c, d[DDLA_CSRC_], pc); }
    void upload() { DDLA_CHECK(ddlaMemcpy(h, data.data(), host.data(), host.size()*sizeof(T))); }
    std::vector<T> download() {
        std::vector<T> out(host.size());
        DDLA_CHECK(ddlaMemcpy(h, out.data(), data.data(), out.size()*sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h)); return out;
    }
};

// via_internal routes the call through the no-check internal components
// (ptrtrm_workspace_size + ptrtrm_run) instead of the composing public entry.
template <DdlaBackend Backend, typename T>
void check_case(DdlaHandle_t h, char uplo, int n, int nb, int mode,
                bool shifted, bool via_internal, int& cases, double& worst)
{
    int pr, pc; ddlaGetGridDims(h, pr, pc);
    const int rs = shifted ? pr-1 : 0, cs = shifted ? pc-1 : 0;
    Matrix<T> a(h, n, n, nb, nb, rs, cs);
    Matrix<T> b(h, n, n, nb, nb, rs, cs);
    Matrix<T> cmat(h, n, n, nb, nb, rs, cs);
    const T poison = scalar<T>(std::numeric_limits<double>::quiet_NaN());
    // mode 0: general alpha and beta; 1: alpha=0 beta=0, C poisoned;
    // 2: alpha=0 beta!=0 (beta-only scale); 3: alpha=1 beta=1, lowercase flag;
    // 4: alpha=0 beta=1 no-op, C poisoned everywhere.
    const T alpha = mode==1 || mode==2 || mode==4 ? T(0)
                  : mode==3 ? T(1) : scalar<T>(-.7,.3);
    const T beta  = mode==1 ? T(0) : mode==2 ? scalar<T>(.75)
                  : mode==3 ? T(1) : mode==4 ? T(1) : scalar<T>(-.5,.2);
    const char u = mode==3 ? char(std::tolower(uplo)) : uplo;
    for (int j=0; j<a.nc; ++j) for (int i=0; i<a.nr; ++i) {
        const int gi=a.gi(i), gj=a.gj(j);
        if (gi<n && gj<n) {
            const bool stored = uplo=='L' ? gi>=gj : gi<=gj;
            a.host[i+std::size_t(j)*a.d[DDLA_LLD_]] = stored ? value<T>(gi,gj,1) : poison;
            b.host[i+std::size_t(j)*b.d[DDLA_LLD_]] = stored ? value<T>(gi,gj,2) : poison;
        }
    }
    for (int j=0; j<cmat.nc; ++j) for (int i=0; i<cmat.nr; ++i) {
        const int gi=cmat.gi(i), gj=cmat.gj(j);
        if (gi<n && gj<n) {
            const bool stored = uplo=='L' ? gi>=gj : gi<=gj;
            const bool init = mode!=1 && mode!=4 && (mode!=2 || stored);
            cmat.host[i+std::size_t(j)*cmat.d[DDLA_LLD_]] =
                stored && init ? value<T>(gi,gj,3) : poison;
        }
    }
    a.upload(); b.upload(); cmat.upload();
    std::vector<T> c0 = cmat.host;
    if (via_internal) {
        // No-check inner use: size the scratch, then run the algorithm.
        std::size_t backend_bytes = 0, host_bytes = 0;
        ptrtrm_impl::ptrtrm_workspace_size<T>(h, 'N', 'N', n, a.d, b.d, cmat.d, &backend_bytes, &host_bytes);
        vector<char> backend_work(h, backend_bytes);
        std::vector<char> host_work(host_bytes);
        ptrtrm_impl::ptrtrm_run<Backend,T>(h, uplo, 'N', uplo, 'N', n, alpha,
            a.data.data(), a.d, b.data.data(), b.d, beta, cmat.data.data(), cmat.d,
            backend_work.data(), backend_bytes, host_work.data(), host_bytes);
    } else {
        ptrtrm<Backend,T>(h,u,'N',u,'N',n,alpha,a.data.data(),a.d,b.data.data(),b.d,beta,
                          cmat.data.data(),cmat.d);
    }
    auto out=cmat.download(), aa=a.download(), bb=b.download();
    int bad = std::memcmp(aa.data(),a.host.data(),aa.size()*sizeof(T))!=0
            || std::memcmp(bb.data(),b.host.data(),bb.size()*sizeof(T))!=0;
    double error=0;
    for (std::size_t ix=0; ix<out.size(); ++ix) {
        const int i=int(ix%cmat.d[DDLA_LLD_]), j=int(ix/cmat.d[DDLA_LLD_]);
        const int gi=cmat.gi(i), gj=cmat.gj(j);
        if (i>=cmat.nr || j>=cmat.nc || gi>=n || gj>=n) continue;   // padding stays poisoned
        const bool stored = uplo=='L' ? gi>=gj : gi<=gj;
        if (alpha==T(0) && beta==T(1)) {   // no-op: the buffer must be untouched
            if (std::memcmp(&out[ix],&c0[ix],sizeof(T))) bad=1;
            continue;
        }
        if (!stored) {
            // Opposite triangle of C must be preserved bit for bit.
            if (std::memcmp(&out[ix],&c0[ix],sizeof(T))) bad=1;
            continue;
        }
        std::complex<double> ref=0; double bound=0;
        if (alpha!=T(0)) for (int k=uplo=='L'?gj:gi; k<=(uplo=='L'?gi:gj); ++k) {
            const std::complex<double> av = (uplo=='L' ? gi>=k : gi<=k) ? wide(value<T>(gi,k,1)) : std::complex<double>(0);
            const std::complex<double> bv = (uplo=='L' ? k>=gj : k<=gj) ? wide(value<T>(k,gj,2)) : std::complex<double>(0);
            ref+=av*bv; bound+=std::abs(av)*std::abs(bv);
        }
        ref*=wide(alpha); bound*=std::abs(wide(alpha));
        const auto c0v=wide(c0[ix]);
        if (beta!=T(0)) { ref+=wide(beta)*c0v; bound+=std::abs(wide(beta))*std::abs(c0v); }
        const double e=std::abs(wide(out[ix])-ref)/(1+bound);
        if (!std::isfinite(e)) bad=1; else error=std::max(error,e);
        if (alpha==T(0) && beta==T(0) && out[ix]!=T(0)) bad=1;
    }
    if (std::getenv("DDLA_PTRTRM_DEBUG")) {
        int shown=0;
        for (std::size_t ix=0; ix<out.size() && shown<8; ++ix) {
            const int i=int(ix%cmat.d[DDLA_LLD_]), j=int(ix/cmat.d[DDLA_LLD_]);
            const int gi=cmat.gi(i), gj=cmat.gj(j);
            if (i>=cmat.nr || j>=cmat.nc || gi>=n || gj>=n) continue;
            const bool stored = uplo=='L' ? gi>=gj : gi<=gj;
            if (!stored) continue;
            std::complex<double> ref=0; double b2=0;
            if (alpha!=T(0)) for (int k=uplo=='L'?gj:gi; k<=(uplo=='L'?gi:gj); ++k) {
                const std::complex<double> av=(uplo=='L'?gi>=k:gi<=k)?wide(value<T>(gi,k,1)):std::complex<double>(0);
                const std::complex<double> bv=(uplo=='L'?k>=gj:k<=gj)?wide(value<T>(k,gj,2)):std::complex<double>(0);
                ref+=av*bv; b2+=std::abs(av)*std::abs(bv);
            }
            ref*=wide(alpha);
            if (beta!=T(0)) ref+=wide(beta)*wide(c0[ix]);
            const double e=std::abs(wide(out[ix])-ref)/(1+b2);
            if (!std::isfinite(e) || e>1e-5)
                { std::cerr<<"DBG m"<<mode<<" rk"<<ddlaGetRank(h)<<" gi="<<gi<<" gj="<<gj
                           <<" got="<<wide(out[ix])<<" want="<<wide(ref)<<" e="<<e<<'\n'; ++shown; }
        }
    }
    int any=0; double global=0;
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
    MPI_Allreduce(&error,&global,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));
    constexpr bool single=std::is_same_v<typename ddla::real_type<T>::type,float>;
    if (any || global>(single ? 3e-6 : 5e-14)) {
        if (ddlaGetRank(h)==0) std::cerr << "PTRTRM_FAIL uplo=" << uplo
            << " n=" << n << " nb=" << nb << " mode=" << mode
            << " shifted=" << shifted << " bytes=" << sizeof(T)
            << " grid=" << pr << 'x' << pc << " backend=" << int(Backend)
            << " via=" << (via_internal ? "internal" : "public")
            << " error=" << global << " guard=" << any << '\n';
        throw std::runtime_error("ptrtrm numerical/padding/input check failed");
    }
    worst=std::max(worst,global); ++cases;
}

// Independent stored triangles can produce either triangular or full C. UL and LU produce
// a full matrix; the oracle deliberately makes that matrix nonsymmetric.
template <typename T> std::complex<long double> extended(T x) {
    if constexpr (std::is_arithmetic_v<T>) return {x,0};
    else return {x.real(),x.imag()};
}

template <DdlaBackend Backend, typename T>
void mixed_case(DdlaHandle_t h, char ua, char ub, int n, int nb, int mode,
                bool shifted, int& cases, double& worst, char ta='N', char tb='N', bool via_internal=false)
{
    using W=std::complex<long double>;
    int pr,pc;ddlaGetGridDims(h,pr,pc);
    const int rs=shifted ? pr-1 : 0,cs=shifted ? pc-1 : 0;
    Matrix<T> a(h,n,n,nb,nb,rs,cs),b(h,n,n,nb,nb,rs,cs),c(h,n,n,nb,nb,rs,cs);
    const char ea=ta=='N'?ua:(ua=='U'?'L':'U'),eb=tb=='N'?ub:(ub=='U'?'L':'U');
    const auto selected=[&](int i,int j){return ea!=eb || (ea=='U'?i<=j:i>=j);};
    const double nan=std::numeric_limits<double>::quiet_NaN();
    const T poison=scalar<T>(nan,nan);
    // 0: general update; 1: alpha=beta=0; 2: beta-only scale;
    // 3: lower-case flags; 4: no-op with null C; 5: beta=0 product.
    const T alpha=mode==1 || mode==2 || mode==4 ? T(0) : scalar<T>(-.75,.25);
    const T beta=mode==1 || mode==5 ? T(0) : mode==4 ? T(1) : scalar<T>(.5,-.25);
    bool owns_a=false,owns_b=false,owns_c=false;
    for(int j=0;j<a.nc;++j) for(int i=0;i<a.nr;++i) {
        const int gi=a.gi(i),gj=a.gj(j);
        if(gi<n && gj<n) {
            const bool sa=ua=='U' ? gi<=gj : gi>=gj;
            const bool sb=ub=='U' ? gi<=gj : gi>=gj;
            a.host[i+std::size_t(j)*a.d[DDLA_LLD_]]=sa ? value<T>(gi,gj,1) : poison;
            b.host[i+std::size_t(j)*b.d[DDLA_LLD_]]=sb ? value<T>(gi,gj,2) : poison;
            owns_a|=sa;owns_b|=sb;
        }
    }
    for(int j=0;j<c.nc;++j) for(int i=0;i<c.nr;++i) {
        const int gi=c.gi(i),gj=c.gj(j);
        if(gi<n && gj<n) {
            c.host[i+std::size_t(j)*c.d[DDLA_LLD_]]=
                beta==T(0) || mode==4 ? poison : value<T>(gi,gj,3);
            owns_c|=selected(gi,gj);
        }
    }
    a.upload();b.upload();c.upload();
    const auto flag=[&](char u) { return mode==3 ? char(std::tolower(static_cast<unsigned char>(u))) : u; };
    const bool product=n>0 && alpha!=T(0),noop=n==0 || (alpha==T(0) && beta==T(1));
    const T* ap=product && owns_a ? a.data.data() : nullptr;
    const T* bp=product && owns_b ? b.data.data() : nullptr;
    T* cp=!noop && owns_c ? c.data.data() : nullptr;
    if(via_internal) {
        ptrtrm_impl::ptrtrm_check<Backend,T>(h,ua,ta,ub,tb,n,alpha,ap,a.d,bp,b.d,beta,cp,c.d);
        std::size_t wb=0,hb=0;
        ptrtrm_impl::ptrtrm_workspace_size<T>(h,ta,tb,n,a.d,b.d,c.d,&wb,&hb);
        ddla::vector<char> work(h,wb);std::vector<char> host(hb);
        for(int repeat=0;repeat<2;++repeat) {
            c.upload();
            ptrtrm_impl::ptrtrm_run<Backend,T>(h,ua,ta,ub,tb,n,alpha,ap,a.d,bp,b.d,beta,cp,c.d,
                work.data(),wb,host.data(),hb);
        }
    } else
        ptrtrm<Backend,T>(h,flag(ua),flag(ta),flag(ub),flag(tb),n,alpha,ap,a.d,bp,b.d,beta,cp,c.d);
    const auto out=c.download(),aa=a.download(),bb=b.download();
    int bad=std::memcmp(aa.data(),a.host.data(),aa.size()*sizeof(T))!=0 ||
            std::memcmp(bb.data(),b.host.data(),bb.size()*sizeof(T))!=0;
    long double error=0;
    for(std::size_t p=0;p<out.size();++p) {
        const int i=int(p%c.d[DDLA_LLD_]),j=int(p/c.d[DDLA_LLD_]);
        const int gi=c.gi(i),gj=c.gj(j);
        if(noop || !selected(gi,gj) || i>=c.nr || j>=c.nc || gi>=n || gj>=n) {
            // This checks both descriptor padding and entries beyond the
            // requested leading block, not only the computed triangle.
            if(std::memcmp(&out[p],&c.host[p],sizeof(T))) bad=1;
            continue;
        }
        W ref=0;long double bound=0;
        if(product) for(int k=0;k<n;++k) {
            if((ea=='U' ? gi>k : gi<k) || (eb=='U' ? k>gj : k<gj)) continue;
            W av=extended(ta=='N'?value<T>(gi,k,1):value<T>(k,gi,1));
            W bv=extended(tb=='N'?value<T>(k,gj,2):value<T>(gj,k,2));
            if(ta=='C') av=std::conj(av);
            if(tb=='C') bv=std::conj(bv);
            ref+=av*bv;bound+=std::abs(av)*std::abs(bv);
        }
        ref*=extended(alpha);bound*=std::abs(extended(alpha));
        if(beta!=T(0)) {
            const W old=extended(beta)*extended(c.host[p]);
            ref+=old;bound+=std::abs(old);
        }
        const long double e=std::abs(extended(out[p])-ref)/(1+bound);
        if(!std::isfinite(e)) bad=1;else error=std::max(error,e);
        if(alpha==T(0) && beta==T(0) && out[p]!=T(0)) bad=1;
    }
    int any=0;long double global=0;
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
    MPI_Allreduce(&error,&global,1,MPI_LONG_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));
    constexpr bool single=std::is_same_v<typename ddla::real_type<T>::type,float>;
    if(any || global>(single ? 3e-6L : 5e-14L)) {
        if(ddlaGetRank(h)==0) std::cerr<<"PTRTRM_MIXED_FAIL flags="<<ua<<ub
            <<" trans="<<ta<<tb<<" n="<<n<<" nb="<<nb<<" mode="<<mode<<" shifted="<<shifted
            <<" bytes="<<sizeof(T)<<" grid="<<pr<<'x'<<pc<<" backend="<<int(Backend)
            <<" error="<<double(global)<<" guard="<<any<<'\n';
        throw std::runtime_error("mixed ptrtrm full-output/guard/input check failed");
    }
    worst=std::max(worst,double(global));++cases;
}

template <DdlaBackend Backend, typename T>
void mixed_scalar_cases(DdlaHandle_t h,int& cases,double& worst,bool stress)
{
    for(char ua:{'U','L'}) {
        for(int mode=0;mode<=4;++mode)
            check_case<Backend,T>(h,ua,13,4,mode,true,false,cases,worst);
        const char ub=ua=='U' ? 'L' : 'U';
        for(bool shifted:{false,true}) {
            mixed_case<Backend,T>(h,ua,ub,13,4,0,shifted,cases,worst);
            mixed_case<Backend,T>(h,ua,ub,1,4,0,shifted,cases,worst);
            mixed_case<Backend,T>(h,ua,ub,64,16,0,shifted,cases,worst);
        }
        for(int mode:{1,2,3,4,5}) mixed_case<Backend,T>(h,ua,ub,13,4,mode,true,cases,worst);
        mixed_case<Backend,T>(h,ua,ub,7,1,0,false,cases,worst);
        mixed_case<Backend,T>(h,ua,ub,5,32,5,true,cases,worst);
        mixed_case<Backend,T>(h,ua,ub,0,4,0,false,cases,worst);
        if(stress) mixed_case<Backend,T>(h,ua,ub,137,32,0,true,cases,worst);
    }
}

template <DdlaBackend Backend, typename T>
void transpose_cases(DdlaHandle_t h,int& cases,double& worst)
{
    for(char ua:{'U','L'}) for(char ub:{'U','L'})
        for(char ta:{'N','T','C'}) for(char tb:{'N','T','C'}) {
            mixed_case<Backend,T>(h,ua,ub,13,4,0,true,cases,worst,ta,tb);
            mixed_case<Backend,T>(h,ua,ub,7,1,5,false,cases,worst,ta,tb,true);
            mixed_case<Backend,T>(h,ua,ub,1,4,3,true,cases,worst,ta,tb);
            for(int mode:{1,2,4}) mixed_case<Backend,T>(h,ua,ub,5,4,mode,true,cases,worst,ta,tb);
            mixed_case<Backend,T>(h,ua,ub,0,4,0,false,cases,worst,ta,tb);
        }
    for(char ua:{'U','L'}) for(char ub:{'U','L'})
        mixed_case<Backend,T>(h,ua,ub,65,32,5,true,cases,worst,'C','T',true);
}

template <DdlaBackend Backend>
void transpose_invalid_cases(DdlaHandle_t h)
{
    Matrix<double> a(h,8,8,4,4,0,0),b(h,8,8,4,4,0,0),c(h,8,8,4,4,0,0);
    a.upload();b.upload();c.upload();
    for(int kind=0;kind<4;++kind) {
        int size=0;MPI_Comm_size(ddlaGetCommunicator(h),&size);if(kind>=2 && size==1) continue;
        const bool root=ddlaGetRank(h)==0;
        const char ta=kind==0 && root?'?':kind==2 && root?'T':'N';
        const char tb=kind==1 && root?'?':kind==3 && root?'C':'N';
        bool caught=false;
        try {ptrtrm<Backend,double>(h,'U',ta,'L',tb,8,1.,a.data.data(),a.d,b.data.data(),b.d,0.,c.data.data(),c.d);}
        catch(const std::invalid_argument&){if(kind>=2) throw;caught=true;}
        catch(const std::runtime_error&){if(kind<2) throw;caught=true;}
        int all=0,yes=caught?1:0,np=0;MPI_Comm_size(ddlaGetCommunicator(h),&np);
        MPI_Allreduce(&yes,&all,1,MPI_INT,MPI_SUM,ddlaGetCommunicator(h));
        if(all!=np) throw std::runtime_error("ptrtrm transpose flag error/disagreement not rejected");
        if(a.download()!=a.host || b.download()!=b.host || c.download()!=c.host)
            throw std::runtime_error("ptrtrm invalid transpose call modified matrices");
    }
}

template <DdlaBackend Backend>
void mixed_invalid_cases(DdlaHandle_t h)
{
    Matrix<double> a(h,12,12,4,4,0,0),b(h,12,12,4,4,0,0),c(h,12,12,4,4,0,0);
    a.upload();b.upload();c.upload();
    const bool root=ddlaGetRank(h)==0;
    for(int kind=0;kind<8;++kind) {
        const char ua=kind==0 && root ? '?' : 'U',ub=kind==1 && root ? '?' : 'L';
        const int n=kind==2 ? -1 : kind==3 ? a.d[DDLA_M_]+1 : 12;
        const double* ap=kind==4 && root ? nullptr : a.data.data();
        double* cp=kind==5 && root ? nullptr : kind==6 && root ? a.data.data() :
                   kind==7 && root ? b.data.data()+1 : c.data.data();
        bool rejected=false;
        try { ptrtrm<Backend,double>(h,ua,'N',ub,'N',n,1.,ap,a.d,b.data.data(),b.d,0.,cp,c.d); }
        catch(const std::exception&) { rejected=true; }
        int caught=rejected ? 1 : 0,all=0;
        MPI_Allreduce(&caught,&all,1,MPI_INT,MPI_MIN,ddlaGetCommunicator(h));
        if(!all) throw std::runtime_error("mixed ptrtrm invalid argument not rejected collectively");
        const auto ao=a.download(),bo=b.download(),co=c.download();
        if(ao!=a.host || bo!=b.host || co!=c.host)
            throw std::runtime_error("mixed ptrtrm invalid call modified input or output");
    }
    int pr,pc,size;
    ddlaGetGridDims(h,pr,pc);
    MPI_Comm_size(ddlaGetCommunicator(h),&size);
    // Every rank has internally matching, valid A/B/C descriptors, but one
    // rank supplies a different global source coordinate. Even a no-op must
    // reject that disagreement collectively before the run phase.
    for(int axis=0;axis<2;++axis) {
        if((axis==0 ? pr : pc)<=1) continue;
        int da[DDLA_DLEN_],db[DDLA_DLEN_],dc[DDLA_DLEN_];
        const int rs=axis==0 && root ? 1 : 0,cs=axis==1 && root ? 1 : 0;
        DDLA_CHECK(ddlaDescInit(da,h,12,12,4,4,rs,cs));
        DDLA_CHECK(ddlaDescInit(db,h,12,12,4,4,rs,cs));
        DDLA_CHECK(ddlaDescInit(dc,h,12,12,4,4,rs,cs));
        int caught=0,all=0;
        try { ptrtrm<Backend,double>(h,'U','N','L','N',0,0.,nullptr,da,nullptr,db,1.,nullptr,dc); }
        catch(const std::exception&) { caught=1; }
        MPI_Allreduce(&caught,&all,1,MPI_INT,MPI_MIN,ddlaGetCommunicator(h));
        if(!all) throw std::runtime_error("ptrtrm source disagreement not rejected collectively");
    }
    if(size>1) for(int kind=0;kind<3;++kind) {
        const char ub=kind==0 && root ? 'U' : 'L';
        const int n=kind==1 && root ? 1 : 0;
        const double alpha=kind==2 && root ? 1. : 0.;
        const double beta=kind==2 && root ? 0. : 1.;
        int caught=0,all=0;
        try { ptrtrm<Backend,double>(h,'U','N',ub,'N',n,alpha,nullptr,a.d,nullptr,b.d,beta,nullptr,c.d); }
        catch(const std::exception&) { caught=1; }
        MPI_Allreduce(&caught,&all,1,MPI_INT,MPI_MIN,ddlaGetCommunicator(h));
        if(!all) throw std::runtime_error("ptrtrm mode/order disagreement not rejected collectively");
    }
}

template <DdlaBackend Backend, typename T>
void scalar_cases(DdlaHandle_t h, int& cases, double& worst, bool stress) {
    for (char uplo : {'U','L'}) {
        for (bool shifted : {false,true}) {
            check_case<Backend,T>(h,uplo,13,4,0,shifted,false,cases,worst);
            check_case<Backend,T>(h,uplo,1,4,0,shifted,false,cases,worst);
            check_case<Backend,T>(h,uplo,64,16,0,shifted,false,cases,worst);
        }
        check_case<Backend,T>(h,uplo,7,1,0,false,false,cases,worst);
        check_case<Backend,T>(h,uplo,5,32,0,true,false,cases,worst);
        check_case<Backend,T>(h,uplo,13,4,1,true,false,cases,worst);
        check_case<Backend,T>(h,uplo,13,4,2,true,false,cases,worst);
        check_case<Backend,T>(h,uplo,13,4,3,true,false,cases,worst);
        check_case<Backend,T>(h,uplo,13,4,4,true,false,cases,worst);
        check_case<Backend,T>(h,uplo,0,4,0,false,false,cases,worst);
        if (stress) check_case<Backend,T>(h,uplo,137,32,0,true,false,cases,worst);
        // The no-check internal path on the same machinery: general scalars,
        // the alpha==0 zero-inner-dimension path, accumulate, and the partial
        // last block under stress.  Mode 4 (the no-op) is the public entry's
        // early return and has no internal counterpart.
        check_case<Backend,T>(h,uplo,13,4,0,false,true,cases,worst);
        check_case<Backend,T>(h,uplo,64,16,0,true,true,cases,worst);
        check_case<Backend,T>(h,uplo,13,4,2,true,true,cases,worst);
        check_case<Backend,T>(h,uplo,13,4,3,true,true,cases,worst);
        if (stress) check_case<Backend,T>(h,uplo,137,32,0,true,true,cases,worst);
    }
}

template <DdlaBackend Backend>
void invalid_cases(DdlaHandle_t h) {
    int pr, pc; ddlaGetGridDims(h, pr, pc);
    int a[DDLA_DLEN_],b[DDLA_DLEN_],cc[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(a,h,12,12,4,4,0,0));
    DDLA_CHECK(ddlaDescInit(b,h,12,12,4,4,0,0));
    DDLA_CHECK(ddlaDescInit(cc,h,12,12,4,4,0,0));
    auto reject_args=[&](char u,int n,const double* A,const double* B2) {
        bool caught=false;
        try { ptrtrm<Backend,double>(h,u,'N',u,'N',n,0.,A,a,B2,b,0.,nullptr,cc); }
        catch(const std::invalid_argument&) { caught=true; }
        if (!caught) throw std::runtime_error("ptrtrm invalid argument not rejected");
    };
    reject_args('?',2,nullptr,nullptr);
    reject_args('U',-1,nullptr,nullptr);
    reject_args('U',13,nullptr,nullptr);
    reject_args('U',2,nullptr,nullptr);          // null matrices with alpha=0
    DDLA_CHECK(ddlaDescInit(b,h,12,12,3,3,0,0));
    reject_args('U',2,nullptr,nullptr);          // B block mismatch
    DDLA_CHECK(ddlaDescInit(b,h,12,12,4,4,0,0));
    DDLA_CHECK(ddlaDescInit(cc,h,12,12,4,3,0,0));
    reject_args('L',2,nullptr,nullptr);          // C non-square blocks
    if (pr > 1) {                                // rsrc=1 is not a valid descriptor on 1 row
        DDLA_CHECK(ddlaDescInit(cc,h,12,12,4,4,1,0));
        reject_args('U',2,nullptr,nullptr);      // C rsrc mismatch
        DDLA_CHECK(ddlaDescInit(cc,h,12,12,4,4,0,0));
        DDLA_CHECK(ddlaDescInit(b,h,12,12,4,4,1,0));
        reject_args('U',2,nullptr,nullptr);      // B rsrc mismatch
        DDLA_CHECK(ddlaDescInit(b,h,12,12,4,4,0,0));
    }
    if (pc > 1) {                                // csrc=1 is not a valid descriptor on 1 column
        DDLA_CHECK(ddlaDescInit(cc,h,12,12,4,4,0,1));
        reject_args('U',2,nullptr,nullptr);      // C csrc mismatch
        DDLA_CHECK(ddlaDescInit(cc,h,12,12,4,4,0,0));
    }
    // A local error must make every rank throw before any panel exchange.
    // Descriptors and buffer addresses are local arguments, so only rank 0
    // is made invalid; the other ranks still have valid storage.
    Matrix<double> ma(h,12,12,4,4,0,0), mb(h,12,12,4,4,0,0), mc(h,12,12,4,4,0,0);
    ma.upload(); mb.upload(); mc.upload();
    for (int kind=0; kind<3; ++kind) {
        int bad_desc[DDLA_DLEN_]; std::copy(ma.d,ma.d+DDLA_DLEN_,bad_desc);
        const bool bad_rank=ddlaGetRank(h)==0;
        if (kind==2 && bad_rank) bad_desc[DDLA_LLD_]=0;
        bool rejected=false;
        try {
            ptrtrm<Backend,double>(h,'U','N','U','N',12,1.,kind==0 && bad_rank ? nullptr : ma.data.data(),bad_desc,
                mb.data.data(),mb.d,0.,kind==1 && bad_rank ? nullptr : mc.data.data(),mc.d);
        } catch(const std::invalid_argument&) { rejected=true; }
        int local=rejected ? 1 : 0, all=0, size=0;
        MPI_Comm_size(ddlaGetCommunicator(h),&size);
        MPI_Allreduce(&local,&all,1,MPI_INT,MPI_SUM,ddlaGetCommunicator(h));
        if (all!=size) throw std::runtime_error("ptrtrm rank-local error not rejected on every rank");
        const auto ao=ma.download(),bo=mb.download(),co=mc.download();
        if (ao!=ma.host || bo!=mb.host || co!=mc.host)
            throw std::runtime_error("ptrtrm invalid call modified a matrix");
    }
    bool caught=false;
    try { ptrtrm<Backend,double>(nullptr,'U','N','U','N',0,0.,nullptr,a,nullptr,b,0.,nullptr,cc); }
    catch(const std::runtime_error&) { caught=true; }
    if (!caught) throw std::runtime_error("ptrtrm null handle not rejected");
#if DDLA_HAS_GPU
    caught=false;
    constexpr auto Wrong = Backend==DdlaBackend::CPU ? DdlaBackend::GPU : DdlaBackend::CPU;
    try { ptrtrm<Wrong,double>(h,'U','N','U','N',0,0.,nullptr,a,nullptr,b,0.,nullptr,cc); }
    catch(const std::runtime_error&) { caught=true; }
    if (!caught) throw std::runtime_error("ptrtrm backend mismatch not rejected");
#endif
}

template <DdlaBackend Backend>
void run(MPI_Comm comm,int pr,int pc,char major,bool stress) {
    DdlaHandle_t h=nullptr; DDLA_CHECK(ddlaInit(h,Backend)); DDLA_CHECK(ddlaSet(h,comm,pr,pc,major));
    int cases=0; double worst=0;
    scalar_cases<Backend,float>(h,cases,worst,stress);
    scalar_cases<Backend,double>(h,cases,worst,stress);
    scalar_cases<Backend,std::complex<float>>(h,cases,worst,stress);
    scalar_cases<Backend,std::complex<double>>(h,cases,worst,stress);
    mixed_scalar_cases<Backend,float>(h,cases,worst,stress);
    mixed_scalar_cases<Backend,double>(h,cases,worst,stress);
    mixed_scalar_cases<Backend,std::complex<float>>(h,cases,worst,stress);
    mixed_scalar_cases<Backend,std::complex<double>>(h,cases,worst,stress);
    transpose_cases<Backend,float>(h,cases,worst);
    transpose_cases<Backend,double>(h,cases,worst);
    transpose_cases<Backend,std::complex<float>>(h,cases,worst);
    transpose_cases<Backend,std::complex<double>>(h,cases,worst);
    transpose_invalid_cases<Backend>(h);
    invalid_cases<Backend>(h);
    mixed_invalid_cases<Backend>(h);
    if(ddlaGetRank(h)==0) std::cout << "PTRTRM_PASS backend=" << int(Backend) << " grid=" << pr << 'x' << pc
        << " major=" << major << " cases=" << cases << " max_scaled_error=" << worst << '\n';
    DDLA_CHECK(ddlaDestroy(h));
}
} // namespace

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    try {
        int np,rank; MPI_Comm_size(MPI_COMM_WORLD,&np); MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        api_grid_test::TestOptions options; std::string error;
        bool stress=false; if(argc>1 && std::string(argv[argc-1])=="--stress") {stress=true;--argc;}
        if(!api_grid_test::parse_options(argc,argv,np,options,error)) throw std::runtime_error(error);
        MPI_Comm reversed; MPI_Comm_split(MPI_COMM_WORLD,0,np-1-rank,&reversed);
        for(auto grid:options.grids) for(char major:{'R','C'}) {
            run<DdlaBackend::CPU>(reversed,grid.first,grid.second,major,stress);
#if DDLA_HAS_GPU
            run<DdlaBackend::GPU>(reversed,grid.first,grid.second,major,stress);
#endif
        }
        MPI_Comm_free(&reversed); MPI_Finalize(); return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
