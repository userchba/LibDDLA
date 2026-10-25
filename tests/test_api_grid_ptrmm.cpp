#include "api_grid_test_common.h"
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

template <DdlaBackend Backend, typename T>
void check_case(DdlaHandle_t h, char side, char uplo, char trans, char diag,
                int m, int n, int nb, int mode, bool shifted, int& cases, double& worst)
{
    int pr, pc; ddlaGetGridDims(h, pr, pc);
    const int order = side == 'L' ? m : n;
    // Independent, deliberately different A/B origins, including an origin
    // outside the opposite process dimension on rectangular grids.
    Matrix<T> a(h, order, order, nb, nb, shifted ? pr-1 : 0, shifted ? pc-1 : 0);
    Matrix<T> b(h, m, n, side == 'L' ? nb : 3, side == 'L' ? 3 : nb,
                shifted ? (pr > 2 ? 1 : 0) : 0, shifted ? (pc > 2 ? 1 : 0) : 0);
    const T poison = scalar<T>(std::numeric_limits<double>::quiet_NaN());
    for (int j=0; j<a.nc; ++j) for (int i=0; i<a.nr; ++i) {
        const int gi=a.gi(i), gj=a.gj(j);
        if (gi<order && gj<order) {
            const bool stored = uplo == 'L' ? gi>=gj : gi<=gj;
            a.host[i+std::size_t(j)*a.d[DDLA_LLD_]] = stored && !(diag=='U' && gi==gj) ? value<T>(gi,gj,1) : poison;
        }
    }
    for (int j=0; j<b.nc; ++j) for (int i=0; i<b.nr; ++i)
        if (b.gi(i)<m && b.gj(j)<n)
            b.host[i+std::size_t(j)*b.d[DDLA_LLD_]] = mode==1 ? poison : value<T>(b.gi(i),b.gj(j),2);
    a.upload(); b.upload();
    const T alpha = mode==1 ? T(0) : mode==2 ? T(1) : scalar<T>(-.7,.3);
    const char s = mode==2 ? char(std::tolower(side)) : side;
    const char u = mode==2 ? char(std::tolower(uplo)) : uplo;
    const char t = mode==2 ? char(std::tolower(trans)) : trans;
    const char d = mode==2 ? char(std::tolower(diag)) : diag;
    const bool empty = m==0 || n==0;
    ptrmm<Backend,T>(h,s,u,t,d,m,n,alpha,alpha==T(0) || empty ? nullptr : a.data.data(),a.d,
                    empty ? nullptr : b.data.data(),b.d);
    auto out=b.download(), aa=a.download();
    int bad=std::memcmp(aa.data(),a.host.data(),aa.size()*sizeof(T))!=0;
    double error=0;
    for (std::size_t ix=0; ix<out.size(); ++ix) {
        const int i=ix%b.d[DDLA_LLD_],j=ix/b.d[DDLA_LLD_],gi=b.gi(i),gj=b.gj(j);
        if (i<b.nr && j<b.nc && gi<m && gj<n) {
            std::complex<double> ref=0; double bound=0;
            if (alpha!=T(0)) for (int k=0; k<order; ++k) {
                int ai=side=='L'?gi:k, aj=side=='L'?k:gj;
                if (trans!='N') std::swap(ai,aj);
                std::complex<double> av=0;
                if (ai==aj && diag=='U') av=1;
                else if (uplo=='L'?ai>=aj:ai<=aj) av=wide(value<T>(ai,aj,1));
                if (trans=='C') av=std::conj(av);
                const auto bv=wide(side=='L'?value<T>(k,gj,2):value<T>(gi,k,2));
                ref+=av*bv; bound+=std::abs(av)*std::abs(bv);
            }
            ref*=wide(alpha); bound*=std::abs(wide(alpha));
            const double e=std::abs(wide(out[ix])-ref)/(1+bound);
            if (!std::isfinite(e)) bad=1; else error=std::max(error,e);
            if (alpha==T(0) && out[ix]!=T(0)) bad=1;
        } else if (std::memcmp(&out[ix],&b.host[ix],sizeof(T))) bad=1;
    }
    int any=0; double global=0;
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
    MPI_Allreduce(&error,&global,1,MPI_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));
    constexpr bool single=std::is_same_v<typename ddla::real_type<T>::type,float>;
    if (any || global>(single ? 3e-6 : 5e-14)) {
        if (ddlaGetRank(h)==0) std::cerr << "PTRMM_FAIL flags=" << side << uplo << trans << diag
            << " shape=" << m << ',' << n << " nb=" << nb << " mode=" << mode
            << " shifted=" << shifted << " bytes=" << sizeof(T) << " error=" << global << " guard=" << any << '\n';
        throw std::runtime_error("ptrmm numerical/padding/input check failed");
    }
    worst=std::max(worst,global); ++cases;
}

template <DdlaBackend Backend, typename T>
void scalar_cases(DdlaHandle_t h, int& cases, double& worst, bool stress) {
    for (char side : {'L','R'}) for (char uplo : {'L','U'})
    for (char trans : {'N','T','C'}) for (char diag : {'N','U'}) {
        for (bool shifted : {false,true}) {
            check_case<Backend,T>(h,side,uplo,trans,diag,13,9,4,0,shifted,cases,worst);
            check_case<Backend,T>(h,side,uplo,trans,diag,1,2,4,0,shifted,cases,worst);
        }
        check_case<Backend,T>(h,side,uplo,trans,diag,8,12,4,2,true,cases,worst);
        check_case<Backend,T>(h,side,uplo,trans,diag,7,5,1,0,true,cases,worst);
        check_case<Backend,T>(h,side,uplo,trans,diag,13,9,4,1,true,cases,worst);
        check_case<Backend,T>(h,side,uplo,trans,diag,0,9,4,0,true,cases,worst);
        check_case<Backend,T>(h,side,uplo,trans,diag,13,0,4,0,true,cases,worst);
        if (stress) check_case<Backend,T>(h,side,uplo,trans,diag,137,119,32,0,true,cases,worst);
    }
}

template <DdlaBackend Backend>
void invalid_cases(DdlaHandle_t h) {
    int a[DDLA_DLEN_],b[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(a,h,12,12,4,4,0,0));
    std::copy(a,a+DDLA_DLEN_,b);
    auto reject=[&](char s,char u,char t,char d,int m,int n) {
        bool caught=false;
        try { ptrmm<Backend,double>(h,s,u,t,d,m,n,0.,nullptr,a,nullptr,b); }
        catch(const std::invalid_argument&) { caught=true; }
        if (!caught) throw std::runtime_error("ptrmm invalid argument not rejected");
    };
    reject('?','U','N','N',2,2); reject('L','?','N','N',2,2);
    reject('L','U','?','N',2,2); reject('L','U','N','?',2,2);
    reject('L','U','N','N',-1,2); reject('R','U','N','N',2,-1);
    reject('L','U','N','N',13,2); reject('R','U','N','N',2,13);
    b[DDLA_M_]=1; reject('L','U','N','N',2,2); b[DDLA_M_]=12;
    b[DDLA_N_]=1; reject('R','U','N','N',2,2); b[DDLA_N_]=12;
    DDLA_CHECK(ddlaDescInit(b,h,12,12,3,3,0,0));
    reject('L','U','N','N',2,2); reject('R','U','N','N',2,2);
    DDLA_CHECK(ddlaDescInit(a,h,12,12,4,3,0,0)); reject('L','U','N','N',2,2);
    bool caught=false;
    try { ptrmm<Backend,double>(nullptr,'L','U','N','N',0,0,0.,nullptr,a,nullptr,b); }
    catch(const std::runtime_error&) { caught=true; }
    if (!caught) throw std::runtime_error("ptrmm null handle not rejected");
#if DDLA_HAS_CPU && DDLA_HAS_GPU
    caught=false;
    constexpr auto Wrong = Backend==DdlaBackend::CPU ? DdlaBackend::GPU : DdlaBackend::CPU;
    try { ptrmm<Wrong,double>(h,'L','U','N','N',0,0,0.,nullptr,a,nullptr,b); }
    catch(const std::runtime_error&) { caught=true; }
    if (!caught) throw std::runtime_error("ptrmm backend mismatch not rejected");
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
    invalid_cases<Backend>(h);
    if(ddlaGetRank(h)==0) std::cout << "PTRMM_PASS backend=" << int(Backend) << " grid=" << pr << 'x' << pc
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
        // Reverse communicator ranks as well as exercising both grid orders.
        MPI_Comm reversed; MPI_Comm_split(MPI_COMM_WORLD,0,np-1-rank,&reversed);
        for(auto grid:options.grids) for(char major:{'R','C'}) {
#if DDLA_HAS_CPU
            run<DdlaBackend::CPU>(reversed,grid.first,grid.second,major,stress);
#endif
#if DDLA_HAS_GPU
            run<DdlaBackend::GPU>(reversed,grid.first,grid.second,major,stress);
#endif
        }
        MPI_Comm_free(&reversed); MPI_Finalize(); return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
