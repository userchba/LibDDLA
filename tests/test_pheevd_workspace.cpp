// Regression coverage for distributed eigensolver workspace reuse.
// Compact and padded row slabs are compared with independent residual checks;
// Padded block-cyclic outputs cover direct output, uneven receive capacity,
// and empty local rows/columns when the matrix is smaller than the grid.
// With no arguments, the test derives every valid grid from the MPI size
// (the standard CTest invocation uses four ranks). --grid RxC selects one.
// DDLA_TEST_BACKEND=CPU|GPU optionally overrides the compiled default backend.
#include "api_grid_test_common.h"
#include "eig_test_common.h"
#include "row_share.h"
#include "Eigensolver/pstedc_impl.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>

using namespace api_grid_test;

namespace {

struct EnvGuard {
    std::string name, old;
    bool had_old;
    EnvGuard(const char* key, const char* value) : name(key), had_old(std::getenv(key) != nullptr)
    {
        if (had_old) old = std::getenv(key);
        if (setenv(key, value, 1) != 0) throw std::runtime_error("setenv failed");
    }
    ~EnvGuard() { if (had_old) setenv(name.c_str(), old.c_str(), 1); else unsetenv(name.c_str()); }
};

double hash_value(std::uint64_t key)
{
    key += 0x9e3779b97f4a7c15ull;
    key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ull;
    key = (key ^ (key >> 27)) * 0x94d049bb133111ebull;
    key ^= key >> 31;
    return static_cast<double>(key >> 11) / 9007199254740992.0 - 0.5;
}

template <typename T> const char* type_name();
template <> const char* type_name<float>() { return "s"; }
template <> const char* type_name<double>() { return "d"; }
template <> const char* type_name<std::complex<float>>() { return "c"; }
template <> const char* type_name<std::complex<double>>() { return "z"; }

void finite_metric(const ddla::DdlaHandle_t& h, const std::string& tag, double value, double bound)
{
    require_close(h, tag, std::isfinite(value) ? value : std::numeric_limits<double>::infinity(), bound);
}

template <typename rt>
void check_rows(const ddla::DdlaHandle_t& h, int n, int kind)
{
    int P = 0;
    MPI_Comm_size(ddlaGetCommunicator(h), &P);
    const int rank = ddlaGetRank(h);
    const ddla::detail::RowShare rows(n, P);
    const int nr = rows.count(rank), lo = rows.lo(rank), ld = std::max(nr, 1);
    const rt sentinel = rt(12345);
    const double eps = std::numeric_limits<rt>::epsilon();
    const std::string tag = std::string("rows ") + type_name<rt>() + " n=" + std::to_string(n) +
                            " kind=" + std::to_string(kind);
    std::vector<rt> d0(n), e0(n, rt(0));
    double normT = 0;
    for (int i = 0; i < n; ++i) {
        if (kind == 0) {
            d0[i] = rt(hash_value(2 * i));
            if (i + 1 < n) e0[i] = rt(hash_value(2 * i + 1));
        } else {
            d0[i] = rt(0.5 * (i % 4));
            if (kind == 1 && i + 1 < n) e0[i] = rt(1e-3 * hash_value(i + 37));
            // kind 2 has exact repeated eigenvalues and zero coupling: K=0 merges.
        }
    }
    for (int i = 0; i < n; ++i)
        normT = std::max(normT, std::abs(double(d0[i])) + std::abs(double(e0[i])) +
                        (i ? std::abs(double(e0[i - 1])) : 0.0));
    normT = std::max(normT, 1.0);

    auto solve = [&](int ldy, std::vector<rt>& w) {
        const std::size_t count = static_cast<std::size_t>(ldy) * n;
        DeviceBuffer<rt> Y(h, count);
        upload(h, Y.ptr, std::vector<rt>(count, sentinel));
        std::vector<rt> d(d0), e(e0);
        w.assign(n, std::numeric_limits<rt>::quiet_NaN());
        ddla::pstedc(h, n, d.data(), e.data(), w.data(), Y.ptr, ldy);
        return download(h, Y.ptr, count);
    };
    std::vector<rt> wc, wp;
    const auto yc = solve(ld, wc);
    const auto yp = solve(ld + 7, wp);
    double bad = 0, padding_bad = 0, wdev = 0, ydev = 0;
    for (int j = 0; j < n; ++j) {
        if (!std::isfinite(double(wc[j])) || !std::isfinite(double(wp[j]))) ++bad;
        else wdev = std::max(wdev, std::abs(double(wc[j]) - double(wp[j])));
        if (j && wc[j] < wc[j - 1]) ++bad;
        for (int r = 0; r < nr; ++r) {
            const rt a = yc[r + static_cast<std::size_t>(j) * ld];
            const rt b = yp[r + static_cast<std::size_t>(j) * (ld + 7)];
            if (!std::isfinite(double(a)) || !std::isfinite(double(b))) ++bad;
            else ydev = std::max(ydev, std::abs(double(a) - double(b)));
        }
        for (int r = nr; r < ld; ++r)
            padding_bad += yc[r + static_cast<std::size_t>(j) * ld] != sentinel;
        for (int r = nr; r < ld + 7; ++r)
            padding_bad += yp[r + static_cast<std::size_t>(j) * (ld + 7)] != sentinel;
    }
    require_close(h, tag + " finite and sorted", bad, 0);
    require_close(h, tag + " padding", padding_bad, 0);
    finite_metric(h, tag + " compact/padded eigenvalues", wdev / (n * eps * normT), 10);
    finite_metric(h, tag + " compact/padded rows", ydev / (n * eps), 10);

    // Assemble a small host reference view solely for an independent T*Y and
    // Y^T*Y check. Production row-slab storage remains distributed.
    std::vector<double> full(static_cast<std::size_t>(n) * n, 0.0);
    for (int j = 0; j < n; ++j)
        for (int r = 0; r < nr; ++r)
            full[lo + r + static_cast<std::size_t>(j) * n] = yp[r + static_cast<std::size_t>(j) * (ld + 7)];
    MPI_Allreduce(MPI_IN_PLACE, full.data(), n * n, MPI_DOUBLE, MPI_SUM, ddlaGetCommunicator(h));
    double residual = 0, ortho = 0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            double v = double(d0[i]) * full[i + static_cast<std::size_t>(j) * n];
            if (i) v += double(e0[i - 1]) * full[i - 1 + static_cast<std::size_t>(j) * n];
            if (i + 1 < n) v += double(e0[i]) * full[i + 1 + static_cast<std::size_t>(j) * n];
            residual = std::max(residual, std::abs(v - double(wp[j]) * full[i + static_cast<std::size_t>(j) * n]));
        }
        for (int k = 0; k < n; ++k) {
            double dot = 0;
            for (int i = 0; i < n; ++i)
                dot += full[i + static_cast<std::size_t>(j) * n] * full[i + static_cast<std::size_t>(k) * n];
            ortho = std::max(ortho, std::abs(dot - (j == k ? 1.0 : 0.0)));
        }
    }
    finite_metric(h, tag + " independent residual", residual / (n * eps * normT), 50);
    finite_metric(h, tag + " independent orthogonality", ortho / (n * eps), 50);
}

template <typename T> std::complex<double> wide(T v) { return {double(v), 0}; }
template <typename rt> std::complex<double> wide(std::complex<rt> v) { return {double(v.real()), double(v.imag())}; }

template <typename T>
T hermitian(int i, int j)
{
    using rt = typename ddla::real_type<T>::type;
    if (i == j) return T(rt(5.0 + 0.03 * i));
    const int hi = std::max(i, j), lo = std::min(i, j);
    const rt re = rt(hash_value(static_cast<std::uint64_t>(hi) * 127 + lo));
    if constexpr (std::is_same<T, rt>::value) return re;
    else {
        const rt im = rt(hash_value(static_cast<std::uint64_t>(hi) * 131 + lo + 19));
        return T(re, i > j ? im : -im);
    }
}

template <ddla::DdlaBackend B, typename T>
void check_padded_pheevd(const ddla::DdlaHandle_t& h, char uplo, int n, int nb, bool shifted_columns=false)
{
    using rt = typename ddla::real_type<T>::type;
    const std::string nbw_value = std::to_string(nb);
    EnvGuard nbw("DDLA_EIG_NBW", nbw_value.c_str());
    const double eps = std::numeric_limits<rt>::epsilon();
    const T sentinel = T(rt(12345));
    int descA[ddla::DDLA_DLEN_], descZ[ddla::DDLA_DLEN_];
    ddlaDescInit(descA, h, n, n, nb, nb, 0, 0);
    int npr=0,npc=0;ddlaGetGridDims(h,npr,npc);
    ddlaDescInit(descZ,h,n+(shifted_columns?5:0),n+(shifted_columns?7:0),nb,nb,0,shifted_columns?npc-1:0);
    descZ[ddla::DDLA_LLD_] += 7;
    DeviceBuffer<T> A(h, local_size(h, descA)), Z(h, local_size(h, descZ));
    const auto original = make_local<T>(h, descA, [](int i, int j) { return hermitian<T>(i, j); });
    auto input = make_local<T>(h, descA, [&](int i, int j) {
        return (uplo == 'U' ? i > j : i < j) ? T(rt(1e6)) : hermitian<T>(i, j);
    });
    upload(h, A.ptr, input);
    upload(h, Z.ptr, std::vector<T>(Z.count, sentinel));
    std::vector<rt> w(n, std::numeric_limits<rt>::quiet_NaN());
    int info = -1;
    ddla::pheevd<B, T>(h, 'V', uplo, n, A.ptr, descA, w.data(), Z.ptr, descZ, info);
    check_ddla_sync(h);
    const std::string tag = std::string("pheevd padded ") + type_name<T>() + " " + uplo +
                            " n=" + std::to_string(n) + " nb=" + std::to_string(nb)+
                            " shifted_columns="+std::to_string(shifted_columns);
    require_close(h, tag + " info", std::abs(info), 0);
    const auto z = download(h, Z.ptr, Z.count);
    const int nr = ddla_test::m_loc(h, descZ), nc = ddla_test::n_loc(h, descZ);
    const int ldz = descZ[ddla::DDLA_LLD_];
    double bad = 0, padding_bad = 0;
    for (int j = 0; j < n; ++j) {
        if (!std::isfinite(double(w[j]))) ++bad;
        if (j && w[j] < w[j - 1]) ++bad;
    }
    std::vector<double> encoded(static_cast<std::size_t>(2) * n * n, 0);
    for (int j = 0; j < nc; ++j) {
        const int gj = ddla::indx_l2g_c(descZ, h, j);
        for (int i = 0; i < nr; ++i) {
            const int gi = ddla::indx_l2g_r(descZ, h, i);
            if(gi>=n || gj>=n) {
                padding_bad+=z[i+static_cast<std::size_t>(j)*ldz]!=sentinel;
                continue;
            }
            const auto v = wide(z[i + static_cast<std::size_t>(j) * ldz]);
            if (!std::isfinite(v.real()) || !std::isfinite(v.imag())) ++bad;
            encoded[2 * (gi + static_cast<std::size_t>(gj) * n)] = v.real();
            encoded[2 * (gi + static_cast<std::size_t>(gj) * n) + 1] = v.imag();
        }
        for (int i = nr; i < ldz; ++i)
            padding_bad += z[i + static_cast<std::size_t>(j) * ldz] != sentinel;
    }
    // DeviceBuffer reserves one element even for zero owned columns. Such
    // ranks must leave that guard untouched rather than escaping the loop.
    if (nc == 0)
        for (const T value : z) padding_bad += value != sentinel;
    require_close(h, tag + " finite and sorted", bad, 0);
    require_close(h, tag + " padding", padding_bad, 0);
    MPI_Allreduce(MPI_IN_PLACE, encoded.data(), 2 * n * n, MPI_DOUBLE, MPI_SUM, ddlaGetCommunicator(h));
    auto v = [&](int i, int j) { const auto k = 2 * (i + static_cast<std::size_t>(j) * n); return std::complex<double>(encoded[k], encoded[k + 1]); };
    double norm2 = 0, residual2 = 0, ortho = 0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            norm2 += std::norm(wide(hermitian<T>(i, j)));
            std::complex<double> av = 0, dot = 0;
            for (int k = 0; k < n; ++k) {
                av += wide(hermitian<T>(i, k)) * v(k, j);
                dot += std::conj(v(k, i)) * v(k, j);
            }
            residual2 += std::norm(av - double(w[j]) * v(i, j));
            ortho = std::max(ortho, std::abs(dot - (i == j ? 1.0 : 0.0)));
        }
    }
    finite_metric(h, tag + " independent residual", std::sqrt(residual2) / (n * std::sqrt(norm2) * eps), 50);
    finite_metric(h, tag + " independent orthogonality", ortho / (n * eps), 50);
    // Existing helpers instantiate the compiled default BLAS backend. The
    // independent host checks above also cover explicit CPU in a dual build.
    if constexpr (B == ddla::default_backend_v) if(!shifted_columns) {
        upload(h, A.ptr, original);
        finite_metric(h, tag + " helper residual", eig_test::residual_metric(h, A.ptr, descA, Z.ptr, descZ, w.data(), n), 50);
        finite_metric(h, tag + " helper orthogonality", eig_test::orthogonality_metric(h, Z.ptr, descZ, n), 50);
    }
}

// mode0: exact compact A=Z; mode1: exact padded parent descriptor;
// mode2: partial spans; mode3: overlap only on rank0. Only storage outside
// BOTH active views is padding: stage1 may legitimately change A where Z's
// padding intersects A, and stage4/5 may change Z where A's padding intersects Z.
template <ddla::DdlaBackend B,typename T>
void check_aliased_pheevd(const ddla::DdlaHandle_t& h,char uplo,int n,int nb,int mode)
{
    using R=typename ddla::real_type<T>::type;
    const std::string nbw_value=std::to_string(nb);
    EnvGuard nbw("DDLA_EIG_NBW",nbw_value.c_str());
    const double eps=std::numeric_limits<R>::epsilon();
    const T sentinel=T(R(12345));
    int descA[ddla::DDLA_DLEN_],descZ[ddla::DDLA_DLEN_];
    const bool parent=mode!=0;
    ddlaDescInit(descA,h,n+(parent?5:0),n+(parent?7:0),nb,nb,0,0);
    if(parent) descA[ddla::DDLA_LLD_]+=7;
    std::copy(descA,descA+ddla::DDLA_DLEN_,descZ);
    const std::size_t asize=std::max<std::size_t>(local_size(h,descA),1),zsize=std::max<std::size_t>(local_size(h,descZ),1);
    const std::size_t offset=mode==2?3:(mode==3 && ddlaGetRank(h)!=0?asize+7:0);
    const std::size_t total=std::max(asize,offset+zsize)+7;
    const int ar=ddla_test::m_loc(h,descA),ac=ddla_test::n_loc(h,descA),lda=descA[ddla::DDLA_LLD_];
    const int zr=ddla_test::m_loc(h,descZ),zc=ddla_test::n_loc(h,descZ),ldz=descZ[ddla::DDLA_LLD_];
    std::vector<T> before(total,sentinel);
    std::vector<unsigned char> active(total,0);
    for(int c=0;c<zc;++c) for(int r=0;r<zr;++r) {
        if(ddla::indx_l2g_r(descZ,h,r)<n && ddla::indx_l2g_c(descZ,h,c)<n) {
            const std::size_t at=offset+r+static_cast<std::size_t>(c)*ldz;
            before[at]=T(std::numeric_limits<R>::quiet_NaN());active[at]=1;
        }
    }
    // Fill A last so Z's poisoned logical output never damages the input.
    for(int c=0;c<ac;++c) for(int r=0;r<ar;++r) {
        const int i=ddla::indx_l2g_r(descA,h,r),j=ddla::indx_l2g_c(descA,h,c);
        if(i<n && j<n) {
            const std::size_t at=r+static_cast<std::size_t>(c)*lda;
            before[at]=(uplo=='U'?i>j:i<j)?T(R(1e6)):hermitian<T>(i,j);active[at]=1;
        }
    }
    DeviceBuffer<T> buffer(h,total);upload(h,buffer.ptr,before);
    T* A=buffer.ptr;T* Z=buffer.ptr+offset;
    std::vector<R> w(n,std::numeric_limits<R>::quiet_NaN());
    int info=-1;
    ddla::pheevd<B,T>(h,'V',uplo,n,A,descA,w.data(),Z,descZ,info);
    check_ddla_sync(h);
    const auto after=download(h,buffer.ptr,total);
    const std::string tag=std::string("pheevd alias ")+type_name<T>()+" "+uplo+" n="+std::to_string(n)+" mode="+std::to_string(mode);
    require_close(h,tag+" info",std::abs(info),0);
    double bad=0,padding=0;
    for(std::size_t at=0;at<total;++at)
        if(!active[at]) padding+=std::memcmp(&after[at],&before[at],sizeof(T))!=0;
    for(int j=0;j<n;++j) {
        if(!std::isfinite(double(w[j]))) ++bad;
        if(j && w[j]<w[j-1]) ++bad;
    }
    std::vector<double> encoded(static_cast<std::size_t>(2)*n*n,0);
    for(int c=0;c<zc;++c) for(int r=0;r<zr;++r) {
        const int i=ddla::indx_l2g_r(descZ,h,r),j=ddla::indx_l2g_c(descZ,h,c);
        if(i<n && j<n) {
            const auto v=wide(after[offset+r+static_cast<std::size_t>(c)*ldz]);
            if(!std::isfinite(v.real()) || !std::isfinite(v.imag())) ++bad;
            const std::size_t at=2*(i+static_cast<std::size_t>(j)*n);
            encoded[at]=v.real();encoded[at+1]=v.imag();
        }
    }
    require_close(h,tag+" finite and sorted",bad,0);
    require_close(h,tag+" padding outside active union",padding,0);
    MPI_Allreduce(MPI_IN_PLACE,encoded.data(),2*n*n,MPI_DOUBLE,MPI_SUM,ddlaGetCommunicator(h));
    auto value=[&](int i,int j) {const std::size_t at=2*(i+static_cast<std::size_t>(j)*n);return std::complex<double>(encoded[at],encoded[at+1]);};
    double norm2=0,residual2=0,orthogonality=0;
    for(int j=0;j<n;++j) for(int i=0;i<n;++i) {
        norm2+=std::norm(wide(hermitian<T>(i,j)));
        std::complex<double> av=0,dot=0;
        for(int k=0;k<n;++k) {
            av+=wide(hermitian<T>(i,k))*value(k,j);
            dot+=std::conj(value(k,i))*value(k,j);
        }
        residual2+=std::norm(av-double(w[j])*value(i,j));
        orthogonality=std::max(orthogonality,std::abs(dot-(i==j?1.0:0.0)));
    }
    finite_metric(h,tag+" independent residual",std::sqrt(residual2)/(n*std::sqrt(norm2)*eps),50);
    finite_metric(h,tag+" independent orthogonality",orthogonality/(n*eps),50);
    if(ddlaGetRank(h)==0) std::cout<<tag<<" PASS"<<std::endl;
}

template <ddla::DdlaBackend B,typename T>
void check_row_layout_rejections(const ddla::DdlaHandle_t& h)
{
    using R=typename ddla::real_type<T>::type;
    int npr=0,npc=0,P=0;ddlaGetGridDims(h,npr,npc);MPI_Comm_size(ddlaGetCommunicator(h),&P);
    // Null matrix pointers are deliberate: each individually valid descriptor
    // combination must be rejected before an allocation, copy or collective.
    auto reject=[&](int n,int amb,int anb,int ar,int ac,int zmb,int zr,bool driver,const char* label) {
        int da[9],dz[9],info=-1,caught=0;
        ddlaDescInit(da,h,n,n,amb,anb,ar,ac);ddlaDescInit(dz,h,n,n,zmb,zmb,zr,0);
        try {
            if(driver)
                ddla::pheevd<B,T>(h,'V','U',n,static_cast<T*>(nullptr),da,static_cast<R*>(nullptr),static_cast<T*>(nullptr),dz,info);
            else
                ddla::punmtr_he2hb<B,T>(h,'C',n,1,static_cast<const T*>(nullptr),da,static_cast<const T*>(nullptr),static_cast<T*>(nullptr),dz,info);
        } catch(const std::exception&) {caught=1;}
        int sum=0;MPI_Allreduce(&caught,&sum,1,MPI_INT,MPI_SUM,ddlaGetCommunicator(h));
        require_close(h,std::string(label)+(driver?" driver":" standalone"),std::abs(sum-P),0);
    };
    for(bool driver:{false,true}) {
        reject(9,4,4,0,0,8,0,driver,"mismatched row block");
        if(npr>1) reject(8,4,4,0,0,4,1,driver,"equal-extent wrong row ownership");
    }
    reject(9,4,8,0,0,4,0,false,"non-square reflector blocks");
    if(npr>1) reject(9,4,4,1,0,4,1,false,"nonzero reflector row source");
    if(npc>1) reject(9,4,4,0,1,4,0,false,"nonzero reflector column source");
}

template <ddla::DdlaBackend B,typename T>
void check_standalone_column_layout(const ddla::DdlaHandle_t& h)
{
    using R=typename ddla::real_type<T>::type;
    constexpr int n=9,nb=4,nbw=1,jref=5,anchor=jref-nbw;
    int npr=0,npc=0;ddlaGetGridDims(h,npr,npc);
    int da[9],dz[9];ddlaDescInit(da,h,n+3,n+7,nb,nb,0,0);
    ddlaDescInit(dz,h,n+5,n+9,nb,3,0,npc-1);da[ddla::DDLA_LLD_]+=5;dz[ddla::DDLA_LLD_]+=7;
    const T sentinel=T(R(12345));
    std::vector<T> v(n,T(0));
    for(int r=0;r<anchor;++r) {
        const R real=R((r*3)%7-3)/R(8);
        if constexpr(std::is_same<T,R>::value) v[r]=real;
        else v[r]=T(real,R((r*5)%5-2)/R(16));
    }
    v[anchor]=T(1);double norm=0;for(const auto x:v) norm+=std::norm(wide(x));
    const T tau=T(R(2/norm));
    auto a=make_local<T>(h,da,[&](int r,int c) {
        if(r>=n || c>=n) return sentinel;
        return c==jref && r<anchor?v[r]:T(0);
    });
    auto z=make_local<T>(h,dz,[&](int r,int c) {return r<n && c<n?hermitian<T>(r,c):sentinel;});
    // make_local's physical LLD padding is not initialized by the callback.
    const int ar=ddla_test::m_loc(h,da),ac=ddla_test::n_loc(h,da),zr=ddla_test::m_loc(h,dz),zc=ddla_test::n_loc(h,dz);
    for(int c=0;c<ac;++c) for(int r=ar;r<da[ddla::DDLA_LLD_];++r) a[r+static_cast<std::size_t>(c)*da[ddla::DDLA_LLD_]]=sentinel;
    for(int c=0;c<zc;++c) for(int r=zr;r<dz[ddla::DDLA_LLD_];++r) z[r+static_cast<std::size_t>(c)*dz[ddla::DDLA_LLD_]]=sentinel;
    if(a.empty()) a.push_back(sentinel);if(z.empty()) z.push_back(sentinel);
    std::vector<T> t(n-1,T(0));t[jref-1]=tau;
    DeviceBuffer<T> A(h,a.size()),Z(h,z.size()),Tmat(h,t.size());upload(h,A.ptr,a);upload(h,Tmat.ptr,t);
    for(char trans:{'N','C'}) {
        upload(h,Z.ptr,z);int info=-1;
        ddla::punmtr_he2hb<B,T>(h,trans,n,nbw,A.ptr,da,Tmat.ptr,Z.ptr,dz,info);check_ddla_sync(h);
        const auto after=download(h,Z.ptr,z.size()),a_after=download(h,A.ptr,a.size());
        double bad=0,padding=0,error=0,scale=1;
        for(int c=0;c<zc;++c) for(int r=0;r<dz[ddla::DDLA_LLD_];++r) {
            const int gr=r<zr?ddla::indx_l2g_r(dz,h,r):n,gc=ddla::indx_l2g_c(dz,h,c);
            const std::size_t at=r+static_cast<std::size_t>(c)*dz[ddla::DDLA_LLD_];
            if(gr>=n || gc>=n) {padding+=after[at]!=z[at];continue;}
            std::complex<double> dot=0;for(int k=0;k<n;++k) dot+=std::conj(wide(v[k]))*wide(hermitian<T>(k,gc));
            const auto expected=wide(hermitian<T>(gr,gc))-wide(tau)*wide(v[gr])*dot,actual=wide(after[at]);
            if(!std::isfinite(actual.real()) || !std::isfinite(actual.imag())) ++bad;
            error=std::max(error,std::abs(actual-expected));scale=std::max(scale,std::abs(expected));
        }
        if(zc==0) padding+=std::memcmp(after.data(),z.data(),z.size()*sizeof(T))!=0;
        const std::string tag=std::string("standalone column layout ")+type_name<T>()+" "+trans;
        require_close(h,tag+" info",std::abs(info),0);require_close(h,tag+" finite",bad,0);require_close(h,tag+" padding",padding,0);
        require_close(h,tag+" reflectors unchanged",std::memcmp(a.data(),a_after.data(),a.size()*sizeof(T))!=0,0);
        finite_metric(h,tag+" independent Householder",error/(n*std::numeric_limits<R>::epsilon()*scale),50);
    }
}

template <ddla::DdlaBackend B>
void run_checks(const ddla::DdlaHandle_t& h)
{
    EnvGuard leaf("DDLA_EIG_DC_LEAF", "32");
    check_row_layout_rejections<B,float>(h);check_row_layout_rejections<B,double>(h);
    check_row_layout_rejections<B,std::complex<float>>(h);check_row_layout_rejections<B,std::complex<double>>(h);
    check_standalone_column_layout<B,float>(h);check_standalone_column_layout<B,double>(h);
    check_standalone_column_layout<B,std::complex<float>>(h);check_standalone_column_layout<B,std::complex<double>>(h);
    for(char uplo:{'U','L'}) {
        check_padded_pheevd<B,float>(h,uplo,17,4,true);check_padded_pheevd<B,double>(h,uplo,17,4,true);
        check_padded_pheevd<B,std::complex<float>>(h,uplo,17,4,true);check_padded_pheevd<B,std::complex<double>>(h,uplo,17,4,true);
    }
    for (int n : {129, 257})
        for (int kind : {0, 1, 2}) {
            check_rows<float>(h, n, kind);
            check_rows<double>(h, n, kind);
        }
    check_rows<float>(h, 2, 0);
    check_rows<double>(h, 2, 0);
    int P = 0;
    MPI_Comm_size(ddlaGetCommunicator(h), &P);
    const ddla::detail::RowShare share(67, P);
    int desc[ddla::DDLA_DLEN_];
    ddlaDescInit(desc, h, 67, 67, 8, 8, 0, 0);
    int local_fallback = ddla_test::m_loc(h, desc) * ddla_test::n_loc(h, desc) > share.count(ddlaGetRank(h)) * 67;
    int fallbacks = 0;
    MPI_Allreduce(&local_fallback, &fallbacks, 1, MPI_INT, MPI_SUM, ddlaGetCommunicator(h));
    if (ddlaGetRank(h) == 0) std::cout << "receive capacity fallback ranks=" << fallbacks << std::endl;
    if (P > 1) require_close(h, "uneven output exercises receive fallback", fallbacks == 0 ? 1 : 0, 0);
    for (const auto problem : {std::make_pair(67, 8), std::make_pair(2, 1)})
        for (char uplo : {'U', 'L'}) {
            check_padded_pheevd<B, float>(h, uplo, problem.first, problem.second);
            check_padded_pheevd<B, double>(h, uplo, problem.first, problem.second);
            check_padded_pheevd<B, std::complex<float>>(h, uplo, problem.first, problem.second);
            check_padded_pheevd<B, std::complex<double>>(h, uplo, problem.first, problem.second);
        }
    // Dense cases have n>nbw, so stage5 really needs the saved reflectors.
    for(int mode:{0,1,2,3}) for(char uplo:{'U','L'}) {
        check_aliased_pheevd<B,float>(h,uplo,67,8,mode);
        check_aliased_pheevd<B,double>(h,uplo,67,8,mode);
        check_aliased_pheevd<B,std::complex<float>>(h,uplo,67,8,mode);
        check_aliased_pheevd<B,std::complex<double>>(h,uplo,67,8,mode);
    }
    // Logical n below the process count exercises empty matrix owners.
    for(char uplo:{'U','L'}) {
        check_aliased_pheevd<B,float>(h,uplo,2,1,0);
        check_aliased_pheevd<B,double>(h,uplo,2,1,0);
        check_aliased_pheevd<B,std::complex<float>>(h,uplo,2,1,0);
        check_aliased_pheevd<B,std::complex<double>>(h,uplo,2,1,0);
    }
}

} // namespace

int main(int argc, char** argv)
{
    int provided=0;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    try {
        if(provided<MPI_THREAD_FUNNELED) throw std::runtime_error("MPI thread support");
        int P = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &P);
        TestOptions opts;
        std::string error;
        if (!parse_options(argc, argv, P, opts, error)) throw std::runtime_error(error);
        auto backend = ddla::default_backend_v;
        if (const char* value = std::getenv("DDLA_TEST_BACKEND")) {
            const std::string name(value);
            if (name == "CPU" || name == "cpu") backend = ddla::DdlaBackend::CPU;
            else if (name == "GPU" || name == "gpu") backend = ddla::DdlaBackend::GPU;
            else throw std::runtime_error("DDLA_TEST_BACKEND must be CPU or GPU");
        }
        for (const auto grid : opts.grids) {
            ddla::DdlaHandle_t h = nullptr;
            ddla::ddlaInit(h, backend);
            ddla::ddlaSet(h, MPI_COMM_WORLD, grid.first, grid.second);
            if (ddlaGetRank(h) == 0) std::cout << "=== workspace grid " << grid_name(h) << " ===" << std::endl;
            switch (backend) {
            case ddla::DdlaBackend::CPU: run_checks<ddla::DdlaBackend::CPU>(h); break;
#if DDLA_HAS_GPU
            case ddla::DdlaBackend::GPU: run_checks<ddla::DdlaBackend::GPU>(h); break;
#endif
            default: throw std::runtime_error("requested backend is unavailable");
            }
            check_ddla_sync(h);
            ddla::ddlaDestroy(h);
        }
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) std::cout << "test_pheevd_workspace passed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "test_pheevd_workspace exception: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return 0;
}
