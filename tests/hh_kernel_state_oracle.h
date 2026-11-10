#ifndef DDLA_HH_KERNEL_STATE_ORACLE_H
#define DDLA_HH_KERNEL_STATE_ORACLE_H
#include "Eigensolver/two_stage/hb2st_householder_cpu.h"
#include "Eigensolver/two_stage/hb2st_householder_paired.h"
#include "mpi_datatype.h"
#include "ddla_connector.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
namespace hh_kernel_state_oracle {
namespace hh=ddla::detail::hh;namespace hhpair=ddla::detail::hhpair;
struct Result {double similarity=0,orthogonality=0,difference=0;std::size_t applications=0,inactive=0;std::vector<unsigned char>first_tiled_output;};
inline void require(bool pass,const char*message){if(!pass)throw std::runtime_error(message);}
template<class T>std::complex<long double> wide(T x){return {static_cast<long double>(x),0};}
template<class R>std::complex<long double> wide(std::complex<R>x){return {x.real(),x.imag()};}
template<class T>T value(int i,int j,int b) {
    using R=hh::real_t<T>;if(std::abs(i-j)>b)return T(0);if(i<j)return hh::adj(value<T>(j,i,b));
    const R re=R(i==j?2+0.01*i:std::sin((i+1)*0.719+(j+1)*0.131));
    if constexpr(std::is_same<T,R>::value)return re;
    else return T(re,R(i==j?0:std::cos((i+1)*0.293-(j+1)*0.817)));
}
template<class T>Result run(const ddla::DdlaHandle_t& h,int n,int b,const int*desc,const T*A,
                           ddla::detail::Hb2stHouseholderCpuOptions::Stage2 stage2,
                           ddla::detail::Hb2stHouseholderCpuOptions::Stage4 method) {
    using R=hh::real_t<T>;using Options=ddla::detail::Hb2stHouseholderCpuOptions;
    const int P=ddla::ddlaGetSize(h),rank=ddla::ddlaGetRank(h),ncols=n/P+(rank<n%P),first=rank*(n/P)+std::min(rank,n%P);
    const int ld=std::max(n,1)+3;const R sentinel=std::numeric_limits<R>::quiet_NaN();
    Options options;options.stage2=stage2;options.stage4=method;
    ddla::detail::Hb2stHouseholderCpu<T> state(h,n,b,true,options);Result result;
    try {
        require(state.paired_replay()==(method!=Options::Stage4::Wy),"tiled selector");
        std::vector<R>d(std::max(n,1)),e(std::max(n,1));std::vector<T>phases;
        state.reduce(A,desc,d.data(),e.data(),phases);const R scale=state.eigenvalue_scale();
        auto phase=[&](int row){return row&&!phases.empty()?phases[row-1]:T(1);};
        auto columns=[&]{std::vector<T>x(std::max<std::size_t>(1,static_cast<std::size_t>(ld)*ncols),T(R(139)));for(int c=0;c<ncols;++c)for(int r=0;r<n;++r)x[r+static_cast<std::size_t>(c)*ld]=r==first+c?phase(r):T(0);return x;};
        auto gather=[&](const std::vector<T>&x,int pitch){std::vector<T>q(static_cast<std::size_t>(n)*n,T(0));for(int c=0;c<ncols;++c)for(int r=0;r<n;++r)q[r+static_cast<std::size_t>(first+c)*n]=x[r+static_cast<std::size_t>(c)*pitch];if(P>1)ddla::MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,q.data(),n*n,ddla::detail::mpi_datatype<T>(),MPI_SUM,ddla::ddlaGetCommunicator(h)));return q;};
        auto check=[&](const std::vector<T>&q,const std::vector<T>*reference){
            long double residual=0,orth=0,norm=0,diff=0;
            for(int c=0;c<n;++c)for(int r=0;r<n;++r) {
                require(hh::finite(q[r+static_cast<std::size_t>(c)*n]),"non-finite shared-state output");
                std::complex<long double> aq=0,dot=0,qt=wide(q[r+static_cast<std::size_t>(c)*n])*static_cast<long double>(d[c]);
                for(int k=0;k<n;++k){aq+=wide(hh::divide_real(value<T>(r,k,b),scale))*wide(q[k+static_cast<std::size_t>(c)*n]);dot+=std::conj(wide(q[k+static_cast<std::size_t>(r)*n]))*wide(q[k+static_cast<std::size_t>(c)*n]);}
                if(c)qt+=wide(q[r+static_cast<std::size_t>(c-1)*n])*static_cast<long double>(e[c-1]);
                if(c+1<n)qt+=wide(q[r+static_cast<std::size_t>(c+1)*n])*static_cast<long double>(e[c]);
                residual+=std::norm(aq-qt);orth+=std::norm(dot-(r==c?1.L:0.L));norm+=std::norm(wide(hh::divide_real(value<T>(r,c,b),scale)));
                if(reference)diff+=std::norm(wide(q[r+static_cast<std::size_t>(c)*n])-wide((*reference)[r+static_cast<std::size_t>(c)*n]));
            }
            const long double denom=std::max(n,1)*std::numeric_limits<R>::epsilon();
            const double rm=std::sqrt(residual)/(denom*std::max(1.L,std::sqrt(norm))),om=std::sqrt(orth)/denom,dm=std::sqrt(diff)/denom;
            require(std::isfinite(rm+om+dm)&&rm<=50&&om<=50&&dm<=50,"paired shared-state numerical oracle");
            result.similarity=std::max(result.similarity,rm);result.orthogonality=std::max(result.orthogonality,om);result.difference=std::max(result.difference,dm);++result.applications;
        };
        auto reference_columns=columns();state.apply_columns(reference_columns.data(),ld,ncols);const auto reference=gather(reference_columns,ld);check(reference,nullptr);
        // Empty strips still enter both publication/retirement batches.
        R empty=R(173);state.apply_tiles(&empty,8,0);
        require(empty==R(173),"paired collective empty strip changed");++result.applications;
        for(int width:{8,16,32,64}) {
            std::size_t elements=0;require(hhpair::layout_elements(n,ncols,width,hhpair::planes<T>,elements),"paired test layout overflow");
            std::vector<R> tiles(std::max<std::size_t>(1,elements),sentinel);
            const std::size_t stride=static_cast<std::size_t>(n)*width*hhpair::planes<T>;
            for(int c=0;c<ncols;++c)for(int r=0;r<n;++r) {
                const T x=r==first+c?phase(r):T(0);R*at=tiles.data()+static_cast<std::size_t>(c/width)*stride+static_cast<std::size_t>(r)*width*hhpair::planes<T>+c%width;
                *at=hh::real_part(x);if constexpr(!std::is_same<T,R>::value)at[width]=x.imag();
            }
            state.apply_tiles(tiles.data(),width,ncols);
            std::vector<T> packed(std::max<std::size_t>(1,static_cast<std::size_t>(n)*ncols),T(0));
            for(int c=0;c<ncols;++c)for(int r=0;r<n;++r) {
                const R*at=tiles.data()+static_cast<std::size_t>(c/width)*stride+static_cast<std::size_t>(r)*width*hhpair::planes<T>+c%width;
                if constexpr(std::is_same<T,R>::value)packed[r+static_cast<std::size_t>(c)*n]=*at;
                else packed[r+static_cast<std::size_t>(c)*n]=T(*at,at[width]);
            }
            const int nt=ncols/width+(ncols%width!=0);
            for(int t=0;t<nt;++t)for(int r=0;r<n;++r)for(int component=0;component<hhpair::planes<T>;++component)
                for(int lane=std::min(width,ncols-t*width);lane<width;++lane) {
                    const R*at=tiles.data()+static_cast<std::size_t>(t)*stride+static_cast<std::size_t>(r)*width*hhpair::planes<T>+component*width+lane;
                    require(std::memcmp(at,&sentinel,sizeof(R))==0,"paired state overwrote inactive tile lane");++result.inactive;
                }
            if(ncols==0)require(std::memcmp(tiles.data(),&sentinel,sizeof(R))==0,"paired empty strip changed");
            const auto actual=gather(packed,std::max(n,1));
            if(width==8) {
                result.first_tiled_output.resize(actual.size()*sizeof(T));
                if(!actual.empty())std::memcpy(result.first_tiled_output.data(),actual.data(),actual.size()*sizeof(T));
            }
            check(actual,&reference);
            auto repeated=columns();state.apply_columns(repeated.data(),ld,ncols);check(gather(repeated,ld),&reference);
        }
        state.release();
    }catch(...){
        // Test assertions may fail on one rank. The native main aborts rather
        // than entering a collective cleanup while peers are still checking.
        throw;
    }
    return result;
}
}
#endif
