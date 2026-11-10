#ifndef DDLA_HH_BLAS_EVENT_ORACLE_H
#define DDLA_HH_BLAS_EVENT_ORACLE_H
#include "Eigensolver/two_stage/hb2st_householder_cpu.h"
#include "Eigensolver/two_stage/hb2st_householder_kernels.h"
#include "hh_scalar_before_blas_reference.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hh_blas_oracle {
namespace hh=ddla::detail::hh;
namespace old=ddla::detail::hh_scalar_reference;
struct Counts {std::uint64_t events=0,mapping=0,unchanged=0,scalar_values=0;double error=0;};
inline void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
template<class T>T value(double re,double im){if constexpr(std::is_same<T,hh::real_t<T>>::value)return T(re);else return T(hh::real_t<T>(re),hh::real_t<T>(im));}
template<class T> bool same(T a,T b){return std::memcmp(&a,&b,sizeof(T))==0;}

template<class T>
Counts run(const ddla::DdlaHandle_t& handle) {
    using R=hh::real_t<T>;Counts counts;
    for(auto shape:std::vector<std::pair<int,int>>{{3,2},{9,2},{17,4},{33,8},{65,13},{97,16},{65,32},{129,8}})
    for(int padding:{0,3}) {
        const int n=shape.first,b=shape.second,ld=hh::band_rows(n,b)+padding;
        const T poison=value<T>(std::numeric_limits<R>::quiet_NaN(),std::numeric_limits<R>::quiet_NaN());
        std::vector<T> reference(static_cast<std::size_t>(n)*ld,poison);
        for(int c=0;c<n;++c)for(int r=0;r<std::min(n-c,hh::band_rows(n,b));++r)
            reference[static_cast<std::size_t>(c)*ld+r]=r>b?T(0):value<T>(r?std::sin((c+r+1)*0.719+(c+1)*0.131):2+0.01*c,r?std::cos((c+r+1)*0.293-(c+1)*0.817):0);
        std::vector<T> scalar=reference;
        std::vector<T> oldv(b),olda(b),oldw(b),v(b),av(b),w(b),bv(b),bc(b),bw(b);
        old::BandView<T> old_band{reference.data(),n,b,ld};
        hh::BandView<T> scalar_band{scalar.data(),n,b,ld};
        for(int i=0;i<n-2;++i)for(int j=0;;++j) {
            hh::Event event;if(!hh::event_geometry(n,b,i,j,event))break;
            old::Event old_event;require(old::event_geometry(n,b,i,j,old_event),"old event geometry");
            const auto before=reference;std::vector<T> blas=before;hh::BandView<T> blas_band{blas.data(),n,b,ld};
            hh::DenseEventViews<T> views;require(hh::dense_event_views(blas_band,event,views),"dense event views rejected valid geometry");
            std::vector<unsigned char> write_mask(before.size(),0);
            auto mark=[&](T* base,int rows,int cols,bool lower) {
                const auto start=static_cast<std::size_t>(base-blas.data());
                for(int c=0;c<cols;++c)for(int r=lower?c:0;r<rows;++r) {
                    const auto at=start+r+static_cast<std::size_t>(c)*views.lda;
                    int gr=0,gc=0;
                    if(base==views.diagonal){gr=event.start+r;gc=event.start+c;}
                    else if(base==views.left){gr=event.start+r;gc=event.left+c;}
                    else {gr=event.start+event.length+r;gc=event.start+c;}
                    const auto packed=static_cast<std::size_t>(gc)*ld+gr-gc;
                    require(at==packed,"dense view does not map to packed lower entry");
                    require(at<before.size()&&gr>=gc&&gr<n&&gc<n,"dense footprint outside storage");
                    write_mask[at]=1;++counts.mapping;
                }
            };
            mark(views.diagonal,event.length,event.length,true);
            mark(views.left,event.length,views.left_columns,false);
            if(views.right_rows)mark(views.right,views.right_rows,event.length,false);
            // The scalar refactor must remain bitwise identical, including
            // padding and record fields, to the frozen pre-BLAS kernel.
            R old_tau=0,tau=0,blas_tau=0;
            require(old::reduce_event(old_band,old_event,oldv.data(),olda.data(),oldw.data(),old_tau),"old scalar event failed");
            require(hh::reduce_event(scalar_band,event,v.data(),av.data(),w.data(),tau),"refactored scalar event failed");
            require(std::memcmp(reference.data(),scalar.data(),reference.size()*sizeof(T))==0,"scalar band changed after generator extraction");
            require(same(old_tau,tau),"scalar tau changed after generator extraction");
            for(int r=0;r<event.length;++r)require(same(oldv[r],v[r]),"scalar vector changed after generator extraction");
            counts.scalar_values+=reference.size()+event.length+1;
            // Poison beta=0 outputs before each call. No BLAS input may depend
            // on their prior contents or on the bogus dense upper triangle.
            std::fill(bc.begin(),bc.end(),poison);std::fill(bw.begin(),bw.end(),poison);
            require(ddla::detail::hh_event_blas_cpu<T>(handle,blas_band,event,bv.data(),bc.data(),bw.data(),blas_tau),"BLAS event failed");
            require(same(tau,blas_tau),"BLAS event did not reuse generator tau");
            for(int r=0;r<event.length;++r)require(same(v[r],bv[r]),"BLAS event did not reuse generator vector");
            double norm=1;
            for(const auto x:before)if(hh::finite(x))norm=std::max(norm,static_cast<double>(std::abs(x)));
            for(std::size_t q=0;q<before.size();++q) {
                if(!write_mask[q]) {
                    require(same(blas[q],before[q]),"BLAS wrote outside the permitted footprint");
                    require(same(reference[q],before[q]),"scalar footprint oracle is incomplete");++counts.unchanged;
                } else {
                    require(hh::finite(blas[q])&&hh::finite(reference[q]),"non-finite updated band");
                    const double error=static_cast<double>(std::abs(blas[q]-reference[q]))/(b*std::numeric_limits<R>::epsilon()*norm);
                    require(std::isfinite(error)&&error<=50,"BLAS event differs from scalar similarity update");counts.error=std::max(counts.error,error);
                }
            }
            // Invalid views must reject before dereferencing an address.
            auto bad_event=event;++bad_event.stop;hh::DenseEventViews<T> invalid;
            require(!hh::dense_event_views(blas_band,bad_event,invalid),"invalid event accepted");
            auto bad_band=blas_band;bad_band.ld=1;
            require(!hh::dense_event_views(bad_band,event,invalid),"invalid leading dimension accepted");
            bad_band=blas_band;bad_band.data=nullptr;
            require(!hh::dense_event_views(bad_band,event,invalid),"null band accepted");
            ++counts.events;
        }
    }
    return counts;
}
}
#endif
