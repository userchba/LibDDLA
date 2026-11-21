#include "selected_tridiagonal_core.h"
#include "row_share.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

// Same native LP64 Fortran ABI as LibDDLA's pstedc_numeric/STERF wrappers.
extern "C" {
#define NATIVE(P,R) \
void P##sterf_(const int*,R*,R*,int*); \
void P##stebz_(const char*,const char*,const int*,const R*,const R*,const int*,const int*,const R*,const R*,const R*,int*,int*,R*,int*,int*,R*,int*,int*); \
void P##lagtf_(const int*,R*,const R*,R*,R*,const R*,R*,int*,int*); \
void P##lagts_(const int*,const int*,const R*,const R*,const R*,const R*,const int*,R*,R*,int*); \
R P##dot_(const int*,const R*,const int*,const R*,const int*); \
void P##axpy_(const int*,const R*,const R*,const int*,R*,const int*)
NATIVE(s,float);
NATIVE(d,double);
#undef NATIVE
}
namespace ddla { namespace spectral_impl { namespace selected {
namespace {
template<class R> MPI_Datatype datatype() { return std::is_same<R,float>::value?MPI_FLOAT:MPI_DOUBLE; }
template<class R> R dot(int n,const R* x,const R* y) {
    if (!n) return R(0);
    const int one=1;
    if constexpr (std::is_same<R,float>::value) return sdot_(&n,x,&one,y,&one);
    else return ddot_(&n,x,&one,y,&one);
}
template<class R> void axpy(int n,R alpha,const R* x,R* y) {
    if (!n) return;
    const int one=1;
    if constexpr (std::is_same<R,float>::value) saxpy_(&n,&alpha,x,&one,y,&one);
    else daxpy_(&n,&alpha,x,&one,y,&one);
}
template<class R> void sterf(int n,R* d,R* e,int& info) {
    if constexpr (std::is_same<R,float>::value) ssterf_(&n,d,e,&info);
    else dsterf_(&n,d,e,&info);
}
template<class R> void stebz(int n,int k,const R* d,const R* e,int& m,int& ns,
    R* w,int* blocks,int* splits,R* work,int* iw,int& info) {
    const char range='I',order='B';
    const R zero=0,abstol=R(2)*std::numeric_limits<R>::min();
    const int one=1;
    if constexpr (std::is_same<R,float>::value)
        sstebz_(&range,&order,&n,&zero,&zero,&one,&k,&abstol,d,e,&m,&ns,w,blocks,splits,work,iw,&info);
    else dstebz_(&range,&order,&n,&zero,&zero,&one,&k,&abstol,d,e,&m,&ns,w,blocks,splits,work,iw,&info);
}
template<class R> void lagtf(int n,R* a,R shift,R* b,R* c,R* d,int* piv,int& info) {
    const R tolerance=0;
    if constexpr (std::is_same<R,float>::value) slagtf_(&n,a,&shift,b,c,&tolerance,d,piv,&info);
    else dlagtf_(&n,a,&shift,b,c,&tolerance,d,piv,&info);
}
template<class R> void lagts(int n,const R* a,const R* b,const R* c,const R* d,
    const int* piv,R* x,R& tolerance,int& info) {
    const int job=-1;
    if constexpr (std::is_same<R,float>::value) slagts_(&job,&n,a,b,c,d,piv,x,&tolerance,&info);
    else dlagts_(&job,&n,a,b,c,d,piv,x,&tolerance,&info);
}
std::size_t add_region(std::size_t& bytes,std::size_t count,std::size_t element) {
    const auto max=std::numeric_limits<std::size_t>::max();
    if (count>(max-63)/element) throw std::length_error("selected: workspace overflow");
    const std::size_t rounded=(count*element+63)/64*64,offset=bytes;
    if (rounded>max-bytes) throw std::length_error("selected: workspace overflow");
    bytes+=rounded;
    return offset;
}
bool overlap(const void* x,std::size_t a,const void* y,std::size_t b) {
    if (!a || !b) return false;
    const auto p=reinterpret_cast<std::uintptr_t>(x),q=reinterpret_cast<std::uintptr_t>(y);
    return p<q?q-p<a:p-q<b;
}
template<class R> double tolerance(const Plan& p) {
    return 64*double(std::numeric_limits<R>::epsilon())*std::max(1.0,std::sqrt(double(p.n)));
}
template<class R> double norm(int n,const R* x) {
    long double result=0;
    for (int i=0;i<n;++i) result=std::hypot(result,static_cast<long double>(x[i]));
    return double(result);
}
template<class R> R norm1(int n,const R* d,const R* e) {
    R result=0;
    for (int i=0;i<n;++i)
        result=std::max(result,std::abs(d[i])+(i?std::abs(e[i-1]):R(0))+(i+1<n?std::abs(e[i]):R(0)));
    return result;
}
} // namespace

template<class R>
Plan check(MPI_Comm comm,int n,int nev,const R* d,const R* e,R* w,R* Y,int ldy,
    int max_iterations,int max_restarts) {
    static_assert(std::is_same<R,float>::value || std::is_same<R,double>::value,"real precision only");
    if (comm==MPI_COMM_NULL) throw std::invalid_argument("selected: null communicator");
    Plan p;
    p.comm=comm; MPI_Comm_rank(comm,&p.rank); MPI_Comm_size(comm,&p.ranks);
    int arguments[5]={n,nev,max_iterations,max_restarts,int(sizeof(R))},low[5],high[5];
    MPI_Allreduce(arguments,low,5,MPI_INT,MPI_MIN,comm);
    MPI_Allreduce(arguments,high,5,MPI_INT,MPI_MAX,comm);
    bool bad=n<0 || n>INT_MAX/30 || nev<0 || nev>n || max_iterations<0 || max_restarts<1;
    for (int i=0;i<5;++i) bad=bad || low[i]!=high[i];
    if (n>=0) {
        p.rows=::ddla::detail::RowShare(n,p.ranks).count(p.rank); p.row0=::ddla::detail::RowShare(n,p.ranks).lo(p.rank);
        bad=bad || ldy<std::max(1,p.rows) || (n && (!d || !w || (n>1 && !e))) ||
            (p.rows && nev && !Y);
        if (n<=INT_MAX/30 && nev>=0 && nev<=n && ldy>=std::max(1,p.rows)) {
            const auto span=p.rows && nev?(std::size_t(nev-1)*ldy+p.rows)*sizeof(R):0;
            bad=bad || overlap(w,std::size_t(n)*sizeof(R),Y,span);
        }
    }
    int error=bad?1:0; MPI_Allreduce(MPI_IN_PLACE,&error,1,MPI_INT,MPI_MAX,comm);
    if (error) throw std::invalid_argument("selected: inconsistent or invalid geometry/pointers/output overlap");
    return geometry<R>(comm,n,nev,p.rank,p.ranks,ldy,max_iterations,max_restarts);
}

template<class R>
Plan geometry(MPI_Comm comm,int n,int nev,int rank,int ranks,int ldy,
    int max_iterations,int max_restarts) {
    static_assert(std::is_same<R,float>::value || std::is_same<R,double>::value,"real precision only");
    if (comm==MPI_COMM_NULL || n<0 || n>INT_MAX/30 || nev<0 || nev>n ||
        ranks<1 || rank<0 || rank>=ranks || max_iterations<0 || max_restarts<1 ||
        ldy<std::max(1,::ddla::detail::RowShare(n,ranks).count(rank)))
        throw std::invalid_argument("selected: invalid prepared geometry");
    Plan p; p.comm=comm; p.rank=rank; p.ranks=ranks;
    p.rows=::ddla::detail::RowShare(n,ranks).count(rank); p.row0=::ddla::detail::RowShare(n,ranks).lo(rank);
    p.n=n; p.nev=nev; p.ldy=ldy; p.max_iterations=max_iterations; p.max_restarts=max_restarts;
    if (!n) return p;
    p.ds=add_region(p.bytes,n,sizeof(R)); p.es=add_region(p.bytes,n,sizeof(R));
    p.values=add_region(p.bytes,n,sizeof(R)); p.phase=add_region(p.bytes,std::size_t(5)*n,sizeof(R));
    p.blocks=add_region(p.bytes,n,sizeof(int)); p.splits=add_region(p.bytes,n,sizeof(int));
    p.order=add_region(p.bytes,n,sizeof(int)); p.integer_phase=add_region(p.bytes,std::size_t(3)*n,sizeof(int));
    p.counts=add_region(p.bytes,p.ranks,sizeof(int)); p.displacements=add_region(p.bytes,p.ranks,sizeof(int));
    return p;
}

template<class R>
Certificate certify_scaled(const Plan& p,const R* d,const R* e,R lambda,const R* x,const R* Q,int previous) {
    Certificate c;
    c.norm=norm(p.n,x);
    long double residual=0;
    bool finite=std::isfinite(c.norm) && std::isfinite(lambda);
    for (int i=0;i<p.n;++i) {
        finite=finite && std::isfinite(x[i]);
        const long double v=(static_cast<long double>(d[i])-lambda)*x[i]+
            (i?static_cast<long double>(e[i-1])*x[i-1]:0)+
            (i+1<p.n?static_cast<long double>(e[i])*x[i+1]:0);
        residual=std::hypot(residual,v);
    }
    const long double denominator=(static_cast<long double>(norm1(p.n,d,e))+std::abs(static_cast<long double>(lambda)))*c.norm;
    c.residual=denominator>0?double(residual/denominator):(residual==0?0:std::numeric_limits<double>::infinity());
    for (int j=0;j<previous;++j) {
        const R* column=p.rows?Q+std::size_t(j)*p.ldy:nullptr;
        R coefficient=dot(p.rows,x+p.row0,column);
        MPI_Allreduce(MPI_IN_PLACE,&coefficient,1,datatype<R>(),MPI_SUM,p.comm);
        finite=finite && std::isfinite(coefficient);
        c.orthogonality=std::isfinite(coefficient)?
            std::max(c.orthogonality,std::abs(double(coefficient))):std::numeric_limits<double>::infinity();
    }
    int accepted=finite && std::abs(c.norm-1)<=tolerance<R>(p) &&
        std::isfinite(c.residual) && c.residual<=tolerance<R>(p) &&
        c.orthogonality<=tolerance<R>(p)?1:0;
    MPI_Allreduce(MPI_IN_PLACE,&accepted,1,MPI_INT,MPI_MIN,p.comm);
    double metrics[3]={std::isfinite(c.norm)?c.norm:std::numeric_limits<double>::infinity(),
        std::isfinite(c.residual)?c.residual:std::numeric_limits<double>::infinity(),
        std::isfinite(c.orthogonality)?c.orthogonality:std::numeric_limits<double>::infinity()};
    MPI_Allreduce(MPI_IN_PLACE,metrics,3,MPI_DOUBLE,MPI_MAX,p.comm);
    c.norm=metrics[0]; c.residual=metrics[1]; c.orthogonality=metrics[2];
    c.accepted=accepted!=0;
    return c;
}

template<class R>
Status run(const Plan& p,const R* d,const R* e,R* w,R* Y,void* work) {
    Status status;
    const int n=p.n,k=p.nev;
    if (!n) return status;
    auto* base=static_cast<unsigned char*>(work);
    R* ds=reinterpret_cast<R*>(base+p.ds); R* es=reinterpret_cast<R*>(base+p.es);
    R* beta=reinterpret_cast<R*>(base+p.values); R* phase=reinterpret_cast<R*>(base+p.phase);
    int* blocks=reinterpret_cast<int*>(base+p.blocks); int* splits=reinterpret_cast<int*>(base+p.splits);
    int* order=reinterpret_cast<int*>(base+p.order); int* piv=reinterpret_cast<int*>(base+p.integer_phase);
    int* counts=reinterpret_cast<int*>(base+p.counts); int* disps=reinterpret_cast<int*>(base+p.displacements);
    // Snapshot BOTH logical inputs before any write through potentially aliased w.
    std::copy_n(d,n,ds); if (n>1) std::copy_n(e,n-1,es); es[n-1]=R(0);
    int control[3]={0,0,0};
    if (!p.rank) {
        R maximum=0;
        for (int i=0;i<n;++i) {
            if (!std::isfinite(ds[i]) || !std::isfinite(es[i])) control[0]=1;
            maximum=std::max({maximum,std::abs(ds[i]),std::abs(es[i])});
        }
        if (maximum>R(0) && !control[0]) std::frexp(maximum,&control[1]);
        if (!control[0]) for (int i=0;i<n;++i) {
            const R old_d=ds[i],old_e=es[i];
            ds[i]=std::scalbn(old_d,-control[1]); es[i]=std::scalbn(old_e,-control[1]);
            // A nonzero subnormal may round without becoming zero. Refuse
            // any scale that cannot recover the represented input exactly.
            if (!std::isfinite(ds[i]) || !std::isfinite(es[i]) ||
                std::scalbn(ds[i],control[1])!=old_d ||
                std::scalbn(es[i],control[1])!=old_e) control[0]=2;
        }
    }
    MPI_Bcast(control,2,MPI_INT,0,p.comm);
    status.scale_exponent=control[1];
    if (control[0]) {
        status.code=control[0]==2?Code::scaling_loss:Code::nonfinite_input;
        return status; // Only caller scratch has been written; w/Y untouched.
    }
    MPI_Bcast(ds,n,datatype<R>(),0,p.comm); MPI_Bcast(es,n,datatype<R>(),0,p.comm);
    if (!p.rank) {
        std::copy_n(ds,n,w); std::copy_n(es,n,phase);
        sterf(n,w,phase,control[0]);
    }
    MPI_Bcast(control,1,MPI_INT,0,p.comm);
    if (control[0]) { status.code=Code::sterf_failed; status.native_info=control[0]; return status; }
    MPI_Bcast(w,n,datatype<R>(),0,p.comm);
    const auto finish=[&](Status result) {
        bool finite=true;
        for (int i=0;i<n;++i) { w[i]=std::scalbn(w[i],status.scale_exponent); finite=finite && std::isfinite(w[i]); }
        if (!finite) result.code=Code::unrepresentable_values;
        return result;
    };
    if (!k) return finish(status);
    int m=0,ns=0,info=0;
    if (!p.rank) stebz(n,k,ds,es,m,ns,beta,blocks,splits,phase,piv,info);
    control[0]=info; control[1]=m; control[2]=ns;
    MPI_Bcast(control,3,MPI_INT,0,p.comm);
    if (control[0]) { status.code=Code::stebz_failed; status.native_info=control[0]; return finish(status); }
    if (control[1]!=k || control[2]<1 || control[2]>n) { status.code=Code::count_mismatch; return finish(status); }
    ns=control[2];
    MPI_Bcast(beta,k,datatype<R>(),0,p.comm); MPI_Bcast(blocks,k,MPI_INT,0,p.comm);
    MPI_Bcast(splits,ns,MPI_INT,0,p.comm);
    for (int j=0;j<k;++j) if (!std::isfinite(beta[j]) || blocks[j]<1 || blocks[j]>ns) {
        status.code=Code::count_mismatch; return finish(status);
    }
    for (int j=0;j<ns;++j) if (splits[j]<(j?splits[j-1]+1:1) || splits[j]>n || (j+1==ns && splits[j]!=n)) {
        status.code=Code::count_mismatch; return finish(status);
    }
    if (!p.rank) {
        for (int j=0;j<k;++j) order[j]=j;
        std::sort(order,order+k,[&](int a,int b) {
            if (beta[a]!=beta[b]) return beta[a]<beta[b];
            return blocks[a]!=blocks[b]?blocks[a]<blocks[b]:a<b;
        });
    }
    MPI_Bcast(order,k,MPI_INT,0,p.comm);
    const R tn=norm1(n,ds,es);
    for (int j=0;j<k;++j) {
        const int at=order[j];
        const R published=std::scalbn(w[j],status.scale_exponent);
        const R roundtrip=std::scalbn(published,-status.scale_exponent);
        if (!std::isfinite(published) || std::abs(double(roundtrip)-double(w[j]))>
            tolerance<R>(p)*(double(tn)+std::abs(double(roundtrip)))) {
            status.code=Code::unrepresentable_values; status.eigenvector=j; return finish(status);
        }
        if (blocks[at]<1 || blocks[at]>ns || !std::isfinite(beta[at]) ||
            std::abs(double(beta[at])-double(w[j]))>tolerance<R>(p)*(double(tn)+std::abs(double(w[j])))) {
            status.code=Code::value_mismatch; status.eigenvector=j; return finish(status);
        }
    }
    // Informational only: a small or zero cutoff does not reject an
    // accurately targeted normwise eigenpair or claim a forward-angle bound.
    if (k<n) {
        const R low=std::scalbn(std::scalbn(w[k-1],status.scale_exponent),-status.scale_exponent);
        const R high=std::scalbn(std::scalbn(w[k],status.scale_exponent),-status.scale_exponent);
        if (!std::isfinite(high)) { status.code=Code::unrepresentable_values; return finish(status); }
        status.cutoff_gap=double(high)-double(low);
        status.selection_budget=tolerance<R>(p)*(double(tn)+std::max(std::abs(double(low)),std::abs(double(high))));
    }
    long double accepted_residual_squared=0;
    for (int q=0;q<p.ranks;++q) { counts[q]=::ddla::detail::RowShare(n,p.ranks).count(q); disps[q]=::ddla::detail::RowShare(n,p.ranks).lo(q); }
    R* a=phase; R* b=a+n; R* c=b+n; R* upper2=c+n; R* x=upper2+n;
    for (int j=0;j<k;++j) {
        const int at=order[j],block=blocks[at],begin=block>1?splits[block-2]:0,end=splits[block-1],size=end-begin;
        const int owner=j%p.ranks;
        R shift=beta[at];
        for (int previous=j-1;previous>=0;--previous) if (blocks[order[previous]]==block) {
            const R separation=R(10)*std::numeric_limits<R>::epsilon()*std::abs(shift);
            shift=std::max(shift,beta[order[previous]]+separation);
            break;
        }
        beta[at]=shift; // Private shifts only; public STERF values never change.
        bool accepted=false;
        status.last_residual=status.last_orthogonality=std::numeric_limits<double>::infinity();
        for (int restart=0;restart<p.max_restarts && !accepted;++restart) {
            R solve_tolerance=0;
            int growth_count=0;
            status.last_seed_solves=status.last_seed_growth=status.first_five_solves=0;
            status.first_five_growth_mask=0;
            status.candidate_available=false;
            status.last_normwise_accepted=false;
            const R block_norm=norm1(size,ds+begin,es+begin);
            const double growth_threshold=std::sqrt(0.1/double(size));
            info=0;
            if (p.rank==owner && size>1) {
                std::copy_n(ds+begin,size,a); std::copy_n(es+begin,size-1,b); std::copy_n(es+begin,size-1,c);
                lagtf(size,a,shift,b,c,upper2,piv,info);
            }
            MPI_Bcast(&info,1,MPI_INT,owner,p.comm);
            if (info) { status.code=Code::factor_failed; status.native_info=info; status.eigenvector=j; return finish(status); }
            if (p.rank==owner) {
                std::fill_n(x,n,R(0));
                for (int i=begin;i<end;++i) {
                    std::uint64_t z=(std::uint64_t(i+1)*0x9e3779b97f4a7c15ULL)^std::uint64_t(j+1)^((std::uint64_t(restart)+1)<<32);
                    z^=z>>30; z*=0xbf58476d1ce4e5b9ULL; z^=z>>27; z*=0x94d049bb133111ebULL; z^=z>>31;
                    x[i]=R(double(z>>11)/9007199254740992.0-0.5);
                }
                if (size==1) x[begin]=R(1);
            }
            const int first_iteration=size==1?0:1;
            for (int iteration=first_iteration;iteration<=p.max_iterations;++iteration) {
                status.candidate_available=false;
                info=0;
                int bad_rhs=0;
                if (size>1 && p.rank==owner) {
                    R rhs_max=0;
                    for (int i=begin;i<end;++i) rhs_max=std::max(rhs_max,std::abs(x[i]));
                    if (!(rhs_max>R(0)) || !std::isfinite(rhs_max)) bad_rhs=1;
                    else {
                        // STEIN prescribes the RHS magnitude before LAGTS.
                        // This scaling must accompany its raw growth test.
                        const long double desired=static_cast<long double>(size)*block_norm*
                            std::max(std::numeric_limits<R>::epsilon(),std::abs(a[size-1]));
                        const long double scale=desired/rhs_max;
                        R scaled_max=0;
                        for (int i=begin;i<end;++i) {
                            x[i]=R(static_cast<long double>(x[i])*scale);
                            if (!std::isfinite(x[i])) bad_rhs=1;
                            scaled_max=std::max(scaled_max,std::abs(x[i]));
                        }
                        if (!(scaled_max>R(0))) bad_rhs=1;
                    }
                }
                MPI_Bcast(&bad_rhs,1,MPI_INT,owner,p.comm);
                if (bad_rhs) break;
                if (size>1) {
                    if (p.rank==owner) lagts(size,a,b,c,upper2,piv,x+begin,solve_tolerance,info);
                    ++status.solves;
                    status.last_seed_solves=iteration;
                    status.first_five_solves=std::min(iteration,5);
                }
                MPI_Bcast(&info,1,MPI_INT,owner,p.comm);
                if (info) { status.code=Code::solve_failed; status.native_info=info; status.eigenvector=j; return finish(status); }
                double native_max=0;
                if (p.rank==owner) {
                    for (int i=begin;i<end;++i) {
                        if (!std::isfinite(x[i])) bad_rhs=1;
                        native_max=std::max(native_max,std::abs(double(x[i])));
                    }
                    if (!(native_max>0)) bad_rhs=1;
                    if (!bad_rhs) for (int i=0;i<n;++i) x[i]=R(x[i]/native_max);
                }
                MPI_Bcast(&bad_rhs,1,MPI_INT,owner,p.comm);
                MPI_Bcast(&native_max,1,MPI_DOUBLE,owner,p.comm);
                if (bad_rhs) break;
                MPI_Bcast(x,n,datatype<R>(),owner,p.comm);
                double incoming=0;
                if (p.rank==owner) incoming=norm(n,x);
                MPI_Bcast(&incoming,1,MPI_DOUBLE,owner,p.comm);
                if (!std::isfinite(incoming) || incoming==0) break;
                // Two modified Gram-Schmidt passes, only within this split
                // block. Independent block ties keep their disjoint supports.
                for (int pass=0;pass<2;++pass) for (int previous=0;previous<j;++previous) {
                    if (blocks[order[previous]]!=block) continue;
                    const R* column=p.rows?Y+std::size_t(previous)*p.ldy:nullptr;
                    R coefficient=dot(p.rows,x+p.row0,column);
                    MPI_Allreduce(MPI_IN_PLACE,&coefficient,1,datatype<R>(),MPI_SUM,p.comm);
                    ++status.dot_reductions;
                    axpy(p.rows,-coefficient,column,x+p.row0);
                }
                MPI_Allgatherv(MPI_IN_PLACE,0,datatype<R>(),x,counts,disps,datatype<R>(),p.comm);
                ++status.allgathers;
                double length=0,post_max=0;
                int grew=0;
                if (p.rank==owner) {
                    length=norm(n,x);
                    for (int i=begin;i<end;++i) post_max=std::max(post_max,std::abs(double(x[i])));
                    // Algebraically native_max*post_max is the unnormalized
                    // post-MGS infinity norm. Division avoids product overflow.
                    grew=size>1 && post_max>0 && native_max>=growth_threshold/post_max;
                }
                MPI_Bcast(&length,1,MPI_DOUBLE,owner,p.comm);
                MPI_Bcast(&post_max,1,MPI_DOUBLE,owner,p.comm);
                MPI_Bcast(&grew,1,MPI_INT,owner,p.comm);
                status.last_native_maximum=native_max;
                status.last_post_reorth_maximum=post_max;
                status.last_growth_magnitude=post_max==0?0:
                    (native_max>std::numeric_limits<double>::max()/post_max?
                        std::numeric_limits<double>::infinity():native_max*post_max);
                if (!std::isfinite(length) || length<=16*std::numeric_limits<R>::epsilon()*incoming) break;
                if (grew) {
                    ++growth_count; ++status.growth_checks;
                    if (iteration<=5) status.first_five_growth_mask|=unsigned(1)<<(iteration-1);
                }
                status.last_seed_growth=growth_count;
                for (int i=0;i<n;++i) x[i]=R(x[i]/length);
                // Certify against the safely transformed actual returned
                // value, including any subnormal output rounding.
                const R public_value=std::scalbn(w[j],status.scale_exponent);
                const R target=std::scalbn(public_value,-status.scale_exponent);
                const auto certificate=certify_scaled<R>(p,ds,es,target,x,Y,j);
                status.dot_reductions+=j;
                status.last_residual=certificate.residual;
                status.last_orthogonality=certificate.orthogonality;
                status.candidate_available=true;
                status.last_candidate_index=j;
                status.last_normwise_accepted=certificate.accepted;
                // STEIN's EXTRA=2 requires three successful growth checks;
                // a failed intervening check does not reset NRMCHK.
                if (certificate.accepted && (size==1 || growth_count>=3)) {
                    const long double absolute_residual=certificate.residual*
                        (static_cast<long double>(tn)+std::abs(static_cast<long double>(target)))*certificate.norm;
                    accepted_residual_squared+=absolute_residual*absolute_residual;
                    if (k<n && status.cutoff_gap>0)
                        status.subspace_residual_ratio=double(std::sqrt(accepted_residual_squared))/status.cutoff_gap;
                    status.maximum_residual=std::max(status.maximum_residual,certificate.residual);
                    status.maximum_orthogonality=std::max(status.maximum_orthogonality,certificate.orthogonality);
                    int pivot=0;
                    for (int i=1;i<n;++i) if (std::abs(x[i])>std::abs(x[pivot])) pivot=i;
                    const R sign=x[pivot]<R(0)?R(-1):R(1);
                    for (int i=0;i<p.rows;++i) Y[i+std::size_t(j)*p.ldy]=sign*x[p.row0+i];
                    accepted=true;
                    break;
                }
            }
        }
        if (!accepted) {
            status.code=Code::no_convergence;
            status.eigenvector=j; return finish(status);
        }
    }
    return finish(status);
}
#define INSTANTIATE(R) \
 template Plan geometry<R>(MPI_Comm,int,int,int,int,int,int,int); \
 template Plan check<R>(MPI_Comm,int,int,const R*,const R*,R*,R*,int,int,int); \
 template Status run<R>(const Plan&,const R*,const R*,R*,R*,void*); \
 template Certificate certify_scaled<R>(const Plan&,const R*,const R*,R,const R*,const R*,int)
INSTANTIATE(float);
INSTANTIATE(double);
#undef INSTANTIATE
} } } // namespace ddla::spectral_impl::selected
