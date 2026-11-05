#ifndef DDLA_PGEMMT_PANEL_IMPL_H
#define DDLA_PGEMMT_PANEL_IMPL_H
#include <ddla/ddla.h>
#include <ddla/ddla_desc.h>
#include "BLAS/blas_impl.h"
#include "BLAS/gemmt_impl.h"
#include "ddla_stream.h"
#include "pblas_workspace.h"
#include "plan_util.h"
#include <algorithm>
#include <cassert>
#include <limits>
namespace ddla { namespace pgemmt_impl { namespace panel {
// Enumerate diagonal blocks once for sizing, pointer construction and copying.
template <typename Visit>
void diagonal_tiles(const DdlaHandle_t& h,int n,const int* dc,const Visit& visit)
{
    const int nb=dc[DDLA_MB_];
    for (int global=0;global<n;) {
        const int width=std::min(nb,n-global);
        if (h->myprow_==indxg2p(global,nb,dc[DDLA_RSRC_],h->nprows_) &&
            h->mypcol_==indxg2p(global,nb,dc[DDLA_CSRC_],h->npcols_)) {
            const int row=num_loc(global,nb,h->myprow_,dc[DDLA_RSRC_],h->nprows_);
            const int column=num_loc(global,nb,h->mypcol_,dc[DDLA_CSRC_],h->npcols_);
            visit(row,column,width);
        }
        global+=width;
    }
}

// Selected rectangles form a staircase in local storage. Coalesce its bulk
// into large tiles and retain descriptor-sized tiles along the boundary.
// Groups 0..3 are coarse shapes, 4..7 the boundary, 8..16 the inner
// diagonal-block rectangles and 17..19 their full/partial diagonal leaves.
template <typename Visit>
void output_tiles(const DdlaHandle_t& h,char uplo,int n,
    const int* dc,const Visit& visit)
{
    const int nb=dc[DDLA_MB_],r=h->myprow_,c=h->mypcol_;
    const int pr=h->nprows_,pc=h->npcols_;
    const int rs=dc[DDLA_RSRC_],cs=dc[DDLA_CSRC_];
    const int mr=num_loc(n,nb,r,rs,pr),nc=num_loc(n,nb,c,cs,pc);
    const int tile=h->backend==DdlaBackend::CPU ? 256 : 512;
    const int block=std::max(nb,(tile/nb)*nb);
    const bool upper_triangle=uplo=='U';
    for (int column=0;column<nc;) {
        const int columns=std::min(block,nc-column);
        const int first_global=indxl2g(column,nb,c,cs,pc);
        const int last_global=indxl2g(column+columns-1,nb,c,cs,pc);
        int boundary;
        if (upper_triangle) {
            const int below=num_loc(first_global,nb,r,rs,pr);
            boundary=below==mr ? mr : below/block*block;
            for (int row=0;row<boundary;) {
                const int rows=std::min(block,boundary-row);
                visit(int(rows<block)*2+int(columns<block),row,column,rows,columns);
                row+=rows;
            }
        } else {
            const int after=num_loc(last_global+1,nb,r,rs,pr);
            boundary=int(std::min<std::size_t>(mr,(std::size_t(after)+block-1)/block*block));
            for (int row=boundary;row<mr;) {
                const int rows=std::min(block,mr-row);
                visit(int(rows<block)*2+int(columns<block),row,column,rows,columns);
                row+=rows;
            }
        }
        for (int j=column;j<column+columns;) {
            const int gj=indxl2g(j,nb,c,cs,pc),nj=std::min(nb,n-gj);
            const int first=upper_triangle ? boundary : num_loc(gj+nj,nb,r,rs,pr);
            const int last=upper_triangle ? num_loc(gj,nb,r,rs,pr) : boundary;
            for (int i=first;i<last;) {
                const int mi=std::min(nb,last-i);
                visit(4+int(mi<nb)*2+int(nj<nb),i,j,mi,nj);
                i+=mi;
            }
            j+=nj;
        }
        column+=columns;
    }
    const int leaf=std::min(nb,128);
    diagonal_tiles(h,n,dc,[&](int row,int column,int width) {
        for (int j=0;j<width;) {
            const int columns=std::min(leaf,width-j);
            const int column_shape=columns==leaf ? 0 : columns==nb%leaf ? 1 : 2;
            for (int i=0;i<width;) {
                const int rows=std::min(leaf,width-i);
                const int row_shape=rows==leaf ? 0 : rows==nb%leaf ? 1 : 2;
                if (i==j) visit(17+row_shape,row+i,column+j,rows,columns);
                else if (upper_triangle ? i<j : i>j)
                    visit(8+3*row_shape+column_shape,row+i,column+j,rows,columns);
                i+=rows;
            }
            j+=columns;
        }
    });
}

// Count output tiles and pointer entries for full and tail contraction panels.
inline void batch_counts(const DdlaHandle_t& h, char uplo, int n, int k, int full, const int* dc,
                 std::size_t& tiles, std::size_t& pointers, std::size_t& diagonals)
{
    tiles=diagonals=0;
    output_tiles(h,uplo,n,dc,[&](int group,int,int,int,int) {
        ++tiles;
        if (group>=17) ++diagonals;
    });
    const int tail=k%full;
    const int pitches=tail>0 && tail!=full ? 2 : 1;
    pointers=tiles*pitches;
}

// Backend lists contain one pointer plane per full/tail pitch.
// Host storage contains positions and those lists, plus diagonal destinations
// on GPU handles. Matrix temporaries remain in scratch()'s reusable tail.
template <typename T>
void batch_bytes(const DdlaHandle_t& h, char uplo, int n, int k, int full, const int* dc,
                std::size_t& backend_bytes, std::size_t& host_bytes)
{
    std::size_t tiles=0,pointers=0,diagonals=0;
    batch_counts(h,uplo,n,k,full,dc,tiles,pointers,diagonals);
    backend_bytes=0;
    if (h->backend==DdlaBackend::GPU)
        backend_bytes+=2*detail::padded_bytes<const T*>(pointers)+detail::padded_bytes<T*>(pointers);
    host_bytes=detail::padded_bytes<int>(3*tiles)
        +2*detail::padded_bytes<const T*>(pointers)+detail::padded_bytes<T*>(pointers);
    if (h->backend==DdlaBackend::GPU)
        host_bytes+=detail::padded_bytes<T*>(diagonals)+detail::padded_bytes<int>(diagonals);
}

template <typename T>
void workspace_size(const DdlaHandle_t& h,char uplo,int n,int k,int full,const int* dc,
    std::size_t* batch_bytes_out,std::size_t* host_bytes,std::size_t* triangle_bytes)
{
    *batch_bytes_out=*host_bytes=*triangle_bytes=0;
    if (n==0 || k==0) return;
    batch_bytes<T>(h,uplo,n,k,full,dc,*batch_bytes_out,*host_bytes);
    std::size_t diagonals=0;
    const std::size_t leaf=std::min(dc[DDLA_MB_],128);
    diagonal_tiles(h,n,dc,[&](int,int,int width) {diagonals+=(width+leaf-1)/leaf;});
    *triangle_bytes=detail::padded_bytes<T>(diagonals*leaf*leaf)
        +detail::padded_bytes<T*>(diagonals)+detail::padded_bytes<int>(diagonals);
}

// Consume prepared panels with the caller's CPU schedule or GPU batching.
// Batched products share the handle's thread budget on CPU; all communicated
// panels keep stable addresses while each contraction is consumed.
template <DdlaBackend Backend, typename T, typename Load>
void run(const DdlaHandle_t& h, char uplo, int n, int k, int full,
    const T& alpha, const T& beta, T* C, const int* dc,
    const Load& load, T* a, T* b,
    char* w, [[maybe_unused]] std::size_t backend_bytes, char* hw, [[maybe_unused]] std::size_t host_bytes,void* triangle_work,std::size_t triangle_bytes,bool cpu_batch=true)
{
    if constexpr (Backend==DdlaBackend::CPU) {
        if (!cpu_batch) {
            for (int g=0;g<k;) {
                const int kw=std::min(full,k-g);
                load(g,kw);
                pblas::triangle_panel_run<Backend,T>(h,uplo,true,n,kw,alpha,a,b,
                    g==0 ? beta : T(1),C,dc,triangle_work,triangle_bytes);
                g+=kw;
            }
            return;
        }
    }
    const int nb=dc[DDLA_MB_],r=h->myprow_,c=h->mypcol_;
    const int mr=num_loc(n,nb,r,dc[DDLA_RSRC_],h->nprows_);
    const int ldc=dc[DDLA_LLD_];
    const int tail=k%full;
    const int pitches=tail>0 && tail!=full ? 2 : 1;

    // The lists, in the order batch_bytes counts them.
    std::size_t count_tiles=0,pointers=0,diagonals=0;
    batch_counts(h,uplo,n,k,full,dc,count_tiles,pointers,diagonals);
    char* const w0=w;
    char* const hw0=hw;
    const T** device_a=nullptr;
    const T** device_b=nullptr;
    T** device_c=nullptr;
    if constexpr (Backend==DdlaBackend::GPU) {
        device_a=detail::take<const T*>(w,pointers);
        device_b=detail::take<const T*>(w,pointers);
        device_c=detail::take<T*>(w,pointers);
    }
    int* const positions=count_tiles ? detail::take<int>(hw,3*count_tiles) : nullptr;
    const T** const host_a=pointers ? detail::take<const T*>(hw,pointers) : nullptr;
    const T** const host_b=pointers ? detail::take<const T*>(hw,pointers) : nullptr;
    T** const host_c=pointers ? detail::take<T*>(hw,pointers) : nullptr;
    T* diagonal_work=nullptr;
    T** diagonal_targets=nullptr;
    T** host_targets=nullptr;
    int* diagonal_widths=nullptr;
    int* host_widths=nullptr;
    const int leaf=std::min(nb,128);
    if (diagonals>0) {
        char* cursor=static_cast<char*>(triangle_work);
        diagonal_work=detail::take<T>(cursor,diagonals*std::size_t(leaf)*leaf);
        diagonal_targets=detail::take<T*>(cursor,diagonals);
        diagonal_widths=detail::take<int>(cursor,diagonals);
        if constexpr (Backend==DdlaBackend::GPU) {
            host_targets=detail::take<T*>(hw,diagonals);
            host_widths=detail::take<int>(hw,diagonals);
        } else {
            host_targets=diagonal_targets;
            host_widths=diagonal_widths;
        }
        assert(std::size_t(cursor-static_cast<char*>(triangle_work))<=triangle_bytes);
    }
    assert(std::size_t(w-w0)<=backend_bytes && (host_bytes==0 || std::size_t(hw-hw0)<=host_bytes));

    std::size_t counts[20]={},first[21]={},next[20]={};
    int rows[20]={},columns[20]={};
    output_tiles(h,uplo,n,dc,[&](int group,int,int,int height,int width) {
        ++counts[group];rows[group]=height;columns[group]=width;
    });
    for (int group=0;group<20;++group) {
        first[group+1]=first[group]+counts[group];next[group]=first[group];
    }
    assert(first[20]==count_tiles);
    const std::size_t diagonal_first=first[17];
    output_tiles(h,uplo,n,dc,[&](int group,int row,int column,int width,int) {
        const auto index=next[group]++;
        positions[3*index]=row;positions[3*index+1]=column;positions[3*index+2]=width;
    });
    for (std::size_t index=diagonal_first;index<count_tiles;++index) {
        host_targets[index-diagonal_first]=C+positions[3*index]+std::size_t(positions[3*index+1])*ldc;
        host_widths[index-diagonal_first]=positions[3*index+2];
    }

    // Panel addresses stay fixed across k panels. C and every pointer table
    // are rebuilt on each run, permitting reuse with another output buffer.
    for (int pitch=0;pitch<pitches;++pitch) {
        const int kw=pitch==0 ? full : tail;
        const T* pa=a;
        const T* pb=b;
        const std::size_t base=pitch*count_tiles;
        for (std::size_t index=0;index<count_tiles;++index) {
            const int row=positions[3*index],column=positions[3*index+1];
            host_a[base+index]=pa+row;
            host_b[base+index]=pb+std::size_t(column)*kw;
            host_c[base+index]=index<diagonal_first ? C+row+std::size_t(column)*ldc
                : diagonal_work+(index-diagonal_first)*std::size_t(leaf)*leaf;
        }
    }
    const T* const* ap=host_a;
    const T* const* bp=host_b;
    T* const* cp=host_c;
    if constexpr (Backend==DdlaBackend::GPU) {
        if (pointers>0) {
            DDLA_CHECK(ddlaMemcpy(h,device_a,host_a,pointers*sizeof(const T*)));
            DDLA_CHECK(ddlaMemcpy(h,device_b,host_b,pointers*sizeof(const T*)));
            DDLA_CHECK(ddlaMemcpy(h,device_c,host_c,pointers*sizeof(T*)));
            if (diagonals>0) {
                DDLA_CHECK(ddlaMemcpy(h,diagonal_targets,host_targets,diagonals*sizeof(T*)));
                DDLA_CHECK(ddlaMemcpy(h,diagonal_widths,host_widths,diagonals*sizeof(int)));
            }
            // The pageable host pointer lists may be released/reused after
            // run returns; complete their asynchronous uploads once here.
            DDLA_CHECK(ddlaSynchronize(h));
        }
        ap=device_a; bp=device_b; cp=device_c;
    }
    for (int g=0;g<k;) {
        const int kw=std::min(full,k-g);
        load(g,kw);
        const T scale=g==0 ? beta : T(1);
        const int pitch=kw==full ? 0 : 1;
        const std::size_t base=pitch*count_tiles;
        for (int group=0;group<20;++group) {
            const bool diagonal=group>=17;
            const T output_beta=diagonal ? T(0) : scale;
            for (std::size_t done=0;done<counts[group];) {
                const int batch=static_cast<int>(std::min<std::size_t>(counts[group]-done,std::numeric_limits<int>::max()));
                const std::size_t start=base+first[group]+done;
                blas_impl::gemmBatched_run<Backend,T>(h,'N','N',rows[group],columns[group],kw,alpha,
                    ap+start,std::max(1,mr),bp+start,kw,output_beta,cp+start,diagonal ? leaf : ldc,batch);
                done+=batch;
            }
        }
        blas_impl::gemmt_copy_run<Backend,T>(h,uplo,leaf,int(diagonals),0,scale,
            diagonal_work,static_cast<T*>(nullptr),ldc,diagonal_targets,diagonal_widths);
        g+=kw;
    }
}


}}}
#endif
