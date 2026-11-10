#ifndef DDLA_HB2ST_FINISH_PANEL_H
#define DDLA_HB2ST_FINISH_PANEL_H

#include <ddla/ddla.h>
#include <functional>
#include "plan_util.h"
#include "row_share.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ddla {
namespace detail {

// Internal entry point; it also permits small forced panels in focused tests.
// Y remains immutable; output must not overlap it.  Host engine only: GPU
// handles finish through hb2st_replay_gpu.
template <typename T>
void hb2st_finish_rows_host(const DdlaHandle_t& h, int n,
                            const typename real_type<T>::type* Y, int ldy,
                            const std::vector<T>& phases, T* Z, const int* descZ,
                            const std::function<void(T*, int, int)>& pack_columns,
                            int threads, int panel_columns = 0);

// Private finish layout. Row ownership is constant across output panels;
// column slots and collective counts are rebuilt for one global interval.
// The lists live in caller scratch (hb2st_finish_panel_map); nli is the
// length of li, nlc that of lj for the current interval.
struct Hb2stFinishPanel {
    int n=0,npr=1,npc=1,P=1,mypr=0,mypc=0,row_lo=0,nrl=0,mb=1,nb=1,rsrc=0,csrc=0;
    bool rowmajor=false;
    int j0 = 0, width = 0, nlc = 0, nli = 0;
    int *nr=nullptr,*rslot=nullptr,*rc_rows=nullptr,*rro=nullptr,*li=nullptr,*cs=nullptr,*lj=nullptr,*nc=nullptr;
    int *sc=nullptr,*sd=nullptr,*rc=nullptr,*rd=nullptr;
    int row_owner(int i) const { return (i / mb + rsrc) % npr; }
    int col_owner(int j) const { return (j / nb + csrc) % npc; }
    int rank_of(int pr, int pc) const { return rowmajor ? pr * npc + pc : pc * npr + pr; }
    void columns(int begin, int count)
    {
        j0 = begin; width = count; nlc = 0;
        std::fill(nc, nc + npc, 0);
        for (int j = 0; j < width; ++j) {
            const int global = j0 + j, pc = col_owner(global);
            cs[j] = nc[pc]++;
            if (pc == mypc) lj[nlc++] = (global / nb / npc) * nb + global % nb;
        }
        long long send_total = 0, recv_total = 0;
        for (int q = 0; q < P; ++q) {
            const int pr = rowmajor ? q / npc : q % npr;
            const int pc = rowmajor ? q % npc : q / npr;
            const long long send_count = static_cast<long long>(nr[pr]) * nc[pc];
            const long long recv_count = static_cast<long long>(rc_rows[q]) * nlc;
            if (send_total + send_count > std::numeric_limits<int>::max() ||
                recv_total + recv_count > std::numeric_limits<int>::max())
                throw std::overflow_error("Hb2stFinishPanel: MPI panel count exceeds int");
            sd[q] = static_cast<int>(send_total); rd[q] = static_cast<int>(recv_total);
            sc[q] = static_cast<int>(send_count); rc[q] = static_cast<int>(recv_count);
            send_total += send_count; recv_total += recv_count;
        }
    }
    std::size_t send_size() const { return static_cast<std::size_t>(sd[P-1])+sc[P-1]; }
    std::size_t recv_size() const { return static_cast<std::size_t>(rd[P-1])+rc[P-1]; }
};

// Maximum columns of a contiguous interval owned by any process column,
// including a panel beginning inside a block. Also gives maximum local rows
// when called with (n, mb, npr). Source coordinates merely rotate owners.
inline std::size_t hb2st_finish_max_owned(int width, int block, int processes)
{
    const std::size_t period = static_cast<std::size_t>(block) * processes;
    return (static_cast<std::size_t>(width) / period) * block +
           std::min(static_cast<std::size_t>(width) % period, static_cast<std::size_t>(block));
}

// Bytes hb2st_finish_panel_map takes: nr (npr), rslot (max(1, this rank's
// rows)), rc_rows and rro (P each), li (the Z rows of this process row), cs
// (panel_width), lj (the most panel columns a process column owns), nc (npc),
// then sc, sd, rc and rd (P each).
inline std::size_t hb2st_finish_panel_map_bytes(int n,int npr,int npc,int mypr,int rank,const int* d,int panel_width)
{
    const int P=npr*npc;
    return padded_bytes<int>(npr)+padded_bytes<int>(std::max(1,RowShare(n,P).count(rank)))+6*padded_bytes<int>(P)+
           padded_bytes<int>(num_loc(n,d[DDLA_MB_],mypr,d[DDLA_RSRC_],npr))+padded_bytes<int>(panel_width)+
           padded_bytes<int>(hb2st_finish_max_owned(panel_width,d[DDLA_NB_],npc))+padded_bytes<int>(npc);
}

// This rank's finish layout with its row ownership filled in, the lists
// carved from cursor in the order hb2st_finish_panel_map_bytes counts them.
inline Hb2stFinishPanel hb2st_finish_panel_map(int n,int npr,int npc,int mypr,int mypc,int rank,char major,
    const int* desc,int panel_width,char*& cursor)
{
    Hb2stFinishPanel m;
    m.n=n;m.npr=npr;m.npc=npc;m.P=npr*npc;m.mypr=mypr;m.mypc=mypc;
    const RowShare rows(n,m.P);m.row_lo=rows.lo(rank);m.nrl=rows.count(rank);
    m.mb=desc[DDLA_MB_];m.nb=desc[DDLA_NB_];m.rsrc=desc[DDLA_RSRC_];m.csrc=desc[DDLA_CSRC_];m.rowmajor=major=='R';
    const int nrows=num_loc(n,m.mb,mypr,m.rsrc,npr);
    m.nr=take<int>(cursor,npr);m.rslot=take<int>(cursor,std::max(1,m.nrl));
    m.rc_rows=take<int>(cursor,m.P);m.rro=take<int>(cursor,m.P);m.li=take<int>(cursor,nrows);
    m.cs=take<int>(cursor,panel_width);m.lj=take<int>(cursor,hb2st_finish_max_owned(panel_width,m.nb,npc));
    m.nc=take<int>(cursor,npc);
    m.sc=take<int>(cursor,m.P);m.sd=take<int>(cursor,m.P);m.rc=take<int>(cursor,m.P);m.rd=take<int>(cursor,m.P);
    std::fill(m.nr,m.nr+npr,0);std::fill(m.rslot,m.rslot+std::max(1,m.nrl),0);
    std::fill(m.rc_rows,m.rc_rows+m.P,0);std::fill(m.rro,m.rro+m.P,0);std::fill(m.nc,m.nc+npc,0);
    std::fill(m.sc,m.sc+m.P,0);std::fill(m.sd,m.sd+m.P,0);std::fill(m.rc,m.rc+m.P,0);std::fill(m.rd,m.rd+m.P,0);
    for (int r = 0; r < m.nrl; ++r) m.rslot[r] = m.nr[m.row_owner(m.row_lo + r)]++;
    for (int q = 0; q < m.P; ++q) {
        m.rro[q] = m.nli;
        for (int r = rows.lo(q); r < rows.hi(q); ++r) {
            if (m.row_owner(r) != mypr) continue;
            ++m.rc_rows[q];
            m.li[m.nli++] = (r / m.mb / npr) * m.mb + r % m.mb;
        }
    }
    return m;
}

// All ranks use shared dimensions only. The 256 MiB target covers product /
// receive reuse, send, broadcast and phase panels, plus the two host-staged
// communication buffers even in a direct GPU-collective build. Metadata,
// phases and the CPU's separately bounded packed-Q panel are O(n/P + P + n).
// One column is the minimum if an exceptionally large problem exceeds the
// target already at width one. No allocation grows as n*n/P here.
inline std::size_t hb2st_finish_panel_bytes(int n, int npr, int npc, int mb, int nb,
                                          int width, std::size_t scalar_bytes,
                                          std::size_t real_bytes, bool cpu, int kp)
{
    const int P = npr * npc;
    const std::size_t rows = RowShare(n, P).count(0);
    const std::size_t out_rows = hb2st_finish_max_owned(n, mb, npr);
    const std::size_t recv = out_rows * hb2st_finish_max_owned(width, nb, npc) * scalar_bytes;
    const std::size_t product = rows * width * scalar_bytes;
    const std::size_t krows = cpu ? static_cast<std::size_t>(kp) : rows;
    const std::size_t broadcast = P > 1 ? krows * width * real_bytes : 0;
    const std::size_t phase = scalar_bytes != real_bytes ? krows * width * scalar_bytes : 0;
    const std::size_t redist = P > 1 ? std::max(product, recv) + product : 0;
    const std::size_t staging = P > 1 ? std::max(broadcast, product) + recv : 0;
    return broadcast + phase + redist + staging;
}

inline int hb2st_finish_panel_width(int n, int npr, int npc, int mb, int nb,
                                    std::size_t scalar_bytes, std::size_t real_bytes, bool cpu, int kp)
{
    if (n <= 0) return 1;
    constexpr std::size_t target = std::size_t(256) << 20;
    int lo = 1, hi = std::min(n, 4096);
    while (lo < hi) {
        const int mid = lo + (hi - lo + 1) / 2;
        if (hb2st_finish_panel_bytes(n, npr, npc, mb, nb, mid, scalar_bytes, real_bytes, cpu, kp) <= target) lo = mid;
        else hi = mid - 1;
    }
    if (lo >= nb) lo = (lo / nb) * nb;
    return lo;
}

} // namespace detail
} // namespace ddla
#endif
