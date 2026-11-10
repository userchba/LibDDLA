#ifndef DDLA_HB2ST_HOUSEHOLDER_CPU_ROWS_H
#define DDLA_HB2ST_HOUSEHOLDER_CPU_ROWS_H

#include <ddla/ddla.h>
#include <cstddef>
#include <vector>
#include "eig_cpu_column_layout.h"
#include "eig_2d_input.h"

namespace ddla { namespace detail {
template<class T> class Hb2stHouseholderCpu;

// phase_count: 0 (no diagonal D) or the n - 1 phases D(1..n-1), D(0) = 1.
struct HhRowsCpuPlan {Eig2dInputPlan input_2d{};CpuReplayPlan map{};int ldy=1,threads=1,tile_width=0;std::size_t phase_count=0;};
template<class T> HhRowsCpuPlan hh_rows_cpu_geometry(const DdlaHandle_t&,int n,int ldy,const int* desc,std::size_t phase_count,
    bool paired,int threads);
template<class T> HhRowsCpuPlan hh_rows_cpu_check(const DdlaHandle_t&,int,const typename real_type<T>::type*,int,
    const Hb2stHouseholderCpu<T>&,const T* phases,std::size_t phase_count,T*,const int*);
// Backend bytes (CPU handle), in carving order: the layout; on a 64-byte
// boundary the column strips (n x ncs, or split-real tiles for a paired
// state); then one region holding in turn the input send/recv panels (P > 1),
// the scratch of apply_columns_run (unpaired) and the output send/recv panels
// (more than one process row).
template<class T> void hh_rows_cpu_workspace_size(const DdlaHandle_t&,const HhRowsCpuPlan&,std::size_t* backend_bytes);
// Non-consuming execution: Y and its padding remain caller-owned and unchanged.
template<class T> void hh_rows_cpu_run(const DdlaHandle_t&,const HhRowsCpuPlan&,const typename real_type<T>::type* Y,
    Hb2stHouseholderCpu<T>&,const T* phases,T* Z,int& info,void* backend_work,std::size_t backend_bytes);

// Private consuming CPU back-transform. Converts the row-slab real Y into
// distributed full-height column strips, applying a supplied diagonal D;
// consumes/frees Y only after every input exchange completes. The Householder
// state then applies Q_band collectively. Output redistribution uses bounded
// panels and preserves parent-descriptor entries and leading-dimension padding.
// On failure before consumption Y remains caller-owned; afterwards it is null.
template<class T>
void hh_rows_cpu(const DdlaHandle_t& handle, int n,
                 typename real_type<T>::type*& Yrows, int ldy,
                 Hb2stHouseholderCpu<T>& state, const std::vector<T>& phases,
                 T* Z, const int* descZ, int& info);
} }
#endif
