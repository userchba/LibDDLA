#include "api_grid_test_common.h"

using namespace api_grid_test;

namespace {

void check_non_square_block_transpose(
    const ddla::DdlaHandle_t& handle, const Shape& base
)
{
    const int mb = std::max(2, base.nb);
    const int nb = mb + 1;
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    int myprow = 0, mypcol = 0;
    ddlaGetGridCoords(handle, myprow, mypcol);
    const int m = round_up_for_grid(base.m, mb, nprows);
    const int n = round_up_for_grid(base.n, nb, npcols);
    const int irsrc = nprows - 1;
    constexpr int icsrc = 0;
    constexpr int tag = 17;

    int desc[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(desc, handle, m, n, mb, nb, irsrc, icsrc));

    auto h_A = make_local<Complex>(handle, desc, [](int i, int j){
        return general_value(i, j, tag);
    });
    DeviceBuffer<Complex> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);

    const int row_panel_rows = mb;
    const int target_row_columns = ddla::num_loc(
        n, nb, myprow, icsrc, nprows
    );
    for(char trans : {'T', 'C'}){
        const size_t output_count = static_cast<size_t>(row_panel_rows)
                                  * target_row_columns;
        const size_t buffer_count = static_cast<size_t>(row_panel_rows)
                                  * std::max(ddla_test::n_loc(handle, desc), target_row_columns);
        DeviceBuffer<Complex> d_row(handle, buffer_count);
        ddla::transport_block(handle, 
            'R', trans, row_panel_rows, n,
            d_A.ptr, 0, 0, desc, d_row.ptr
        );
        auto h_row = download(handle, d_row.ptr, output_count);

        double err = 0.0;
        for(int jloc = 0; jloc < target_row_columns; ++jloc){
            const int j = ddla::indxl2g(
                jloc, nb, myprow, icsrc, nprows
            );
            if(j >= n) continue;
            for(int i = 0; i < row_panel_rows; ++i){
                err = std::max(
                    err,
                    std::abs(h_row[i + jloc * row_panel_rows]
                           - general_value(i, j, tag))
                );
            }
        }
        require_close(
            handle,
            std::string("transport_block non-square blocks (R,") + trans + ')',
            err, 1e-12
        );
    }

    const int col_panel_cols = nb;
    const int target_col_rows = ddla::num_loc(
        m, mb, mypcol, irsrc, npcols
    );
    for(char trans : {'T', 'C'}){
        const size_t output_count = static_cast<size_t>(col_panel_cols)
                                  * target_col_rows;
        const size_t buffer_count = static_cast<size_t>(col_panel_cols)
                                  * std::max(ddla_test::m_loc(handle, desc), target_col_rows);
        DeviceBuffer<Complex> d_col(handle, buffer_count);
        ddla::transport_block(handle, 
            'C', trans, m, col_panel_cols,
            d_A.ptr, 0, 0, desc, d_col.ptr
        );
        auto h_col = download(handle, d_col.ptr, output_count);

        double err = 0.0;
        for(int iloc = 0; iloc < target_col_rows; ++iloc){
            const int i = ddla::indxl2g(
                iloc, mb, mypcol, irsrc, npcols
            );
            if(i >= m) continue;
            for(int j = 0; j < col_panel_cols; ++j){
                Complex expected = general_value(i, j, tag);
                if(trans == 'C') expected = std::conj(expected);
                err = std::max(
                    err,
                    std::abs(h_col[j + iloc * col_panel_cols] - expected)
                );
            }
        }
        require_close(
            handle,
            std::string("transport_block non-square blocks (C,") + trans + ')',
            err, 1e-12
        );
    }
}

void check_host_tunnel_workspace(const ddla::DdlaHandle_t& handle)
{
    constexpr int mb = 65;
    constexpr int nb = 2;
    constexpr int local_columns = 128;
    constexpr int tag = 29;
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    const int m = mb * nprows;
    const int n = local_columns * npcols;

    int desc[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(desc, handle, m, n, mb, nb, 0, 0));

    auto h_A = make_local<Complex>(handle, desc, [](int i, int j){
        return general_value(i, j, tag);
    });
    DeviceBuffer<Complex> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);

    const size_t count = static_cast<size_t>(mb) * ddla_test::n_loc(handle, desc);
    DeviceBuffer<Complex> d_row(handle, count);
    ddla::transport_block(handle, 'R', 'N', mb, n, d_A.ptr, 0, 0, desc, d_row.ptr);
    auto h_row = download(handle, d_row.ptr, count);

    double err = 0.0;
    for(int jloc = 0; jloc < ddla_test::n_loc(handle, desc); ++jloc){
        const int j = indx_l2g_c(desc, handle, jloc);
        for(int i = 0; i < mb; ++i){
            err = std::max(
                err,
                std::abs(h_row[i + jloc * mb] - general_value(i, j, tag))
            );
        }
    }
    require_close(handle, "transport_block host workspace (R,N)", err, 1e-12);
}

} // namespace

void check_transport_block(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    int nprows = 0, npcols = 0;
    ddlaGetGridDims(handle, nprows, npcols);
    int myprow = 0, mypcol = 0;
    ddlaGetGridCoords(handle, myprow, mypcol);
    const int m = round_up_for_grid(base.m, nb, nprows);
    const int n = round_up_for_grid(base.n, nb, npcols);
    int desc[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(desc, handle, m, n, nb, nb, 0, 0));

    auto h_A = make_local<Complex>(handle, desc, [](int i, int j){ return general_value(i, j, 7); });
    DeviceBuffer<Complex> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);

    const int rows = std::min(nb, m);
    DeviceBuffer<Complex> d_row(handle, static_cast<size_t>(rows) * ddla_test::n_loc(handle, desc));
    ddla::transport_block(handle, 'R', 'N', rows, n, d_A.ptr, 0, 0, desc, d_row.ptr);
    auto h_row = download(handle, d_row.ptr, static_cast<size_t>(rows) * ddla_test::n_loc(handle, desc));
    double err_row = 0.0;
    for(int jloc = 0; jloc < ddla_test::n_loc(handle, desc); ++jloc){
        const int j = indx_l2g_c(desc, handle, jloc);
        for(int r = 0; r < rows; ++r){
            err_row = std::max(err_row, std::abs(h_row[r + jloc * rows] - general_value(r, j, 7)));
        }
    }
    require_close(handle, "transport_block(handle, R,N)", err_row, 1e-12);

    const int cols = std::min(nb, n);
    DeviceBuffer<Complex> d_col(handle, static_cast<size_t>(ddla_test::m_loc(handle, desc)) * cols);
    ddla::transport_block(handle, 'C', 'N', m, cols, d_A.ptr, 0, 0, desc, d_col.ptr);
    auto h_col = download(handle, d_col.ptr, static_cast<size_t>(ddla_test::m_loc(handle, desc)) * cols);
    double err_col = 0.0;
    for(int c = 0; c < cols; ++c){
        for(int iloc = 0; iloc < ddla_test::m_loc(handle, desc); ++iloc){
            const int i = indx_l2g_r(desc, handle, iloc);
            err_col = std::max(err_col, std::abs(h_col[iloc + c * ddla_test::m_loc(handle, desc)] - general_value(i, c, 7)));
        }
    }
    require_close(handle, "transport_block(handle, C,N)", err_col, 1e-12);

    check_non_square_block_transpose(handle, base);
    check_host_tunnel_workspace(handle);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_transport_block", check_transport_block);
}
