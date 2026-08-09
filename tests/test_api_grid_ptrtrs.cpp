#include "api_grid_test_common.h"

using namespace api_grid_test;

namespace {

// Stored triangle of A: lower generator for uplo='L', its transpose for 'U'.
Complex stored_tri(char uplo, int i, int j)
{
    return (uplo == 'L') ? triangular_l_value(i, j) : triangular_l_value(j, i);
}

// op(A)(i,j) with trans in {'N','T','C'} applied to the stored triangle.
// For diag='U' the diagonal of op(A) is the unit value (1,0), matching what
// ptrtrs treats as the implicit unit diagonal.
Complex op_tri(char uplo, char trans, char diag, int i, int j)
{
    if(diag == 'U' && i == j) return Complex(1.0, 0.0);
    if(trans == 'N') return stored_tri(uplo, i, j);
    const Complex raw = stored_tri(uplo, j, i);
    return (trans == 'T') ? raw : std::conj(raw);
}

} // namespace

void check_ptrtrs(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int n = square_size(handle, base);
    const int nrhs = nrhs_size(base, 5);
    ddla::DdlaDesc descA(handle);
    descA.init(n, n, nb, nb, 0, 0);

    auto run_case = [&](char side, char uplo, char trans, char diag){
        const std::string name = std::string("ptrtrs(") + side + "," + uplo + "," + trans + "," + diag + ")";
        // side='L': B is n x nrhs, solve op(A)*X = B;
        // side='R': B is nrhs x n, solve X*op(A) = B.
        const int b_rows = (side == 'L') ? n : nrhs;
        const int b_cols = (side == 'L') ? nrhs : n;
        ddla::DdlaDesc descB(handle);
        descB.init(b_rows, b_cols, nb, nb, 0, 0);

        auto h_A = make_local<Complex>(descA, [&](int i, int j){ return stored_tri(uplo, i, j); });
        auto h_B = make_local<Complex>(descB, [&](int i, int j) -> Complex {
            Complex sum(0.0, 0.0);
            for(int l = 0; l < n; ++l){
                if(side == 'L'){
                    sum += op_tri(uplo, trans, diag, i, l) * x_value(l, j);
                }else{
                    sum += x_value(i, l) * op_tri(uplo, trans, diag, l, j);
                }
            }
            return sum;
        });

        DeviceBuffer<Complex> d_A(handle, h_A.size());
        DeviceBuffer<Complex> d_B(handle, h_B.size());
        upload(handle, d_A.ptr, h_A);
        upload(handle, d_B.ptr, h_B);
        check_ddla_sync(handle);

        ddla::ptrtrs(side, uplo, trans, diag, b_rows, b_cols, d_A.ptr, descA, d_B.ptr, descB);
        check_solution(handle, descB, d_B.ptr, h_B.size(), name, 2e-10);
    };

    // Full side x uplo x trans x diag enumeration: 2 x 2 x 3 x 2 = 24 cases.
    const char sides[]  = {'L', 'R'};
    const char uplos[]  = {'L', 'U'};
    const char transes[] = {'N', 'T', 'C'};
    const char diags[]  = {'N', 'U'};
    for(char side : sides)
        for(char uplo : uplos)
            for(char trans : transes)
                for(char diag : diags)
                    run_case(side, uplo, trans, diag);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_ptrtrs", check_ptrtrs);
}
