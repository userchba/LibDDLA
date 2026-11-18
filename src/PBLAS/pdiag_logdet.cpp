#include "diag_logdet_impl.h"
#include "ddla_desc.h"
#include "backend_api.h"
#include "check_agree.h"
#include "ddla_connector.h"
#include "instantiate.h"
#include "require_backend.h"
#include <stdexcept>

namespace ddla {
template<DdlaBackend B, class T>
void pdiag_logdet(const DdlaHandle_t& h, int n, const T* a, const int* desc,
                  std::complex<typename real_type<T>::type>& result)
{
    detail::DiagLogLayout layout{};
    detail::agree_check(h, "pdiag_logdet", [&](detail::Agree&) {
        detail::require_backend<B>(h, "pdiag_logdet");
        check_desc(desc, h);
        if (n < 0 || n > desc[DDLA_M_] || n > desc[DDLA_N_])
            throw std::invalid_argument("pdiag_logdet: order exceeds descriptor");
        int pr, pc, row, col;
        ddlaGetGridDims(h, pr, pc);
        ddlaGetGridCoords(h, row, col);
        const int rows = num_loc(n, desc[DDLA_MB_], row, desc[DDLA_RSRC_], pr);
        const int columns = num_loc(n, desc[DDLA_NB_], col, desc[DDLA_CSRC_], pc);
        if (rows > 0 && columns > 0 && !a)
            throw std::invalid_argument("pdiag_logdet: null local matrix");
        detail::check_diag_span<T>(rows, columns, desc[DDLA_LLD_]);
        const int count = rows == 0 || columns == 0 ? 0 : pr >= pc ? rows : columns;
        layout = {n, count, desc[DDLA_LLD_], desc[DDLA_MB_], desc[DDLA_NB_],
                  pr, pc, row, col, desc[DDLA_RSRC_], desc[DDLA_CSRC_], pr >= pc};
    });
    int signature[8] = {n};
    detail::descriptor_signature(signature + 1, desc);
    detail::agree_signature(h, "pdiag_logdet", signature);
    detail::DiagLogState state{};
    detail::agree_local(h, "pdiag_logdet reduction", [&] {
        state = detail::diagonal_logsum<B>(h, a, layout);
        if (state.flags & 1) throw std::domain_error("pdiag_logdet: nonfinite diagonal");
    });
    // Only four host scalars move, independent of matrix transport policy.
    const double local[] = {state.magnitude, state.phase,
                            double((state.flags & 2) != 0), double((state.flags & 4) != 0)};
    double total[4];
    MPI_CHECK(MPI_Allreduce(local, total, 4, MPI_DOUBLE, MPI_SUM, ddlaGetCommunicator(h)));
    result = detail::finish_diagonal_log<typename real_type<T>::type>(
        total[0], total[1], total[2] != 0, int(total[3]) & 1);
}
#define INST(B,T) template void pdiag_logdet<B,T>(const DdlaHandle_t&,int,const T*,const int*,std::complex<typename real_type<T>::type>&);
#define TYPES(B) INST(B,float) INST(B,double) INST(B,std::complex<float>) INST(B,std::complex<double>)
DDLA_FOR_EACH_BACKEND(TYPES)
#undef TYPES
#undef INST
} // namespace ddla
