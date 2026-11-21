#include "pgemm_geometry.h"
#include "check_agree.h"
#include "ddla_desc.h"
#include "require_backend.h"
#include "instantiate.h"

#include <cstdlib>
#include <stdexcept>

namespace ddla { namespace pgemm_impl {
namespace {

int kpanel_width(int default_width)
{
    int panel_width = default_width;
    const char* s = std::getenv("DDLA_PGEMM_KPANEL");
    if (s != nullptr && *s != '\0') {
        const int v = std::atoi(s);
        if (v > 0) panel_width = v;
    }
    return panel_width;
}

} // namespace

// Component 1.
template <DdlaBackend Backend>
Algorithm pgemm_check(
    const DdlaHandle_t& h, char transa, char transb, int m, int n, int k,
    bool alpha_is_zero, const int* descA, const int* descB, const int* descC,
    Algorithm algorithm, int& panel_width)
{
    return detail::agree_check(h, "pgemm", [&](detail::Agree& agree) {
        detail::require_backend<Backend>(h, "pgemm");
        // Each descriptor is validated against this handle's process grid
        // (dense block-cyclic type, slot ranges, LLD vs LOCr); a descriptor
        // built against a different process grid is rejected there.
        check_desc(descA, h);
        check_desc(descB, h);
        check_desc(descC, h);
        int nprows = 0, npcols = 0;
        ddlaGetGridDims(h, nprows, npcols);


        // Shared validation for every algorithm.
        if (transa != 'N' && transa != 'T' && transa != 'C')
            throw std::invalid_argument("pgemm: transa must be 'N', 'T' or 'C'");
        if (transb != 'N' && transb != 'T' && transb != 'C')
            throw std::invalid_argument("pgemm: transb must be 'N', 'T' or 'C'");
        if (m < 0 || n < 0 || k < 0)
            throw std::invalid_argument("pgemm: negative dimension");

        const int opA_m = (transa == 'N') ? descA[DDLA_M_] : descA[DDLA_N_];
        const int opA_n = (transa == 'N') ? descA[DDLA_N_] : descA[DDLA_M_];
        const int opB_m = (transb == 'N') ? descB[DDLA_M_] : descB[DDLA_N_];
        const int opB_n = (transb == 'N') ? descB[DDLA_N_] : descB[DDLA_M_];
        if (m > opA_m || k > opA_n || k > opB_m || n > opB_n ||
            m > descC[DDLA_M_] || n > descC[DDLA_N_])
            throw std::invalid_argument("pgemm: m x n x k exceeds the descriptors");
        {
            const int mbC = descC[DDLA_MB_];
            const int nbC = descC[DDLA_NB_];
            const int mbA = (transa == 'N') ? descA[DDLA_MB_] : descA[DDLA_NB_];
            const int kbA = (transa == 'N') ? descA[DDLA_NB_] : descA[DDLA_MB_];
            const int kbB = (transb == 'N') ? descB[DDLA_MB_] : descB[DDLA_NB_];
            const int nbB = (transb == 'N') ? descB[DDLA_NB_] : descB[DDLA_MB_];
            if (mbA != mbC || kbA != kbB || nbB != nbC)
                throw std::invalid_argument("pgemm: block sizes of A, B and C do not match");
        }

        panel_width = kDefaultPanelWidth;
        if (m == 0 || n == 0 || k == 0 || alpha_is_zero) return Algorithm::AB;
        if constexpr (Backend == DdlaBackend::GPU) {
            if (h->communication() == DdlaCommunication::CCL && h->communication_group_depth)
                throw std::invalid_argument("pgemm: active CCL group must be closed before a GPU product");
        }

        Algorithm chosen = algorithm;
        if (chosen == Algorithm::Auto) {
            chosen = select(transa, transb, m, n, k, descA, descB, descC, nprows, npcols);
            // The cost model ignores source coordinates; SUMMA (AB) accepts any
            // alignment, so an unaligned stationary pick falls back to it.
            if (chosen != Algorithm::AB &&
                !stationary_aligned(chosen == Algorithm::BC, transa, transb, descA, descB, descC))
                chosen = Algorithm::AB;
        } else if (chosen != Algorithm::AB &&
                   !stationary_aligned(chosen == Algorithm::BC, transa, transb, descA, descB, descC)) {
            throw std::invalid_argument("pgemm: stationary branch requires aligned non-transposed panel owners");
        }
        panel_width = kpanel_width(transa == 'N' && transb == 'N'
            ? kDefaultPanelWidth : ab_block(transa, descA));
        agree.mode("DDLA_PGEMM_KPANEL", panel_width);
        return chosen;
    });
}
bool stationary_aligned(bool keep_a, char ta, char tb, const int* da, const int* db, const int* dc)
{
    const int* fixed = keep_a ? da : db;
    const int* thin = keep_a ? db : da;
    const bool k_rows = keep_a ? ta != 'N' : tb == 'N';
    const bool thin_k_rows = keep_a ? tb == 'N' : ta != 'N';
    const int kb = fixed[k_rows ? DDLA_MB_ : DDLA_NB_];
    const int ks = fixed[k_rows ? DDLA_RSRC_ : DDLA_CSRC_];
    const int fb = fixed[k_rows ? DDLA_NB_ : DDLA_MB_];
    const int fs = fixed[k_rows ? DDLA_CSRC_ : DDLA_RSRC_];
    const bool transpose_output = keep_a == k_rows;
    return !((thin_k_rows == k_rows && thin[k_rows ? DDLA_RSRC_ : DDLA_CSRC_] != ks) ||
             (!transpose_output && fs != dc[keep_a ? DDLA_RSRC_ : DDLA_CSRC_]));
}
#define INSTANTIATE_PGEMM_CHECK(BACKEND) \
    template pgemm_impl::Algorithm pgemm_impl::pgemm_check<BACKEND>(const DdlaHandle_t&, \
        char, char, int, int, int, bool, const int*, const int*, const int*, pgemm_impl::Algorithm, int&)

DDLA_FOR_EACH_BACKEND(INSTANTIATE_PGEMM_CHECK)
#undef INSTANTIATE_PGEMM_CHECK

}} // namespace ddla::pgemm_impl
