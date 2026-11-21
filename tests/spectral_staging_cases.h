// Included once inside test_staging_preparation's private namespace. Numerical
// fixtures stay bounded/distributed; this header introduces no production API.
#if DDLA_HAS_GPU
namespace sp = ddla::spectral_impl;
namespace cb = ddla::spectral_impl::chebyshev;
constexpr auto GPU = DdlaBackend::GPU;

template <class T> T spectral_value(double r, double i = 0)
{
    if constexpr (std::is_arithmetic_v<T>)
        return T(r);
    else
        return T(r, i);
}
template <class T> T spectral_pattern(int i, int j)
{
    return spectral_value<T>((i == j ? 5. : 0) + double((i * 7 + j * 11) % 19 - 9) / 37.,
                             double((i * 13 + j * 3) % 17 - 8) / 43.);
}
template <class Query>
std::array<std::size_t, 2> spectral_query(DdlaHandle_t h, const Trace &trace, const char *label,
                                          Query &&query)
{
    // Resolve the workspace interposer before enabling the allocation observer.
    require_all(workspace_test::available(), "spectral suite requires workspace_trace preload");
    workspace_test::Scope scope(true);
    trace.begin(nullptr, nullptr);
    const auto bytes = query();
    StagingPreparationTrace seen{};
    trace.end(&seen);
    const auto calls = scope.finish();
    require_all(!seen.host_allocations && !seen.native_calls && !calls.vendor_query &&
                    !calls.ddla_alloc && !calls.ddla_free && !calls.pinned_alloc &&
                    !calls.raw_gpu_alloc && !calls.resource_setup && !calls.copies &&
                    !calls.launches && !calls.syncs,
                "spectral staging query was not pure");
    if (!h->myid)
        std::cout << "STAGING_QUERY_PURE operation=" << label << " major=" << h->major
                  << " host_new=0 native_calls=0 native_query=0\n";
    return bytes;
}
template <class Run>
StagingPreparationTrace spectral_observe(DdlaHandle_t h, const Trace &trace,
                                         const std::array<std::size_t, 2> &bytes, const char *label,
                                         Run &&run, bool bound = false, int relation = 0)
{
    cold(h);
    const auto slots = reserve(h, bytes);
    DDLA_CHECK(ddlaSynchronize(h));
    workspace_test::Scope scope;
    trace.refresh_slots(h);
    trace.begin(slots[0], slots[1]);
    run();
    DDLA_CHECK(ddlaSynchronize(h));
    StagingPreparationTrace seen{};
    trace.end(&seen);
    const auto calls = scope.finish();
    const bool exact = !bound || relation > 0;
    const bool shrink = relation < 0 && transport == DdlaCommunication::HostStagedMPI;
    if (seen.host_allocations || calls.vendor_query ||
        seen.slot_bytes[0] > bytes[0] || seen.slot_bytes[1] > bytes[1] ||
        (exact && (seen.slot_bytes[0] != bytes[0] || seen.slot_bytes[1] != bytes[1])) ||
        (shrink && seen.slot_bytes[0] >= bytes[0]))
        std::cerr << "STAGING_OBSERVER_FAILURE operation=" << label << " rank=" << h->myid
                  << " grid=" << h->nprows_ << 'x' << h->npcols_ << " major=" << h->major
                  << " requested=" << bytes[0] << ',' << bytes[1]
                  << " observed=" << seen.slot_bytes[0] << ',' << seen.slot_bytes[1]
                  << " native_calls=" << seen.native_calls << " host_new=" << seen.host_allocations
                  << " native_query=" << calls.vendor_query << " messages=" << seen.messages
                  << " bound=" << bound << " relation=" << relation << std::endl;
    if(seen.host_allocations)trace.allocation_dump(h->myid,label);
    require_all(
        !seen.host_allocations && !calls.vendor_query,
        "cold reserved spectral raw allocated host memory or made a native workspace query");
    require_all(seen.slot_bytes[0] <= bytes[0] && seen.slot_bytes[1] <= bytes[1],
                "spectral staging query underestimates actual native MPI payload");
    if (!bound || relation > 0)
        require_all(seen.slot_bytes[0] == bytes[0] && seen.slot_bytes[1] == bytes[1],
                    "exact/attaining spectral staging query differs from actual payload");
    if (relation < 0 && transport == DdlaCommunication::HostStagedMPI)
        require_all(seen.slot_bytes[0] < bytes[0],
                    "deflated PBDSDC payload did not shrink below its reachable supremum");
    std::cout << (bound ? "STAGING_BOUND_RESERVED" : "STAGING_COLD_RESERVED")
              << " operation=" << label << " rank=" << h->myid << " major=" << h->major
              << " requested=" << bytes[0] << ',' << bytes[1] << " observed=" << seen.slot_bytes[0]
              << ',' << seen.slot_bytes[1] << " host_new=" << seen.host_allocations
              << " native_query=" << calls.vendor_query << " messages=" << seen.messages
              << " relation=" << relation << std::endl;
    return seen;
}

template <class T> void spectral_qr(int pr, int pc, const Trace &trace, bool inplace, int m, int n)
{
    auto h = create(GPU, pr, pc);
    {
        int d[DDLA_DLEN_];
        descriptor(h, d, m + 2, n + 2, 3, 2);
        const auto input =
            make_local<T>(h, d, [](int i, int j) { return spectral_pattern<T>(i, j); });
        DeviceBuffer<T> a(h, input.size());
        upload(h, a.ptr, input);
        const int columns = num_loc(n, d[DDLA_NB_], h->mypcol_, 0, pc);
        std::vector<T> tau(std::max(1, columns));
        std::size_t bb = 0, hb = 0;
        qr_impl::pgeqrf_workspace_size<T>(h, 'U', inplace, m, n, a.ptr, d, tau.data(), &bb, &hb);
        ddla::vector<char> work(h, bb);
        std::vector<char> host(hb);
        const auto bytes =
            spectral_query(h, trace, "qr", [&]
                           { return qr_impl::pgeqrf_staging_bytes<T>(h, 'U', inplace, m, n, d); });
        spectral_observe(h, trace, bytes, "qr",
                         [&]
                         {
                             qr_impl::pgeqrf_run<GPU, T>(h, 'U', inplace, m, n, a.ptr, d,
                                                         tau.data(), work.data(), bb, host.data(),
                                                         hb);
                         });
        bool finite = true;
        for (int j = 0; j < columns; ++j)
            finite = finite && std::isfinite(double(std::abs(tau[j])));
        require_all(finite, "reserved QR returned nonfinite tau");
        if (!h->myid)
            std::cout << "SPECTRAL_PRIMITIVE_CASE operation=qr inplace=" << inplace << " m=" << m
                      << " n=" << n << " type_bytes=" << sizeof(T) << '\n';
    }
    DDLA_CHECK(ddlaDestroy(h));
}

template <class T> void spectral_punmbr(int pr, int pc, const Trace &trace, bool left)
{
    auto h = create(GPU, pr, pc);
    {
        const int m = left ? 17 : 9, n = left ? 9 : 17, k = 5;
        int da[DDLA_DLEN_], dc[DDLA_DLEN_];
        descriptor(h, da, left ? m : k, left ? k : n, left ? 3 : 2, left ? 2 : 3);
        descriptor(h, dc, m + 2, n + 2, left ? 3 : 2, left ? 2 : 3);
        const auto reflectors = make_local<T>(h, da, [](int, int) { return T(0); });
        const auto initial =
            make_local<T>(h, dc, [](int i, int j) { return spectral_pattern<T>(i, j); });
        DeviceBuffer<T> a(h, reflectors.size()), c(h, initial.size());
        upload(h, a.ptr, reflectors);
        upload(h, c.ptr, initial);
        std::vector<T> tau(k, T(2));
        const auto p =
            sp::punmbr_check(h, GPU, std::is_arithmetic_v<T>, left ? 'Q' : 'P', left ? 'L' : 'R',
                             'N', m, n, k, a.ptr, da, tau.data(), c.ptr, dc);
        std::size_t bb = 0, hb = 0;
        sp::punmbr_workspace_size<T>(h, p, &bb, &hb);
        ddla::vector<char> work(h, bb);
        std::vector<char> host(hb);
        const auto bytes =
            spectral_query(h, trace, "punmbr", [&] { return sp::punmbr_staging_bytes<T>(h, p); });
        int info = -1;
        spectral_observe(h, trace, bytes, "punmbr",
                         [&]
                         {
                             sp::punmbr_run<GPU, T>(h, p, a.ptr, da, tau.data(), c.ptr, dc, info,
                                                    work.data(), bb, host.data(), hb);
                         });
        require_all(info == 0, "reserved PUNMBR failed");
        const auto output = download(h, c.ptr, initial.size());
        double error = 0;
        for (int j = 0; j < ddla_test::n_loc(h, dc); ++j)
            for (int i = 0; i < ddla_test::m_loc(h, dc); ++i)
            {
                const int gi = indx_l2g_r(dc, h, i), gj = indx_l2g_c(dc, h, j);
                const auto at = i + std::size_t(j) * dc[DDLA_LLD_];
                const bool flipped = gi < m && gj < n && (left ? gi : gj) < k;
                error = std::max(error, workspace_test::relative_error(
                                            output[at], flipped ? -initial[at] : initial[at]));
            }
        require_close(h, "manual coordinate-reflector PUNMBR oracle", error,
                      std::is_same_v<typename real_type<T>::type, float> ? 2e-5 : 2e-12);
    }
    DDLA_CHECK(ddlaDestroy(h));
}

template <class T> void spectral_gebrd(int pr, int pc, const Trace &trace, int m, int n)
{
    auto h = create(GPU, pr, pc);
    {
        using R = typename real_type<T>::type;
        int d[DDLA_DLEN_];
        descriptor(h, d, m + 2, n + 2, 3, 2);
        const auto initial =
            make_local<T>(h, d, [](int i, int j) { return spectral_pattern<T>(i, j); });
        DeviceBuffer<T> a(h, initial.size());
        upload(h, a.ptr, initial);
        const auto p = sp::pgebrd_geometry(h, GPU, m, n, d);
        const int k = std::min(m, n);
        std::vector<R> diagonal(k), off(std::max(1, k - 1));
        std::vector<T> q(k), v(k);
        std::size_t bb = 0;
        sp::pgebrd_workspace_size<T>(h, p, &bb);
        ddla::vector<char> work(h, bb);
        const auto bytes =
            spectral_query(h, trace, "pgebrd", [&] { return sp::pgebrd_staging_bytes<T>(h, p); });
        int info = -1;
        spectral_observe(h, trace, bytes, "pgebrd",
                         [&]
                         {
                             sp::pgebrd_run<GPU, T>(h, p, a.ptr, d, diagonal.data(), off.data(),
                                                    q.data(), v.data(), info, work.data(), bb);
                         });
        bool finite = info == 0;
        for (auto x : diagonal)
            finite = finite && std::isfinite(x);
        for (int i = 0; i < k - 1; ++i)
            finite = finite && std::isfinite(off[i]);
        require_all(finite, "reserved GEBRD returned nonfinite bidiagonal");
    }
    DDLA_CHECK(ddlaDestroy(h));
}

void spectral_transfer(int pr, int pc, const Trace &trace)
{
    auto h = create(GPU, pr, pc);
    {
        int d[DDLA_DLEN_];
        descriptor(h, d, 19, 17, 3, 2);
        const auto initial =
            make_local<Z>(h, d, [](int i, int j) { return spectral_pattern<Z>(i, j); });
        DeviceBuffer<Z> a(h, initial.size()),
            window(h, h->myid == h->rc_to_rank(pr - 1, pc - 1) ? 45 : 0);
        upload(h, a.ptr, initial);
        const int root = h->rc_to_rank(pr - 1, pc - 1);
        std::size_t bb = 0, hb = 0;
        matrix_impl::transfer_workspace_size<Z>(h, 9, 5, 2, 3, d, root, &bb, &hb);
        ddla::vector<char> work(h, bb);
        std::vector<char> host(hb);
        const auto bytes = spectral_query(
            h, trace, "matrix_transfer",
            [&] { return matrix_impl::transfer_staging_bytes(h, 9, 5, 2, 3, d, root, sizeof(Z)); });
        spectral_observe(
            h, trace, bytes, "matrix_transfer",
            [&]
            {
                matrix_impl::gather_run<GPU, Z>(h, 9, 5, a.ptr, 2, 3, d, root, window.ptr, 9,
                                                work.data(), bb, host.data(), hb);
                matrix_impl::scatter_run<GPU, Z>(h, 9, 5, a.ptr, 2, 3, d, root, window.ptr, 9, true,
                                                 work.data(), bb, host.data(), hb);
            });
        const auto after = download(h, a.ptr, initial.size());
        require_all(after == initial, "reserved gather/scatter round trip changed data");
    }
    DDLA_CHECK(ddlaDestroy(h));
}

void spectral_scatter_rows(int pr, int pc, const Trace &trace, bool transpose, int k)
{
    auto h = create(GPU, pr, pc);
    {
        const int m = 15, n = 11;
        const detail::RowShare share(k, h->nprocs);
        const int ld = std::max(1, share.count(h->myid));
        std::vector<double> source(std::size_t(ld) * k, 0.);
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < share.count(h->myid); ++i)
                source[i + std::size_t(j) * ld] = share.lo(h->myid) + i == j ? 1 : 0;
        DeviceBuffer<double> x(h, source.size());
        upload(h, x.ptr, source);
        int d[DDLA_DLEN_];
        descriptor(h, d, m + 2, n + 2, 3, 2);
        const auto initial = make_local<Z>(h, d, [](int, int) { return Z(8157); });
        DeviceBuffer<Z> y(h, initial.size());
        upload(h, y.ptr, initial);
        std::size_t bb = 0, hb = 0;
        matrix_impl::scatter_rows_workspace_size<Z>(h, k, &bb, &hb);
        ddla::vector<char> work(h, bb);
        std::vector<char> host(hb);
        const auto bytes = spectral_query(h, trace, "scatter_rows",
                                          [&]
                                          {
                                              return matrix_impl::scatter_rows_staging_bytes(
                                                  h, k, transpose, m, n, d, sizeof(Z));
                                          });
        spectral_observe(h, trace, bytes, "scatter_rows",
                         [&]
                         {
                             matrix_impl::scatter_rows_run<GPU, Z>(h, k, x.ptr, ld, transpose, m, n,
                                                                   y.ptr, d, work.data(), bb,
                                                                   host.data(), hb);
                         });
        const auto actual = download(h, y.ptr, initial.size());
        double error = 0;
        for (int j = 0; j < ddla_test::n_loc(h, d); ++j)
            for (int i = 0; i < ddla_test::m_loc(h, d); ++i)
            {
                const int gi = indx_l2g_r(d, h, i), gj = indx_l2g_c(d, h, j);
                const auto at = i + std::size_t(j) * d[DDLA_LLD_];
                const Z expected = gi < m && gj < n ? Z(gi == gj && gi < k ? 1 : 0) : initial[at];
                error = std::max(error, workspace_test::relative_error(actual[at], expected));
            }
        require_close(h, "reserved row-scatter clipping/extension", error, 0);
    }
    DDLA_CHECK(ddlaDestroy(h));
}

void spectral_vector(int pr, int pc, const Trace &trace, bool to_rows, int n)
{
    auto h = create(GPU, pr, pc);
    {
        const int nb = 2, nvc = 3;
        auto p = sp::ptranspose_vectors_geometry(h, GPU, n, nvc, nb, 1, 1, false, to_rows);
        const int nr = num_loc(n, nb, p.myprow, 0, p.nprows),
                  nc = num_loc(n, nb, p.mypcol, 0, p.npcols);
        p.ld_src = std::max(1, nr) + 2;
        p.ld_dst = std::max(1, nc) + 3;
        std::vector<Z> source(std::size_t(p.ld_src) * nvc, Z(8157)),
            initial(std::size_t(p.ld_dst) * nvc, Z(8157));
        for (int j = 0; j < nvc; ++j)
            for (int i = 0; i < nr; ++i)
                source[i + std::size_t(j) * p.ld_src] =
                    entry(indxl2g(i, nb, p.myprow, 0, p.nprows), j);
        DeviceBuffer<Z> x(h, source.size()), y(h, initial.size());
        upload(h, x.ptr, source);
        upload(h, y.ptr, initial);
        std::size_t bb = 0;
        sp::ptranspose_vectors_workspace_size<Z>(h, n, nvc, nb, &bb);
        ddla::vector<char> work(h, bb);
        const auto bytes =
            spectral_query(h, trace, "vector_transpose",
                           [&] { return sp::ptranspose_vectors_staging_bytes(h, p, sizeof(Z)); });
        spectral_observe(
            h, trace, bytes, "vector_transpose",
            [&] { sp::ptranspose_vectors_run<GPU, Z>(h, p, x.ptr, y.ptr, work.data(), bb); });
        const auto actual = download(h, y.ptr, initial.size());
        double error = 0;
        for (int j = 0; j < nvc; ++j)
            for (int i = 0; i < p.ld_dst; ++i)
            {
                const Z expected = i < nc ? entry(indxl2g(i, nb, p.mypcol, 0, p.npcols), j)
                                          : initial[i + std::size_t(j) * p.ld_dst];
                error = std::max(error, workspace_test::relative_error(
                                            actual[i + std::size_t(j) * p.ld_dst], expected));
            }
        require_close(h, "reserved vector transpose/empty owners", error, 0);
    }
    DDLA_CHECK(ddlaDestroy(h));
}

template <class T> std::complex<double> spectral_wide(T x)
{
    if constexpr (std::is_arithmetic_v<T>)
        return {double(x), 0};
    else
        return {double(x.real()), double(x.imag())};
}
template <class T>
std::vector<T> spectral_column(DdlaHandle_t h, const int *d, const std::vector<T> &local, int n,
                               int column, bool row = false)
{
    std::vector<T> result(n, T(0));
    for (int j = 0; j < ddla_test::n_loc(h, d); ++j)
        for (int i = 0; i < ddla_test::m_loc(h, d); ++i)
        {
            const int gi = indx_l2g_r(d, h, i), gj = indx_l2g_c(d, h, j);
            if ((row ? gi : gj) == column && (row ? gj : gi) < n)
                result[row ? gj : gi] = local[i + std::size_t(j) * d[DDLA_LLD_]];
        }
    if (n)
        MPI_Allreduce(MPI_IN_PLACE, result.data(), n, detail::mpi_datatype<T>(), MPI_SUM, h->comm);
    return result;
}
template <class T>
void spectral_diagonal_oracle(DdlaHandle_t h, int n, int nev, const int *d,
                              const std::vector<T> &local, const typename real_type<T>::type *w,
                              bool exact)
{
    double error = 0;
    for (int j = 0; j < nev; ++j)
    {
        if (!std::isfinite(w[j]))
            error = INFINITY;
        if (exact)
            error = std::max(error, std::abs(double(w[j]) - (j + 1)));
        const auto x = spectral_column(h, d, local, n, j);
        double norm = 0, residual = 0;
        for (int i = 0; i < n; ++i)
        {
            const auto v = spectral_wide(x[i]);
            norm += std::norm(v);
            residual += std::norm((i + 1 - double(w[j])) * v);
        }
        if (!std::isfinite(norm) || !std::isfinite(residual) || !(norm > 0))
            error = INFINITY;
        else if (exact)
            error = std::max(error, std::sqrt(residual) /
                                        ((n + std::abs(double(w[j]))) * std::sqrt(norm)));
    }
    require_close(h, "reserved Hermitian finite/diagonal spectral oracle", error,
                  std::is_same_v<typename real_type<T>::type, float> ? 5e-4 : 3e-9);
}
template <class T> void spectral_one_stage(int pr, int pc, const Trace &trace, int n,
                                             char job='V',char uplo='L',bool dense=false)
{
    auto h = create(GPU, pr, pc);
    {
        using R = typename real_type<T>::type;
        int da[DDLA_DLEN_], dz[DDLA_DLEN_];
        descriptor(h, da, n + 2, n + 2, 1, 1);
        descriptor(h, dz, n + 2, n + 2, 1, 1);
        const auto entry=[&](int i,int j){return dense?generalized_test::standard<T>(i,j,n):T(i==j?i+1:0);};
        auto initial = make_local<T>(h, da, [&](int i, int j) {
            return i<n&&j<n?((uplo=='L'?i>=j:i<=j)?entry(i,j):T(NAN)):T(8157);
        });
        DeviceBuffer<T> a(h, initial.size()), z(h, local_size(h, dz));
        initial.resize(a.count,T(8157));
        auto initial_z=make_local<T>(h,dz,[](int,int){return T(9173);});
        initial_z.resize(z.count,T(9173));
        upload(h,a.ptr,initial);upload(h,z.ptr,initial_z);
        constexpr int guard=4;
        std::vector<R> w(n+2*guard,R(-9137));
        auto* values=w.data()+guard;
        auto p = sp::one_stage_check<T>(h,GPU,job,uplo,n,a.ptr,da,values,
                                        job=='V'?z.ptr:nullptr,job=='V'?dz:nullptr,n);
        sp::OneStageResources resource(h, p);
        std::size_t bb = 0, hb = 0;
        sp::one_stage_workspace_size<T>(h, p, &bb, &hb);
        ddla::vector<char> work(h, bb);
        std::vector<char> host(hb);
        const auto bytes = spectral_query(h, trace, "one_stage",
                                          [&] { return sp::one_stage_staging_bytes<T>(h, p); });
        int info = -1;
        spectral_observe(h,trace,bytes,"one_stage",[&] {
            sp::one_stage_run<GPU,T>(h,p,resource,a.ptr,values,job=='V'?z.ptr:nullptr,info,
                                     nullptr,work.data(),bb,host.data(),hb);
        });
        require_all(info==0,"cold reserved OneStage failed");
        bool guards=true;
        for(int i=0;i<guard;++i)guards=guards&&w[i]==R(-9137)&&w[guard+n+i]==R(-9137);
        require_all(guards,"cold OneStage host eigenvalue guards changed");
        const auto after_a=download(h,a.ptr,a.count),after_z=download(h,z.ptr,z.count);
        const auto outside=[&](const int* d,std::size_t at){
            const int row=int(at%std::size_t(d[DDLA_LLD_])),col=int(at/std::size_t(d[DDLA_LLD_]));
            return row>=ddla_test::m_loc(h,d)||col>=ddla_test::n_loc(h,d)||
                indx_l2g_r(d,h,row)>=n||indx_l2g_c(d,h,col)>=n;
        };
        bool padding=true;
        for(std::size_t at=0;at<initial.size();++at)
            if(outside(da,at)&&std::memcmp(&after_a[at],&initial[at],sizeof(T)))padding=false;
        for(std::size_t at=0;at<initial_z.size();++at)
            if((job=='N'||outside(dz,at))&&std::memcmp(&after_z[at],&initial_z[at],sizeof(T)))padding=false;
        require_all(padding,"cold OneStage parent matrix/Z padding or job=N Z sentinel changed");
        double error=0,matrix_norm=0;
        for(int i=0;i<n;++i){
            double sum=0;for(int j=0;j<n;++j)sum+=std::abs(spectral_wide(entry(i,j)));
            matrix_norm=std::max(matrix_norm,sum);
            const double expected=dense?i-3.:i+1.;
            if(!std::isfinite(double(values[i])))error=INFINITY;
            else error=std::max(error,std::abs(double(values[i])-expected));
        }
        if(job=='V')for(int j=0;j<n;++j){
            // Gather one vector at a time; no root/per-rank dense matrix copy.
            const auto x=spectral_column(h,dz,after_z,n,j);
            double norm=0,residual=0;
            for(int i=0;i<n;++i){
                const auto xi=spectral_wide(x[i]);norm+=std::norm(xi);
                std::complex<double> ax=0.;
                for(int k=0;k<n;++k)ax+=spectral_wide(entry(i,k))*spectral_wide(x[k]);
                residual+=std::norm(ax-double(values[j])*xi);
            }
            if(!(norm>0)||!std::isfinite(norm)||!std::isfinite(residual))error=INFINITY;
            else{
                error=std::max(error,std::abs(norm-1.));
                error=std::max(error,std::sqrt(residual)/((matrix_norm+std::abs(double(values[j])))*std::sqrt(norm)));
            }
            for(int k=0;k<j;++k){
                const auto y=spectral_column(h,dz,after_z,n,k);std::complex<double> dot=0.;
                for(int i=0;i<n;++i)dot+=std::conj(spectral_wide(y[i]))*spectral_wide(x[i]);
                if(!std::isfinite(std::abs(dot)))error=INFINITY;
                else error=std::max(error,std::abs(dot));
            }
        }
        require_close(h,"cold OneStage independent spectrum/residual/orthogonality",error,
                      std::is_same_v<R,float>?5e-4:3e-9);
        if(!h->myid)std::cout<<"ONE_STAGE_STAGING_REGRESSION n="<<n<<" job="<<job<<" uplo="<<uplo
            <<" dense="<<dense<<" type_bytes="<<sizeof(T)<<" grid="<<pr<<'x'<<pc<<" major="<<h->major
            <<" exact_payload=1 host_new=0 native_query=0 eigenvalue_guards=1 parent_guards=1 no_Z_write_for_N="<<(job=='N')
            <<" residual_orthogonality="<<(job=='V')<<'\n';
    }
    DDLA_CHECK(ddlaDestroy(h));
}
template <class T>
void spectral_chebyshev(int pr, int pc, const Trace &trace, int n, int nev, int nex, bool warm,
                        bool alias = false)
{
    auto h = create(GPU, pr, pc);
    {
        using R = typename real_type<T>::type;
        int da[DDLA_DLEN_], dz[DDLA_DLEN_];
        descriptor(h, da, n + 2, n + 2, 3, 3);
        descriptor(h, dz, n + 2, nev + 2, 2, 1);
        if (alias)
            std::copy_n(da, DDLA_DLEN_, dz);
        const auto initial = make_local<T>(
            h, da, [&](int i, int j) { return i < n && j < n ? T(i == j ? i + 1 : 0) : T(8157); });
        const auto initial_z = make_local<T>(
            h, dz, [&](int i, int j)
            { return i < n && j < nev ? (warm ? T(i == n - 1 - j) : T(NAN)) : T(8157); });
        DeviceBuffer<T> a(h, initial.size()), separate(h, initial_z.size());
        T *z = alias ? a.ptr : separate.ptr;
        upload(h, a.ptr, initial);
        if (!alias)
            upload(h, z, initial_z);
        std::vector<R> w(nev + 8, R(-313));
        const double tol = std::is_same_v<R, float> ? 4e-5 : 2e-11;
        auto p = cb::check<T>(h, GPU, 'L', n, nev, nex, a.ptr, da, w.data() + 4, z, dz, tol, 1, 12,
                              warm);
        cb::Resources<T> resource(h, p);
        cb::prepare<GPU>(h, p, resource);
        std::size_t bb = 0, hb = 0;
        cb::workspace_size(p, &bb, &hb);
        ddla::vector<char> work(h, bb);
        std::vector<char> host(hb);
        const auto bytes =
            spectral_query(h, trace, "chebyshev", [&] { return cb::staging_bytes<T>(h, p); });
        int iterations = -1, converged = -1, info = -1;
        spectral_observe(h, trace, bytes, "chebyshev",
                         [&]
                         {
                             cb::run<GPU>(h, p, resource, a.ptr, w.data() + 4, z, iterations,
                                          converged, info, nullptr, work.data(), bb, host.data(),
                                          hb);
                         });
        require_all(info == cb::success || info == cb::iteration_limit,
                    "reserved Chebyshev failed before a Ritz output");
        for (int j = 0; j < 4; ++j)
            require_all(w[j] == R(-313) && w[nev + 4 + j] == R(-313),
                        "reserved Chebyshev overwrote nev scalar guards");
        spectral_diagonal_oracle(h, n, nev, dz, download(h, z, initial_z.size()), w.data() + 4,
                                 info == cb::success);
        if (!h->myid)
            std::cout << "SPECTRAL_CHEB_CASE n=" << n << " nev=" << nev << " nex=" << nex
                      << " warm=" << warm << " alias=" << alias << " type_bytes=" << sizeof(T)
                      << '\n';
    }
    DDLA_CHECK(ddlaDestroy(h));
}

// PBDSDC is the one value-dependent staged protocol. Nondeflated input must
// attain its reachable root bound; zero off-diagonals/repeated values must
// use less. Host factors are kept distributed and gathered one vector at a time.
void spectral_pbdsdc(int pr, int pc, const Trace &trace, bool deflated)
{
    auto h = create(GPU, pr, pc);
    {
        const int n = 13;
        const detail::RowShare share(n, h->nprocs);
        const int ld = std::max(1, share.count(h->myid));
        std::vector<double> d(n), e(n - 1), values(n);
        for (int i = 0; i < n; ++i)
            d[i] = deflated ? 1. : 2. + .017 * i;
        std::fill(e.begin(), e.end(), deflated ? 0. : .8);
        DeviceBuffer<double> u(h, std::size_t(ld) * n), v(h, std::size_t(ld) * n);
        setenv("DDLA_SVD_DC_LEAF", "4", 1);
        const auto p =
            sp::pbdsdc_check(h, GPU, 'U', n, d.data(), e.data(), values.data(), u.ptr, v.ptr, ld);
        unsetenv("DDLA_SVD_DC_LEAF");
        detail::DcHostTeam team(p.threads);
        std::size_t bb = 0, hb = 0;
        sp::pbdsdc_workspace_size<double>(h, p, &bb, &hb);
        ddla::vector<char> work(h, bb);
        std::vector<char> host(hb);
        const auto bytes = spectral_query(h, trace, "pbdsdc",
                                          [&] { return sp::pbdsdc_staging_bound<double>(h, p); });
        // Enumerate all smaller node/K values: the root expression dominates
        // every possible split/deflation extent and respects the rank share.
        const auto cap = std::min(512, detail::RowShare(n, h->nprocs).count(0));
        bool dominates = true;
        for (int node = 1; node <= n; ++node)
            for (int k = 1; k <= node; ++k)
                dominates =
                    dominates &&
                    (2 * std::size_t(k) * std::min(cap, detail::RowShare(k, h->nprocs).count(0)) *
                             sizeof(double) <=
                         bytes[0] ||
                     transport != DdlaCommunication::HostStagedMPI);
        require_all(dominates, "PBDSDC root bound does not dominate a possible node/share");
        int info = -1;
        spectral_observe(
            h, trace, bytes, "pbdsdc",
            [&]
            {
                sp::pbdsdc_run<GPU, double>(h, p, team, d.data(), e.data(), values.data(), u.ptr,
                                            v.ptr, info, work.data(), bb, host.data(), hb);
            },
            true, transport == DdlaCommunication::HostStagedMPI ? (deflated ? -1 : 1) : 0);
        bool okay = info == 0;
        for (int i = 0; i < n; ++i)
            okay = okay && std::isfinite(values[i]) && (i == 0 || values[i - 1] >= values[i]);
        require_all(okay, "reserved PBDSDC finite/sorted output failure");
        if (!h->myid)
            std::cout << "SPECTRAL_PBDSDC_CASE deflated=" << deflated
                      << " bound_attainment=" << !deflated << '\n';
    }
    DDLA_CHECK(ddlaDestroy(h));
}

void spectral_svd_oracle(DdlaHandle_t h, int m, int n, const int *du, const int *dv,
                         const std::vector<Z> &u, const std::vector<Z> &vt,
                         const std::vector<double> &values)
{
    const int k = std::min(m, n);
    double error = 0;
    for (int j = 0; j < k; ++j)
    {
        if (!std::isfinite(values[j]))
            error = INFINITY;
        error = std::max(error, std::abs(values[j] - (k - j)));
        const auto x = spectral_column(h, du, u, m, j),
                   row = spectral_column(h, dv, vt, n, j, true);
        double norm = 0, residual = 0;
        for (int i = 0; i < m; ++i)
        {
            const Z right = i < k ? std::conj(row[i]) : Z(0);
            norm += std::norm(x[i]);
            residual += std::norm(double(i + 1) * right - values[j] * x[i]);
        }
        for (int i = 0; i < n; ++i)
        {
            const Z left = i < k ? x[i] : Z(0), right = std::conj(row[i]);
            norm += std::norm(right);
            residual += std::norm(double(i + 1) * left - values[j] * right);
        }
        if (!std::isfinite(norm) || !std::isfinite(residual) || !(norm > 0))
            error = INFINITY;
        else
            error = std::max(error,
                             std::sqrt(residual) / ((k + std::abs(values[j])) * std::sqrt(norm)));
    }
    require_close(h, "reserved shape SVD independent singular-pair oracle", error, 2e-9);
}
void spectral_svd(int pr, int pc, const Trace &trace, bool wide, bool small_leaf)
{
    const char* prior_shape=std::getenv("DDLA_SVD_SHAPE");
    const bool had_shape=prior_shape!=nullptr;const std::string saved_shape=prior_shape?prior_shape:"";
    setenv("DDLA_SVD_SHAPE","qr",1);
    auto h = create(GPU, pr, pc);
    {
        const int k = small_leaf ? 19 : std::max(12, 4 * h->nprocs), m = wide ? k : 4 * k,
                  n = wide ? 4 * k : k;
        int da[DDLA_DLEN_], du[DDLA_DLEN_], dv[DDLA_DLEN_];
        descriptor(h, da, m, n, 1, 1);
        descriptor(h, du, m, k, 1, 1);
        descriptor(h, dv, k, n, 1, 1);
        const auto initial =
            make_local<Z>(h, da, [&](int i, int j) { return Z(i == j ? i + 1 : 0); });
        DeviceBuffer<Z> a(h, initial.size()), u(h, local_size(h, du)), v(h, local_size(h, dv));
        upload(h, a.ptr, initial);
        std::vector<double> values(k);
        if (small_leaf)
            setenv("DDLA_SVD_DC_LEAF", "4", 1);
        const auto p = sp::pgesvd_check<Z>(h, GPU, 'V', 'V', m, n, a.ptr, da, values.data(), u.ptr,
                                           du, v.ptr, dv);
        if (small_leaf)
            unsetenv("DDLA_SVD_DC_LEAF");
        std::optional<detail::DcHostTeam> team;
        if (const int threads = sp::pgesvd_team_threads(p))
            team.emplace(threads);
        std::size_t bb = 0, hb = 0;
        sp::pgesvd_workspace_size<Z>(h, p, &bb, &hb);
        ddla::vector<char> work(h, bb);
        std::vector<char> host(hb);
        const auto bytes =
            spectral_query(h, trace, "pgesvd", [&] { return sp::pgesvd_staging_bytes<Z>(h, p); });
        int info = -1;
        spectral_observe(
            h, trace, bytes, "pgesvd",
            [&]
            {
                sp::pgesvd_run<GPU, Z>(h, p, team ? &*team : nullptr, a.ptr, values.data(), u.ptr,
                                       v.ptr, info, work.data(), bb, host.data(), hb);
            },
            small_leaf);
        require_all(info == 0, "reserved shape SVD failed");
        spectral_svd_oracle(h, m, n, du, dv, download(h, u.ptr, local_size(h, du)),
                            download(h, v.ptr, local_size(h, dv)), values);
        if (!h->myid)
            std::cout << "SPECTRAL_SVD_CASE route=" << int(p.route) << " wide=" << wide
                      << " small_leaf=" << small_leaf << " m=" << m << " n=" << n << '\n';
    }
    DDLA_CHECK(ddlaDestroy(h));
    if(had_shape)setenv("DDLA_SVD_SHAPE",saved_shape.c_str(),1);else unsetenv("DDLA_SVD_SHAPE");
}

void chebyshev_public_staging_fault(int pr, int pc, const Trace &trace, int slot)
{
    auto h = create(GPU, pr, pc);
    {
        constexpr int nev = 3, nex = 2, parent = 45;
        int da[DDLA_DLEN_], dz[DDLA_DLEN_];
        descriptor(h, da, parent, parent, 3, 3);
        descriptor(h, dz, parent, nev + 2, 2, 1);
        DeviceBuffer<Z> a(h, local_size(h, da)), z(h, local_size(h, dz));
        std::vector<double> w(nev + 8, -313.);
        int n = 0, target_rank = h->nprocs;
        std::array<std::size_t, 2> bytes{};
        for (int candidate = 13; candidate <= 39; ++candidate)
        {
            auto p = cb::check<Z>(h, GPU, 'L', candidate, nev, nex, a.ptr, da, w.data() + 4, z.ptr,
                                  dz, 2e-11, 80, 12, false);
            cb::Resources<Z> resource(h, p);
            cb::prepare<GPU>(h, p, resource);
            bytes = cb::staging_bytes<Z>(h, p);
            const auto b = bytes[slot];
            // Distinguish the actual std::byte cache allocation from every
            // unpadded owned linear vector in the public prepared resources.
            const bool unique = b && b % 256 && b != p.stats_elements * sizeof(double) &&
                                b != 2 * std::size_t(p.q) * sizeof(double) &&
                                b != std::size_t(p.q) * sizeof(Z) &&
                                b != std::size_t(std::max(1, p.columns)) * sizeof(Z) &&
                                b != std::size_t(p.q) * sizeof(double);
            int owner = unique ? h->myid : h->nprocs;
            MPI_Allreduce(&owner, &target_rank, 1, MPI_INT, MPI_MIN, h->comm);
            if (target_rank < h->nprocs)
            {
                n = candidate;
                break;
            }
        }
        require_all(n > 0, "no uniquely identified Chebyshev staging-slot allocation geometry");
        const auto initial = make_local<Z>(
            h, da, [&](int i, int j)
            { return i < n && j < n ? (i >= j ? Z(i == j ? i + 1 : 0) : Z(NAN)) : Z(8157); });
        const auto initial_z = make_local<Z>(h, dz, [](int, int) { return Z(717); });
        upload(h, a.ptr, initial);
        upload(h, z.ptr, initial_z);
        cold(h);
        const auto target = bytes[slot];
        const unsigned ordinal = slot == 1 && bytes[0] == target ? 2 : 1;
        trace.arm(h->myid == target_rank ? target : 0, ordinal);
        int iterations = -11, converged = -22, info = -33;
        bool threw = false;
        try
        {
            pheevd_chebyshev<Z>(h, 'L', n, nev, nex, a.ptr, da, w.data() + 4, z.ptr, dz, 2e-11, 80,
                                12, false, iterations, converged, info);
        }
        catch (const std::exception &)
        {
            threw = true;
        }
        trace.disarm();
        std::cerr << "STAGING_FAULT_OBSERVATION operation=chebyshev rank=" << h->myid
                  << " slot=" << slot << " target_rank=" << target_rank << " target=" << target
                  << " bytes=" << bytes[0] << ',' << bytes[1] << " ordinal=" << ordinal
                  << " threw=" << threw << " fired=" << trace.fired()
                  << " failed_bytes=" << trace.failed_bytes() << " iterations=" << iterations
                  << " converged=" << converged << " info=" << info << std::endl;
        if(!threw||trace.fired()!=unsigned(h->myid==target_rank)||
           (trace.fired()&&trace.failed_bytes()!=target))trace.dump(h->myid,"chebyshev");
        require_all(threw, "Chebyshev staging failure was not agreed before raw execution");
        require_all(
            trace.fired() == unsigned(h->myid == target_rank) &&
                (!trace.fired() || trace.failed_bytes() == target),
            "actual Chebyshev staging allocation fault did not fire exactly once on target rank");
        const auto after = download(h, a.ptr, initial.size()),
                   after_z = download(h, z.ptr, initial_z.size());
        bool unchanged =
            !std::memcmp(after.data(), initial.data(), initial.size() * sizeof(Z)) &&
            !std::memcmp(after_z.data(), initial_z.data(), initial_z.size() * sizeof(Z)) &&
            iterations == -11 && converged == -22 && info == -33;
        for (double x : w)
            unchanged = unchanged && x == -313.;
        require_all(unchanged, "Chebyshev failed reservation changed A/Z/w/status");
        pheevd_chebyshev<Z>(h, 'L', n, nev, nex, a.ptr, da, w.data() + 4, z.ptr, dz, 2e-11, 80, 12,
                            false, iterations, converged, info);
        require_all(info == 0 && converged == nev, "same-handle Chebyshev staging retry failed");
        spectral_diagonal_oracle(h, n, nev, dz, download(h, z.ptr, initial_z.size()), w.data() + 4,
                                 true);
        if (!h->myid)
            std::cout
                << "STAGING_PUBLIC_FAULT_PASS operation=chebyshev slot=" << slot
                << " target_rank=" << target_rank << " major=" << h->major
                << " all_rank_throw=1 input_output_status_unchanged=1 same_handle_recovery=1\n";
    }
    DDLA_CHECK(ddlaDestroy(h));
}

void svd_public_staging_fault(int pr, int pc, const Trace &trace, bool wide, int slot)
{
    const char* prior_shape=std::getenv("DDLA_SVD_SHAPE");
    const bool had_shape=prior_shape!=nullptr;const std::string saved_shape=prior_shape?prior_shape:"";
    setenv("DDLA_SVD_SHAPE","qr",1);
    auto h = create(GPU, pr, pc);
    {
        const int first_k = std::max(12, 4 * h->nprocs), max_k = first_k + 9, max_long = 4 * max_k;
        int da[DDLA_DLEN_], du[DDLA_DLEN_], dv[DDLA_DLEN_];
        descriptor(h, da, wide ? max_k : max_long, wide ? max_long : max_k, 1, 1);
        descriptor(h, du, wide ? max_k : max_long, max_k, 1, 1);
        descriptor(h, dv, max_k, wide ? max_long : max_k, 1, 1);
        DeviceBuffer<Z> a(h, local_size(h, da)), u(h, local_size(h, du)), v(h, local_size(h, dv));
        std::vector<double> values(max_k + 8, -313.);
        int k = 0, target_rank = h->nprocs;
        std::array<std::size_t, 2> bytes{};
        for (int candidate = first_k; candidate <= max_k; ++candidate)
        {
            const int m = wide ? candidate : 4 * candidate, n = wide ? 4 * candidate : candidate;
            const auto p = sp::pgesvd_check<Z>(h, GPU, 'V', 'V', m, n, a.ptr, da, values.data() + 4,
                                               u.ptr, du, v.ptr, dv);
            bytes = sp::pgesvd_staging_bytes<Z>(h, p);
            std::size_t bb = 0, hb = 0;
            sp::pgesvd_workspace_size<Z>(h, p, &bb, &hb);
            const auto b = bytes[slot];
            const bool unique = b && b % 256 && b != hb;
            int owner = unique ? h->myid : h->nprocs;
            MPI_Allreduce(&owner, &target_rank, 1, MPI_INT, MPI_MIN, h->comm);
            if (target_rank < h->nprocs)
            {
                k = candidate;
                break;
            }
        }
        require_all(k > 0, "no uniquely identified SVD staging-slot allocation geometry");
        const int m = wide ? k : 4 * k, n = wide ? 4 * k : k;
        const auto initial = make_local<Z>(
            h, da, [&](int i, int j) { return i < m && j < n ? Z(i == j ? i + 1 : 0) : Z(8157); });
        const auto initial_u = make_local<Z>(h, du, [](int, int) { return Z(717); }),
                   initial_v = make_local<Z>(h, dv, [](int, int) { return Z(717); });
        upload(h, a.ptr, initial);
        upload(h, u.ptr, initial_u);
        upload(h, v.ptr, initial_v);
        cold(h);
        const auto target = bytes[slot];
        const unsigned ordinal = slot == 1 && bytes[0] == target ? 2 : 1;
        trace.arm(h->myid == target_rank ? target : 0, ordinal);
        // Existing pgesvd resets INFO to zero before validation/preparation.
        // Preserve that behavior, rather than asserting an invented sentinel contract.
        int info = 0;
        bool threw = false;
        try
        {
            pgesvd<Z>(h, 'V', 'V', m, n, a.ptr, da, values.data() + 4, u.ptr, du, v.ptr, dv, info);
        }
        catch (const std::exception &)
        {
            threw = true;
        }
        trace.disarm();
        std::cerr << "STAGING_FAULT_OBSERVATION operation=pgesvd rank=" << h->myid
                  << " wide=" << wide << " slot=" << slot << " target_rank=" << target_rank
                  << " target=" << target << " bytes=" << bytes[0] << ',' << bytes[1]
                  << " ordinal=" << ordinal << " threw=" << threw << " fired=" << trace.fired()
                  << " failed_bytes=" << trace.failed_bytes() << " info=" << info << std::endl;
        if(!threw||trace.fired()!=unsigned(h->myid==target_rank)||
           (trace.fired()&&trace.failed_bytes()!=target))trace.dump(h->myid,"pgesvd");
        require_all(threw, "SVD staging allocation failure was not collectively agreed");
        require_all(trace.fired() == unsigned(h->myid == target_rank) &&
                        (!trace.fired() || trace.failed_bytes() == target),
                    "actual SVD staging fault did not fire once on target rank");
        const auto after = download(h, a.ptr, initial.size()),
                   after_u = download(h, u.ptr, initial_u.size()),
                   after_v = download(h, v.ptr, initial_v.size());
        bool unchanged =
            !std::memcmp(after.data(), initial.data(), initial.size() * sizeof(Z)) &&
            !std::memcmp(after_u.data(), initial_u.data(), initial_u.size() * sizeof(Z)) &&
            !std::memcmp(after_v.data(), initial_v.data(), initial_v.size() * sizeof(Z)) &&
            info == 0;
        for (double x : values)
            unchanged = unchanged && x == -313.;
        require_all(unchanged,
                    "SVD failed reservation changed A/U/VT/s or its existing INFO reset");
        pgesvd<Z>(h, 'V', 'V', m, n, a.ptr, da, values.data() + 4, u.ptr, du, v.ptr, dv, info);
        require_all(info == 0, "same-handle SVD staging retry failed");
        std::vector<double> active(values.begin() + 4, values.begin() + 4 + k);
        spectral_svd_oracle(h, m, n, du, dv, download(h, u.ptr, initial_u.size()),
                            download(h, v.ptr, initial_v.size()), active);
        if (!h->myid)
            std::cout << "STAGING_PUBLIC_FAULT_PASS operation=pgesvd wide=" << wide
                      << " slot=" << slot << " target_rank=" << target_rank << " major=" << h->major
                      << " all_rank_throw=1 output_unchanged=1 existing_info_reset_preserved=1 "
                         "same_handle_recovery=1\n";
    }
    DDLA_CHECK(ddlaDestroy(h));
    if(had_shape)setenv("DDLA_SVD_SHAPE",saved_shape.c_str(),1);else unsetenv("DDLA_SVD_SHAPE");
}

void run_spectral_staging(int pr, int pc, const Trace &trace)
{
    spectral_transfer(pr, pc, trace);
    for (bool transpose : {false, true})
        for (int k : {1, 13})
            spectral_scatter_rows(pr, pc, trace, transpose, k);
    for (bool rows : {false, true})
        for (int n : {0, 1, 13})
            spectral_vector(pr, pc, trace, rows, n);
#define SPECTRAL_TYPE(T)                                                                           \
    for (bool inplace : {false, true})                                                             \
    {                                                                                              \
        spectral_qr<T>(pr, pc, trace, inplace, 17, 5);                                             \
        spectral_qr<T>(pr, pc, trace, inplace, 1, 1);                                              \
    }                                                                                              \
    for (bool left : {false, true})                                                                \
        spectral_punmbr<T>(pr, pc, trace, left);                                                   \
    spectral_gebrd<T>(pr, pc, trace, 17, 13);                                                      \
    spectral_gebrd<T>(pr, pc, trace, 13, 17);                                                      \
    spectral_gebrd<T>(pr, pc, trace, 1, 1);                                                        \
    spectral_one_stage<T>(pr, pc, trace, 7);                                                       \
    spectral_one_stage<T>(pr, pc, trace, 1);                                                       \
    spectral_chebyshev<T>(pr, pc, trace, 17, 3, 2, false);                                         \
    spectral_chebyshev<T>(pr, pc, trace, 17, 3, 2, true);                                          \
    spectral_chebyshev<T>(pr, pc, trace, 1, 1, 0, false);                                          \
    spectral_chebyshev<T>(pr, pc, trace, 0, 0, 0, false)
    SPECTRAL_TYPE(float);
    SPECTRAL_TYPE(double);
    SPECTRAL_TYPE(std::complex<float>);
    SPECTRAL_TYPE(Z);
#undef SPECTRAL_TYPE
    spectral_chebyshev<Z>(pr, pc, trace, 17, 3, 2, false, true);
    spectral_chebyshev<Z>(pr, pc, trace, 17, 3, 2, true, true);
    spectral_pbdsdc(pr, pc, trace, false);
    spectral_pbdsdc(pr, pc, trace, true);
    for (bool wide : {false, true})
    {
        spectral_svd(pr, pc, trace, wide, false);
        spectral_svd(pr, pc, trace, wide, true);
    }
}
#endif
