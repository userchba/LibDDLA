// Included inside the staging fixture's namespace. Reuse the calibrated native
// payload/fault interposer; no new attribution implementation lives here.
#if DDLA_HAS_GPU
namespace composite_staging {
namespace sp=ddla::spectral_impl;
struct Method {
    bool present=std::getenv("DDLA_EIGEN_METHOD")!=nullptr;
    std::string saved=present?std::getenv("DDLA_EIGEN_METHOD"):"";
    explicit Method(const char* value){setenv("DDLA_EIGEN_METHOD",value,1);}
    ~Method(){if(present)setenv("DDLA_EIGEN_METHOD",saved.c_str(),1);else unsetenv("DDLA_EIGEN_METHOD");}
};
struct Problem {
    DdlaHandle_t h;int n,nev;char uplo;
    std::array<int,DDLA_DLEN_> da{},db{},dz{};
    std::vector<Z> initial_a,initial_b,initial_z;
    DeviceBuffer<Z> a,b,z;
    std::vector<double>w;
    Problem(DdlaHandle_t handle,int order,int count,char triangle)
        :h(handle),n(order),nev(count),uplo(triangle),a(h,extent(h,n,n,da)),
         b(h,extent(h,n,n,db)),z(h,extent(h,n,nev,dz)),w(n,-313.)
    {
        initial_a=make_local<Z>(h,da.data(),[&](int i,int j){return i<n&&j<n?
            ((uplo=='L'?i>=j:i<=j)?Z(i==j?i+1:0):Z(NAN)):Z(8157);});
        initial_b=make_local<Z>(h,db.data(),[&](int i,int j){return i<n&&j<n?
            ((uplo=='L'?i>=j:i<=j)?Z(i==j?1:0):Z(NAN)):Z(8157);});
        initial_z=make_local<Z>(h,dz.data(),[](int,int){return Z(717);});reset();
    }
    static std::size_t extent(DdlaHandle_t h,int m,int n,std::array<int,DDLA_DLEN_>& d){
        descriptor(h,d.data(),m+2,n+2,4,4);return local_size(h,d.data());
    }
    void reset(){upload(h,a.ptr,initial_a);upload(h,b.ptr,initial_b);upload(h,z.ptr,initial_z);
        std::fill(w.begin(),w.end(),-313.);DDLA_CHECK(ddlaSynchronize(h));}
    sp::PhegvdPlan check(){auto p=sp::phegvd_check<GPU,Z>(h,'V',uplo,n,nev,a.ptr,da.data(),b.ptr,db.data(),w.data(),z.ptr,dz.data());
        p.complete=sp::pherm_complete_check(h,GPU,uplo,n,a.ptr,da.data());return p;}
    void oracle(){
        spectral_diagonal_oracle(h,n,nev,dz.data(),download(h,z.ptr,initial_z.size()),w.data(),true);
    }
};
struct Prepared {
    sp::PhegvdPlan p;std::unique_ptr<sp::Phb2stTopology> topology;
    sp::PheevdDriverPlan eig;std::optional<sp::PheevdDriverResources<Z>> resources;
    sp::PhegvdWorkspace q;
    Prepared(Problem& x,const char* method,char two_method='A',DdlaBackend reduction=DdlaBackend::CPU){
        Method policy(method);p=x.check();
        auto route=detail::pheevd_select<GPU,Z>(x.h,'V',x.n,x.nev,x.da.data(),x.dz.data());
        if(route==detail::PheevdRoute::TwoStage)topology=sp::phb2st_topology_prepare(x.h);
        eig=sp::pheevd_driver_check<GPU,Z>(x.h,route,'V',x.uplo,x.n,x.nev,x.a.ptr,x.da.data(),x.w.data(),x.z.ptr,x.dz.data(),topology.get());
        if(eig.two&&two_method!='A')eig.two=sp::pheevd_check<Z>(x.h,GPU,'V',x.uplo,x.n,x.a.ptr,x.da.data(),x.w.data(),x.z.ptr,x.dz.data(),*topology,two_method,nullptr,reduction,x.nev);
        resources.emplace(x.h,eig);q=sp::phegvd_workspace_size<Z>(x.h,p,eig,*resources,x.a.ptr,x.b.ptr,x.w.data());
    }
    void release(){resources->release();resources.reset();if(topology){topology->release();topology.reset();}}
};
void phase_payload(DdlaHandle_t h,const Trace& trace,const std::array<std::size_t,2>& bytes,const char* label,
                   const std::function<void()>& run,bool cold_cache=true)
{
    if(cold_cache)cold(h);
    sp::phegvd_prepare_staging(h,bytes);const auto slots=h->host_buffers<std::byte>(0,0);
    DDLA_CHECK(ddlaSynchronize(h));workspace_test::Scope scope;
    trace.begin(slots[0],slots[1]);run();DDLA_CHECK(ddlaSynchronize(h));
    StagingPreparationTrace seen{};trace.end(&seen);const auto work=scope.finish();
    if(seen.host_allocations||work.host_new||work.vendor_query||seen.slot_bytes[0]!=bytes[0]||seen.slot_bytes[1]!=bytes[1])
        std::cerr<<"PHEGVD_PHASE_PAYLOAD_MISMATCH operation="<<label<<" rank="<<h->myid
            <<" requested="<<bytes[0]<<','<<bytes[1]<<" observed="<<seen.slot_bytes[0]<<','<<seen.slot_bytes[1]
            <<" host_new="<<seen.host_allocations<<" workspace_host_new="<<work.host_new<<" native_query="<<work.vendor_query<<std::endl;
    if(seen.host_allocations)trace.allocation_dump(h->myid,label);
    require_all(!seen.host_allocations&&!work.host_new&&!work.vendor_query,"cold PHEGVD phase allocated or queried in execution");
    require_all(seen.slot_bytes[0]==bytes[0]&&seen.slot_bytes[1]==bytes[1],"PHEGVD phase staging payload differs from successful geometry query");
    const auto capacity=h->host_buffer_capacities();
    std::cout<<"PHEGVD_PHASE_COLD_RESERVED operation="<<label<<" rank="<<h->myid<<" major="<<h->major
        <<" requested="<<bytes[0]<<','<<bytes[1]<<" observed="<<seen.slot_bytes[0]<<','<<seen.slot_bytes[1]<<" cache_capacity="<<capacity[0]<<','<<capacity[1]<<" host_new=0\n";
}
void raw(int pr,int pc,const Trace& trace,const char* method,char uplo,bool fault_case=false,char two_method='A',DdlaBackend reduction=DdlaBackend::CPU)
{
    auto h=create(GPU,pr,pc);
    {
        Problem x(h,17,5,uplo);Prepared prepared(x,method,two_method,reduction);
        auto& p=prepared.p;auto& eig=prepared.eig;auto& r=*prepared.resources;const auto& q=prepared.q;
        // Pure root-only query must not enter a collective.
        bool pure=true;MPI_Barrier(h->comm);
        if(!h->myid){workspace_test::Scope scope(true);trace.begin(nullptr,nullptr);
            (void)sp::phegvd_staging_bytes<Z>(h,p,eig,r);StagingPreparationTrace seen{};trace.end(&seen);
            const auto calls=scope.finish();pure=!seen.host_allocations&&!seen.native_calls&&!calls.host_new&&!calls.vendor_query&&!calls.ddla_alloc&&!calls.resource_setup&&!calls.copies&&!calls.launches&&!calls.syncs;}
        MPI_Barrier(h->comm);require_all(pure,"root-only PHEGVD staging query is not pure");
        const auto bytes=spectral_query(h,trace,"phegvd",[&]{return sp::phegvd_staging_bytes<Z>(h,p,eig,r);});
        ddla::vector<char> work(h,q.backend_bytes);std::vector<char> host(q.host_bytes);
        int info=-33;
        if(fault_case&&transport==DdlaCommunication::HostStagedMPI&&(bytes[0]||bytes[1])){
            cold(h);const int slot=bytes[1]?1:0;int owner=bytes[slot]?h->myid:h->nprocs,target=0;
            MPI_Allreduce(&owner,&target,1,MPI_INT,MPI_MIN,h->comm);
            const auto target_bytes=bytes[slot];const unsigned ordinal=slot==1&&bytes[0]==target_bytes?2:1;
            trace.arm(h->myid==target?target_bytes:0,ordinal);bool threw=false;
            try{sp::phegvd_prepare_staging(h,bytes);}catch(const std::exception&){threw=true;}
            AllocationProbeSnapshot attribution{};symbol<void(*)(AllocationProbeSnapshot*)>("staging_allocation_probe_end")(&attribution);
            trace.disarm();std::cerr<<"PHEGVD_RAW_PREP_FAULT rank="<<h->myid<<" slot="<<slot<<" target="<<target
                <<" bytes="<<target_bytes<<" ordinal="<<ordinal<<" threw="<<threw<<" fired="<<trace.fired()<<" info="<<info<<std::endl;
            if(!threw||trace.fired()!=unsigned(h->myid==target))trace.dump(h->myid,"phegvd_raw_prep");
            require_all(threw&&trace.fired()==unsigned(h->myid==target)&&(!trace.fired()||trace.failed_bytes()==target_bytes)&&
                (h->myid!=target||attribution.library_with_helper),"PHEGVD complete preparation byte-cache fault was not agreed");
            const auto a=download(h,x.a.ptr,x.initial_a.size()),b=download(h,x.b.ptr,x.initial_b.size()),z=download(h,x.z.ptr,x.initial_z.size());
            bool same=info==-33&&!std::memcmp(a.data(),x.initial_a.data(),a.size()*sizeof(Z))&&
                !std::memcmp(b.data(),x.initial_b.data(),b.size()*sizeof(Z))&&!std::memcmp(z.data(),x.initial_z.data(),z.size()*sizeof(Z));
            for(auto v:x.w)same=same&&v==-313.;require_all(same,"PHEGVD preparation failure touched inputs/output/status");
        }
        // A failed slot1 reservation may have retained slot0; retry still has
        // no warmup solver and the exact same agreed preparation boundary.
        phase_payload(h,trace,bytes,"phegvd",[&]{sp::phegvd_run<GPU,Z>(h,p,eig,r,q,x.a.ptr,x.b.ptr,x.w.data(),x.z.ptr,info,work.data(),work.size(),host.data(),host.size());},!fault_case);
        require_all(info==0,"cold prepared PHEGVD failed");x.oracle();
        if(!h->myid)std::cout<<"PHEGVD_COMPOSITE_PASS route="<<int(eig.route)<<" uplo="<<uplo<<" major="<<h->major
            <<" two_method="<<two_method<<" reduction="<<int(reduction)<<" raw_fault="<<fault_case
            <<" arena="<<q.backend_bytes<<','<<q.host_bytes<<" staging_required="<<bytes[0]<<','<<bytes[1]<<" cache_capacity="<<h->host_buffer_capacities()[0]<<','<<h->host_buffer_capacities()[1]<<'\n';
        prepared.release();
    }
    DDLA_CHECK(ddlaDestroy(h));
}
void phase_queries(int pr,int pc,const Trace& trace,char uplo)
{
    // Distinct fresh handles keep each individual phase's first transport cold;
    // complete-composite maxima cannot conceal an underestimated child query.
    for(int phase=0;phase<3;++phase){
        auto h=create(GPU,pr,pc);
        {
            Problem x(h,37,5,uplo);Prepared p(x,"one_stage");
            ddla::vector<char> work(h,p.q.backend_bytes);std::vector<char> host(p.q.host_bytes);int info=0;
            if(phase==0){
                const auto bytes=spectral_query(h,trace,"ppotrf",[&]{return sp::phegvd_factor_staging_bytes<Z>(h,p.p);});
                phase_payload(h,trace,bytes,"ppotrf",[&]{sp::phegvd_factor_run<GPU,Z>(h,p.p,x.b.ptr,info,work.data(),work.size(),host.data(),host.size());});
            }else if(phase==1){
                // Identity B is already its valid Cholesky factor; no factor warmup.
                const auto bytes=spectral_query(h,trace,"phegvd_congruence",[&]{return hegst_impl::congruence_staging_bytes<Z>(h,p.p.complete,p.p.a.data(),p.p.b.data());});
                phase_payload(h,trace,bytes,"phegvd_congruence",[&]{hegst_impl::congruence_run<GPU,Z>(h,p.p.complete,x.a.ptr,p.p.a.data(),x.b.ptr,p.p.b.data(),work.data(),work.size(),host.data(),host.size());});
            }else{
                const auto bytes=spectral_query(h,trace,"phegvd_back",[&]{return sp::phegvd_back_staging_bytes<Z>(h,p.p);});
                phase_payload(h,trace,bytes,"phegvd_back",[&]{sp::phegvd_back_run<GPU,Z>(h,p.p,x.b.ptr,x.z.ptr,work.data(),work.size(),host.data(),host.size());});
            }
            require_all(info==0,"PHEGVD primitive phase failed");p.release();
        }
        DDLA_CHECK(ddlaDestroy(h));
    }
}
void public_factor_fault(int pr,int pc,const Trace& trace,char uplo)
{
    if(pr*pc==1||transport!=DdlaCommunication::HostStagedMPI)return;
    auto h=create(GPU,pr,pc);
    {
        Problem x(h,37,5,uplo);auto p=x.check();
        const auto bytes=sp::phegvd_factor_staging_bytes<Z>(h,p);cold(h);
        int owner=bytes[0]?h->myid:h->nprocs,target=0;MPI_Allreduce(&owner,&target,1,MPI_INT,MPI_MIN,h->comm);
        require_all(target<h->nprocs,"no PPOTRF staging allocation to fault");
        Method method("invalid-future-method-must-not-be-read");
        trace.arm(h->myid==target?bytes[0]:0,1);bool threw=false;int info=-33;
        try{phegvd<Z>(h,'V',uplo,x.n,x.nev,x.a.ptr,x.da.data(),x.b.ptr,x.db.data(),x.w.data(),x.z.ptr,x.dz.data(),info);}
        catch(const std::exception&){threw=true;}
        AllocationProbeSnapshot attribution{};symbol<void(*)(AllocationProbeSnapshot*)>("staging_allocation_probe_end")(&attribution);
        trace.disarm();trace.dump(h->myid,"phegvd_public_factor");
        const auto a=download(h,x.a.ptr,x.initial_a.size()),b=download(h,x.b.ptr,x.initial_b.size()),z=download(h,x.z.ptr,x.initial_z.size());
        bool same=info==0&&!std::memcmp(a.data(),x.initial_a.data(),a.size()*sizeof(Z))&&
            !std::memcmp(b.data(),x.initial_b.data(),b.size()*sizeof(Z))&&!std::memcmp(z.data(),x.initial_z.data(),z.size()*sizeof(Z));
        for(auto v:x.w)same=same&&v==-313.;
        require_all(threw&&trace.fired()==unsigned(h->myid==target)&&(!trace.fired()||trace.failed_bytes()==bytes[0])&&
            (h->myid!=target||attribution.library_with_helper),"public PHEGVD factor byte-cache fault not fired/agreed");
        require_all(same,"public PHEGVD cold factor failure mutated a destructive phase");
        setenv("DDLA_EIGEN_METHOD","one_stage",1);
        phegvd<Z>(h,'V',uplo,x.n,x.nev,x.a.ptr,x.da.data(),x.b.ptr,x.db.data(),x.w.data(),x.z.ptr,x.dz.data(),info);
        require_all(info==0,"public PHEGVD same-handle allocation recovery failed");x.oracle();
        if(!h->myid)std::cout<<"PHEGVD_PUBLIC_FACTOR_FAULT_PASS uplo="<<uplo<<" all_rank_throw=1 before_B_mutation=1 future_method_unread=1 retry=1\n";
    }
    DDLA_CHECK(ddlaDestroy(h));
}
void public_later_fault(int pr,int pc,const Trace& trace,char uplo,int phase,const char* method)
{
    if(pr*pc==1||transport!=DdlaCommunication::HostStagedMPI)return;
    auto h=create(GPU,pr,pc);
    {
        int n=0,slot=0,target=h->nprocs;std::array<std::size_t,2> wanted{};
        std::array<std::size_t,2> allocation{};std::array<unsigned,2> ordinals{};
        // Find an actually growing cache boundary. If prior phases already
        // cover an eigen stage, no eigen-stage cache allocation can fail.
        for(int candidate=3;candidate<=45&&!n;++candidate){
            Problem probe(h,candidate,std::min(candidate,5),uplo);Prepared p(probe,method);
            const auto congruence=hegst_impl::congruence_staging_bytes<Z>(h,p.p.complete,p.p.a.data(),p.p.b.data());
            wanted=phase==1?congruence:sp::pheevd_driver_staging_bytes<Z>(h,p.eig,*p.resources);
            if(phase==2)ddla::pblas::staging_max(wanted,sp::phegvd_back_staging_bytes<Z>(h,p.p));
            // Match the actual standard byte-vector capacity growth rather
            // than confusing payload bytes with an allocator request. These
            // fixture-only vectors are never queried/observed numerical work.
            const auto factor=sp::phegvd_factor_staging_bytes<Z>(h,p.p);
            const std::array<std::array<std::size_t,2>,3> phases{factor,congruence,wanted};
            const int boundary=phase==1?1:2;
            for(int candidate_slot=0;candidate_slot<2;++candidate_slot){
                std::vector<std::byte> calibration[2];std::array<std::size_t,6> allocations{};unsigned used=0;
                std::size_t target_allocation=0;unsigned target_index=0;
                for(int f=0;f<=boundary;++f)for(int k=0;k<2;++k){
                    const auto before=calibration[k].capacity();
                    if(calibration[k].size()<phases[f][k])calibration[k].resize(phases[f][k]);
                    if(calibration[k].capacity()!=before){
                        allocations[used++]=calibration[k].capacity();
                        if(f==boundary&&k==candidate_slot){target_allocation=calibration[k].capacity();target_index=used;}
                    }
                }
                unsigned ordinal=0;for(unsigned k=0;k<target_index;++k)if(allocations[k]==target_allocation)++ordinal;
                const bool unique=target_allocation&&target_allocation%256&&target_allocation!=p.q.host_bytes;
                int owner=unique?h->myid:h->nprocs;
                MPI_Allreduce(&owner,&target,1,MPI_INT,MPI_MIN,h->comm);
                if(target<h->nprocs){n=candidate;slot=candidate_slot;allocation[candidate_slot]=target_allocation;ordinals[candidate_slot]=ordinal;break;}
            }
            p.release();
        }
        if(!n){
            if(!h->myid)std::cout<<"PHEGVD_PUBLIC_LATE_FAULT_NOT_APPLICABLE phase="<<phase<<" method="<<method<<" no_unique_growth_in_test_shapes=1 n_range=3:45\n";
        }else{
            Problem x(h,n,std::min(n,5),uplo);Prepared p(x,method);cold(h);
            // A separate handle computes the exact retired previous-phase
            // bytes. It never warms the tested handle's transport cache.
            std::vector<Z> expected_a,expected_b;
            auto reference=create(GPU,pr,pc);
            {
                Problem ref(reference,n,x.nev,uplo);Prepared rp(ref,method);
                ddla::vector<char> work(reference,rp.q.backend_bytes);std::vector<char> host(rp.q.host_bytes);int info=0;
                auto before=sp::phegvd_factor_staging_bytes<Z>(reference,rp.p);
                if(phase==2)ddla::pblas::staging_max(before,hegst_impl::congruence_staging_bytes<Z>(reference,rp.p.complete,rp.p.a.data(),rp.p.b.data()));
                sp::phegvd_prepare_staging(reference,before);
                sp::phegvd_factor_run<GPU,Z>(reference,rp.p,ref.b.ptr,info,work.data(),work.size(),host.data(),host.size());
                if(phase==2)hegst_impl::congruence_run<GPU,Z>(reference,rp.p.complete,ref.a.ptr,rp.p.a.data(),ref.b.ptr,rp.p.b.data(),work.data(),work.size(),host.data(),host.size());
                expected_a=download(reference,ref.a.ptr,ref.initial_a.size());expected_b=download(reference,ref.b.ptr,ref.initial_b.size());
                require_all(info==0,"PHEGVD pre-phase reference failed");rp.release();
            }
            DDLA_CHECK(ddlaDestroy(reference));
            const auto bytes=allocation[slot];const unsigned ordinal=ordinals[slot];
            trace.arm(h->myid==target?bytes:0,ordinal);Method policy(method);bool threw=false;int info=-33;
            try{phegvd<Z>(h,'V',uplo,x.n,x.nev,x.a.ptr,x.da.data(),x.b.ptr,x.db.data(),x.w.data(),x.z.ptr,x.dz.data(),info);}
            catch(const std::exception&){threw=true;}
            AllocationProbeSnapshot attribution{};symbol<void(*)(AllocationProbeSnapshot*)>("staging_allocation_probe_end")(&attribution);
            trace.disarm();trace.dump(h->myid,phase==1?"phegvd_public_congruence":"phegvd_public_eigen");
            std::cerr<<"PHEGVD_PUBLIC_LATE_FAULT phase="<<phase<<" method="<<method<<" rank="<<h->myid<<" target="<<target
                <<" slot="<<slot<<" bytes="<<bytes<<" ordinal="<<ordinal<<" fired="<<trace.fired()<<" threw="<<threw<<" info="<<info<<std::endl;
            const auto a=download(h,x.a.ptr,x.initial_a.size()),b=download(h,x.b.ptr,x.initial_b.size()),z=download(h,x.z.ptr,x.initial_z.size());
            bool same=info==0&&!std::memcmp(a.data(),expected_a.data(),a.size()*sizeof(Z))&&
                !std::memcmp(b.data(),expected_b.data(),b.size()*sizeof(Z))&&!std::memcmp(z.data(),x.initial_z.data(),z.size()*sizeof(Z));
            for(auto v:x.w)same=same&&v==-313.;
            require_all(threw&&trace.fired()==unsigned(h->myid==target)&&(!trace.fired()||trace.failed_bytes()==bytes)&&
                (h->myid!=target||attribution.library_with_helper),"public PHEGVD late byte-cache fault not fired/agreed");
            require_all(same,"public PHEGVD late preparation failure crossed a destructive phase");
            x.reset();phegvd<Z>(h,'V',uplo,x.n,x.nev,x.a.ptr,x.da.data(),x.b.ptr,x.db.data(),x.w.data(),x.z.ptr,x.dz.data(),info);
            require_all(info==0,"public PHEGVD late same-handle retry failed");x.oracle();
            if(!h->myid)std::cout<<"PHEGVD_PUBLIC_LATE_FAULT_PASS phase="<<phase<<" method="<<method<<" uplo="<<uplo
                <<" before_phase_mutation=1 output_unchanged=1 retry=1\n";
            p.release();
        }
    }
    DDLA_CHECK(ddlaDestroy(h));
}
void run(int pr,int pc,const Trace& trace)
{
    for(char uplo:{'L','U'}){
        phase_queries(pr,pc,trace,uplo);
        for(const char* method:{"auto","one_stage","two_stage"})raw(pr,pc,trace,method,uplo,true);
        // Exercise the nondefault prepared TwoStage engine geometries too;
        // these calls do not alter the public method/default policy.
        raw(pr,pc,trace,"two_stage",uplo,false,'H',DdlaBackend::CPU);
        raw(pr,pc,trace,"two_stage",uplo,false,'G',DdlaBackend::GPU);
        raw(pr,pc,trace,"two_stage",uplo,true,'G',DdlaBackend::GPU);
        public_factor_fault(pr,pc,trace,uplo);
        public_later_fault(pr,pc,trace,uplo,1,"one_stage");
        for(const char* method:{"one_stage","two_stage"})public_later_fault(pr,pc,trace,uplo,2,method);
    }
}
} // namespace composite_staging
#endif
