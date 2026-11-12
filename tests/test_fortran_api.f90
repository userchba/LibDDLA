! ---------------------------------------------------------------------------
! test_fortran_api.f90 -- MPI test of the LibDDLA Fortran bindings (module
! ddla).  Runs identically on a CPU handle and (through ddla_malloc device
! buffers + ddla_memcpy) on an explicitly selected GPU handle.
!
! What it exercises, in order:
!   1. handle lifecycle (init / set with the auto process grid / grid queries
!      / destroy) and the ScaLAPACK-layout ddla_descinit + ddla_numloc pair;
!   2. ddla_malloc / ddla_memcpy / ddla_memcpy2d / ddla_memset /
!      ddla_synchronize round trips through backend memory;
!   3. double pgesv: X_exact filled per local block-cyclic piece, B := A X by
!      ddla_d_pgemm, solve, residual ||A X - B|| / ||B|| via a second pgemm;
!   4. double ppotrf + ddla_d_pdam + ppotrs: A = M M^T + shift built with a
!      transposed pgemm, factor (info must be 0), solve, residual;
!   5. complex pgesv (ddla_z_pgemm / ddla_z_pgesv): the ddla_complex_double
!      scalar marshalling and complex device buffers;
!   6. ddla_d_pgeqrf smoke and a QL-specific ddla_d_pgeqlf output check.
!
! Pass criterion: every check green on every rank; the binary prints
! "test_fortran_api passed" on rank 0 and exits 0.
! ---------------------------------------------------------------------------
program test_fortran_api
  use iso_c_binding
  use ddla
  implicit none
  include 'mpif.h'

  type(c_ptr) :: h
  integer(c_int) :: st, be
  integer(c_int) :: desca(9), descb(9), descx(9), descm(9)
  integer :: provided
  integer :: ierr, me, nproc
  integer(c_int) :: nprow, npcol, mypr, mypc
  integer(c_int) :: n, nrhs, mb, nb
  integer(c_int) :: mloc, ncola, bcols, llda, lldb
  real(c_double) :: one, zero, mone, shift
  type(ddla_complex_double) :: zone, zzero, zmone
  real(c_double), target, allocatable :: ha(:,:), ha0(:,:), hb(:,:), hx(:,:), hr(:,:)
  real(c_double), target, allocatable :: hm(:,:), ha2(:,:), ha20(:,:), hb2(:,:), hx2(:,:), hr2(:,:)
  real(c_double), target, allocatable :: tau(:), h1d(:), h1d2(:), hpat(:), hback(:)
  complex(c_double), target, allocatable :: za(:,:), za0(:,:), zb(:,:), zx(:,:), zr(:,:)
  type(c_ptr) :: pa, pa0, pb, pb0, px, pr, pm, pa2, pa20, pb2, pb20, px2, pr2
  type(c_ptr) :: za_p, za0_p, zb_p, zb0_p, zx_p, zr_p, pbuf, pbuf2
  real(c_double) :: resid2, bnorm2, rel, tval, dval
  real(c_double) :: rsum(2), rmax
  integer(c_int) :: info, i, j, nfail, global_row, global_col
  integer(c_int64_t) :: nbytes, seed
  real(c_double), parameter :: TOL = 1.0d-8

  nfail = 0
  call MPI_Init_thread(MPI_THREAD_FUNNELED, provided, ierr)
  if (ierr /= MPI_SUCCESS) stop 1
  if (provided < MPI_THREAD_FUNNELED) call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  call mpi_comm_rank(MPI_COMM_WORLD, me, ierr)
  call mpi_comm_size(MPI_COMM_WORLD, nproc, ierr)

  zone  = ddla_complex_double(1.0_c_double, 0.0_c_double)
  zzero = ddla_complex_double(0.0_c_double, 0.0_c_double)
  zmone = ddla_complex_double(-1.0_c_double, 0.0_c_double)
  one = 1.0_c_double; zero = 0.0_c_double; mone = -1.0_c_double

  ! ---- 1. lifecycle + grid + descriptor -----------------------------------
  be = DDLA_BACKEND_CPU
  if (ddla_backend_available(DDLA_BACKEND_GPU) /= 0) be = DDLA_BACKEND_GPU
  st = ddla_init_backend(h, be)
  call ddla_check(st, 'ddla_init_backend')
  st = ddla_set(h, int(MPI_COMM_WORLD, c_int), 0, 0, 'R')
  call ddla_check(st, 'ddla_set')
  st = ddla_grid_dims(h, nprow, npcol)
  call ddla_check(st, 'ddla_grid_dims')
  st = ddla_grid_coords(h, mypr, mypc)
  call ddla_check(st, 'ddla_grid_coords')
  if (nprow * npcol /= nproc) then
     write (*, '(A,I0,A,I0,A,I0)') 'rank ', me, ': grid ', nprow, 'x', npcol
     nfail = nfail + 1
  end if
  if (ddla_rank(h) /= me) then
     write (*, '(A,I0)') 'rank ', me, ': ddla_rank mismatch'; flush(6)
     nfail = nfail + 1
  end if
  if (ddla_size(h) /= nproc) then
     write (*, '(A,I0)') 'rank ', me, ': ddla_size mismatch'; flush(6)
     nfail = nfail + 1
  end if
  be = ddla_get_backend(h)
  write (*, '(A,I0)') 'sec1 done r', me; flush(6)
  if (be /= DDLA_BACKEND_CPU .and. be /= DDLA_BACKEND_GPU) then
     write (*, '(A,I0)') 'rank ', me, ': backend id mismatch'; flush(6)
     nfail = nfail + 1
  end if

  n = 333_c_int; nrhs = 11_c_int; mb = 64_c_int; nb = 64_c_int
  st = ddla_descinit(desca, h, n, n, mb, nb, 0, 0, -1)
  call ddla_check(st, 'ddla_descinit A')
  st = ddla_descinit(descb, h, n, nrhs, mb, nb, 0, 0, -1)
  call ddla_check(st, 'ddla_descinit B')
  descx = descb; descm = desca
  if (desca(1) /= 1_c_int) then
     write (*, '(A,I0)') 'rank ', me, ': desc dtype mismatch'; flush(6)
     nfail = nfail + 1
  end if

  mloc  = ddla_numloc(n, mb, mypr, 0, nprow)     ! local rows
  ncola = ddla_numloc(n, nb, mypc, 0, npcol)     ! local columns of A
  bcols = ddla_numloc(nrhs, nb, mypc, 0, npcol)  ! local columns of B/X
  llda = desca(9); lldb = descb(9)
  if (llda < mloc .or. lldb < mloc) then
     write (*, '(A,I0)') 'rank ', me, ': lld too small'; flush(6)
     nfail = nfail + 1
  end if

  ! ---- 2. memory round trips ----------------------------------------------
  nbytes = int(8, c_int64_t) * 1024_c_int64_t
  st = ddla_malloc(h, pbuf, nbytes)
  call ddla_check(st, 'ddla_malloc')
  st = ddla_memset(h, pbuf, 0, nbytes)
  call ddla_check(st, 'ddla_memset')
  allocate (hpat(1024), hback(1024))
  do i = 1, 1024
     hpat(i) = dble(mod(i * 37 + 11, 251)) - 125.0_c_double
  end do
  st = ddla_memcpy(h, pbuf, c_loc(hpat(1)), nbytes)
  call ddla_check(st, 'ddla_memcpy h2b')
  st = ddla_memcpy(h, c_loc(hback(1)), pbuf, nbytes)
  call ddla_check(st, 'ddla_memcpy b2h')
  st = ddla_synchronize(h)
  call ddla_check(st, 'ddla_synchronize')
  do i = 1, 1024
     if (hback(i) /= hpat(i)) then
        write (*, '(A,I0,A)') 'rank ', me, ': memcpy round trip differs'; flush(6)
        nfail = nfail + 1
        exit
     end if
  end do
  ! pitched copy of the first two rows (spitch = dpitch = 256 bytes)
  allocate (h1d(32 * 32), h1d2(64))
  h1d = 0.0_c_double
  do i = 1, 16
     do j = 1, 32
        h1d(i + (j - 1) * 32) = dble(i * 100 + j)
     end do
  end do
  st = ddla_malloc(h, pbuf2, int(8, c_int64_t) * 32_c_int64_t * 16_c_int64_t)
  call ddla_check(st, 'ddla_malloc 2d')
  st = ddla_memcpy(h, pbuf2, c_loc(h1d(1)), int(8, c_int64_t) * 32_c_int64_t * 16_c_int64_t)
  call ddla_check(st, 'ddla_memcpy 2d src')
  st = ddla_memcpy2d(h, c_loc(h1d2(1)), int(8, c_int64_t) * 32_c_int64_t, &
                     pbuf2, int(8, c_int64_t) * 32_c_int64_t, &
                     int(8, c_int64_t) * 32_c_int64_t, 2_c_int64_t)
  call ddla_check(st, 'ddla_memcpy2d')
  st = ddla_synchronize(h)
  call ddla_check(st, 'ddla_synchronize 2d')
  ! the fill is column-major: memcpy2d "row" k holds fill column k+1, i.e.
  ! dble(i*100 + (k+1)) for rows i = 1..16 and zero padding beyond
  do i = 1, 16
     if (h1d2(i) /= dble(i * 100 + 1) .or. h1d2(32 + i) /= dble(i * 100 + 2)) then
        write (*, '(A,I0,A)') 'rank ', me, ': memcpy2d result differs'; flush(6)
        nfail = nfail + 1
        exit
     end if
  end do
  st = ddla_free(h, pbuf)
  call ddla_check(st, 'ddla_free')
  st = ddla_free(h, pbuf2)
  call ddla_check(st, 'ddla_free 2d')
  write (*, '(A,I0)') 'sec2 done r', me; flush(6)

  ! ---- 3. double pgesv end to end -----------------------------------------
  allocate (ha(llda, max(ncola, 1)), ha0(llda, max(ncola, 1)), &
            hb(lldb, max(bcols, 1)), hx(lldb, max(bcols, 1)), hr(lldb, max(bcols, 1)))
  seed = 1000 + me
  do j = 1, ncola
     do i = 1, mloc
        ha(i, j) = rnd(seed) - 0.5_c_double
        if (l2g(i, mb, mypr, 0, nprow) == l2g(j, nb, mypc, 0, npcol)) &
           ha(i, j) = ha(i, j) + dble(n)          ! diagonally dominant
     end do
  end do
  do j = 1, bcols
     do i = 1, mloc
        hx(i, j) = rnd(seed) - 0.5_c_double
     end do
  end do
  st = ddla_malloc(h, pa, bytes_d(llda, ncola))
  call ddla_check(st, 'malloc A')
  st = ddla_malloc(h, pa0, bytes_d(llda, ncola))
  call ddla_check(st, 'malloc A0')
  st = ddla_malloc(h, pb, bytes_d(lldb, bcols))
  call ddla_check(st, 'malloc B')
  st = ddla_malloc(h, px, bytes_d(lldb, bcols))
  call ddla_check(st, 'malloc X')
  st = ddla_malloc(h, pb0, bytes_d(lldb, bcols))
  call ddla_check(st, 'malloc B0')
  st = ddla_malloc(h, pr, bytes_d(lldb, bcols))
  call ddla_check(st, 'malloc R')
  st = ddla_memcpy(h, pa, c_loc(ha(1, 1)), bytes_d(llda, ncola))
  call ddla_check(st, 'upload A')
  st = ddla_memcpy(h, px, c_loc(hx(1, 1)), bytes_d(lldb, bcols))
  call ddla_check(st, 'upload X')
  ! B := 1 * A * X + 0 * B
  st = ddla_d_pgemm(h, 'N', 'N', n, nrhs, n, one, pa, desca, px, descx, zero, pb, descb)
  call ddla_check(st, 'pgemm B=AX')
  st = ddla_synchronize(h)
  call ddla_check(st, 'sync B=AX')
  st = ddla_memcpy(h, c_loc(hb(1, 1)), pb, bytes_d(lldb, bcols))  ! keep ||B||
  call ddla_check(st, 'download B')
  st = ddla_memcpy(h, pb0, pb, bytes_d(lldb, bcols))              ! B0 = B
  call ddla_check(st, 'copy B->B0')
  st = ddla_memcpy(h, pa0, pa, bytes_d(llda, ncola))              ! A0 = A
  call ddla_check(st, 'copy A->A0')

  st = ddla_d_pgesv(h, 'L', 'N', n, nrhs, pa, desca, pb, descb)
  call ddla_check(st, 'pgesv')

  ! R := A0 * X - B, residual relative to B (pb holds X after pgesv)
  st = ddla_memcpy(h, pr, pb0, bytes_d(lldb, bcols))
  call ddla_check(st, 'copy B0->R')
  st = ddla_d_pgemm(h, 'N', 'N', n, nrhs, n, one, pa0, desca, pb, descb, mone, pr, descb)
  call ddla_check(st, 'pgemm residual')
  st = ddla_memcpy(h, c_loc(hr(1, 1)), pr, bytes_d(lldb, bcols))
  call ddla_check(st, 'download R')
  st = ddla_synchronize(h)
  call ddla_check(st, 'sync residual')
  rsum = 0.0_c_double
  do j = 1, bcols
     do i = 1, mloc
        rsum(1) = rsum(1) + hr(i, j) * hr(i, j)
        rsum(2) = rsum(2) + hb(i, j) * hb(i, j)
     end do
  end do
  call mpi_allreduce(rsum(1), resid2, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
  call mpi_allreduce(rsum(2), bnorm2, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
  rel = sqrt(resid2 / max(bnorm2, 1.0d-300))
  if (rel > TOL) then
     write (*, '(A,I0,A,ES9.2)') 'rank ', me, ': d pgesv relative residual ', rel
     nfail = nfail + 1
  end if

  write (*, '(A,I0,A,ES9.2)') 'sec3 done r', me, ' rel ', rel; flush(6)
  ! ---- 4. double ppotrf + pdam + ppotrs -----------------------------------
  allocate (hm(llda, max(ncola, 1)), ha2(llda, max(ncola, 1)), ha20(llda, max(ncola, 1)), &
            hb2(lldb, max(bcols, 1)), hx2(lldb, max(bcols, 1)), hr2(lldb, max(bcols, 1)))
  seed = 7000 + me
  do j = 1, ncola
     do i = 1, mloc
        hm(i, j) = rnd(seed) - 0.5_c_double
     end do
  end do
  do j = 1, bcols
     do i = 1, mloc
        hx2(i, j) = rnd(seed) - 0.5_c_double
     end do
  end do
  st = ddla_malloc(h, pm, bytes_d(llda, ncola))
  call ddla_check(st, 'malloc M')
  st = ddla_malloc(h, pa2, bytes_d(llda, ncola))
  call ddla_check(st, 'malloc A2')
  st = ddla_malloc(h, pa20, bytes_d(llda, ncola))
  call ddla_check(st, 'malloc A20')
  st = ddla_malloc(h, pb2, bytes_d(lldb, bcols))
  call ddla_check(st, 'malloc B2')
  st = ddla_malloc(h, px2, bytes_d(lldb, bcols))
  call ddla_check(st, 'malloc X2')
  st = ddla_malloc(h, pb20, bytes_d(lldb, bcols))
  call ddla_check(st, 'malloc B20')
  st = ddla_malloc(h, pr2, bytes_d(lldb, bcols))
  call ddla_check(st, 'malloc R2')
  st = ddla_memcpy(h, pm, c_loc(hm(1, 1)), bytes_d(llda, ncola))
  call ddla_check(st, 'upload M')
  st = ddla_memcpy(h, px2, c_loc(hx2(1, 1)), bytes_d(lldb, bcols))
  call ddla_check(st, 'upload X2')
  ! A2 := M M^T  (both operands the same buffer, read-only)
  st = ddla_d_pgemm(h, 'N', 'T', n, n, n, one, pm, descm, pm, descm, zero, pa2, desca)
  call ddla_check(st, 'pgemm A2=MM^T')
  shift = dble(n) + 5.0_c_double
  st = ddla_d_pdam(h, shift, pa2, desca, n)
  call ddla_check(st, 'pdam diag shift')
  st = ddla_memcpy(h, pa20, pa2, bytes_d(llda, ncola))
  call ddla_check(st, 'copy A2->A20')

  info = -1_c_int
  st = ddla_d_ppotrf(h, 'L', n, pa2, 1_c_int, 1_c_int, desca, info)
  call ddla_check(st, 'ppotrf')
  if (info /= 0_c_int) then
     write (*, '(A,I0,A,I0)') 'rank ', me, ': d ppotrf info ', info
     nfail = nfail + 1
  end if

  st = ddla_d_pgemm(h, 'N', 'N', n, nrhs, n, one, pa20, desca, px2, descx, zero, pb2, descb)
  call ddla_check(st, 'pgemm B2=A2X')
  st = ddla_memcpy(h, pb20, pb2, bytes_d(lldb, bcols))
  call ddla_check(st, 'copy B2->B20')
  st = ddla_d_ppotrs(h, 'L', 'L', 'N', n, nrhs, pa2, desca, pb2, descb)
  call ddla_check(st, 'ppotrs')
  st = ddla_memcpy(h, pr2, pb20, bytes_d(lldb, bcols))
  call ddla_check(st, 'copy B20->R2')
  st = ddla_d_pgemm(h, 'N', 'N', n, nrhs, n, one, pa20, desca, pb2, descb, mone, pr2, descb)
  call ddla_check(st, 'pgemm residual 2')
  st = ddla_memcpy(h, c_loc(hr2(1, 1)), pr2, bytes_d(lldb, bcols))
  call ddla_check(st, 'download R2')
  st = ddla_memcpy(h, c_loc(hb2(1, 1)), pb2, bytes_d(lldb, bcols))
  call ddla_check(st, 'download B2')
  st = ddla_synchronize(h)
  call ddla_check(st, 'sync residual 2')
  rsum = 0.0_c_double
  do j = 1, bcols
     do i = 1, mloc
        rsum(1) = rsum(1) + hr2(i, j) * hr2(i, j)
        rsum(2) = rsum(2) + hb2(i, j) * hb2(i, j)
     end do
  end do
  call mpi_allreduce(rsum(1), resid2, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
  call mpi_allreduce(rsum(2), bnorm2, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
  rel = sqrt(resid2 / max(bnorm2, 1.0d-300))
  if (rel > TOL) then
     write (*, '(A,I0,A,ES9.2)') 'rank ', me, ': d ppotrf/ppotrs relative residual ', rel
     nfail = nfail + 1
  end if

  write (*, '(A,I0,A,ES9.2)') 'sec4 done r', me, ' rel ', rel; flush(6)
  ! ---- 5. complex pgesv ----------------------------------------------------
  allocate (za(llda, max(ncola, 1)), za0(llda, max(ncola, 1)), &
            zb(lldb, max(bcols, 1)), zx(lldb, max(bcols, 1)), zr(lldb, max(bcols, 1)))
  seed = 13000 + me
  do j = 1, ncola
     do i = 1, mloc
        za(i, j) = cmplx(rnd(seed) - 0.5_c_double, rnd(seed) - 0.5_c_double, c_double)
        if (l2g(i, mb, mypr, 0, nprow) == l2g(j, nb, mypc, 0, npcol)) &
           za(i, j) = za(i, j) + dble(n)
     end do
  end do
  do j = 1, bcols
     do i = 1, mloc
        zx(i, j) = cmplx(rnd(seed) - 0.5_c_double, rnd(seed) - 0.5_c_double, c_double)
     end do
  end do
  st = ddla_malloc(h, za_p, bytes_z(llda, ncola))
  call ddla_check(st, 'malloc zA')
  st = ddla_malloc(h, za0_p, bytes_z(llda, ncola))
  call ddla_check(st, 'malloc zA0')
  st = ddla_malloc(h, zb_p, bytes_z(lldb, bcols))
  call ddla_check(st, 'malloc zB')
  st = ddla_malloc(h, zx_p, bytes_z(lldb, bcols))
  call ddla_check(st, 'malloc zX')
  st = ddla_malloc(h, zb0_p, bytes_z(lldb, bcols))
  call ddla_check(st, 'malloc zB0')
  st = ddla_malloc(h, zr_p, bytes_z(lldb, bcols))
  call ddla_check(st, 'malloc zR')
  st = ddla_memcpy(h, za_p, c_loc(za(1, 1)), bytes_z(llda, ncola))
  call ddla_check(st, 'upload zA')
  st = ddla_memcpy(h, zx_p, c_loc(zx(1, 1)), bytes_z(lldb, bcols))
  call ddla_check(st, 'upload zX')
  st = ddla_z_pgemm(h, 'N', 'N', n, nrhs, n, zone, za_p, desca, zx_p, descx, zzero, zb_p, descb)
  call ddla_check(st, 'z pgemm')
  st = ddla_memcpy(h, c_loc(zb(1, 1)), zb_p, bytes_z(lldb, bcols))
  call ddla_check(st, 'download zB')
  st = ddla_memcpy(h, zb0_p, zb_p, bytes_z(lldb, bcols))
  call ddla_check(st, 'copy zB->zB0')
  st = ddla_memcpy(h, za0_p, za_p, bytes_z(llda, ncola))
  call ddla_check(st, 'copy zA->zA0')
  st = ddla_z_pgesv(h, 'L', 'N', n, nrhs, za_p, desca, zb_p, descb)
  call ddla_check(st, 'z pgesv')
  st = ddla_memcpy(h, zr_p, zb0_p, bytes_z(lldb, bcols))
  call ddla_check(st, 'copy zB0->zR')
  st = ddla_z_pgemm(h, 'N', 'N', n, nrhs, n, zone, za0_p, desca, zb_p, descb, zmone, zr_p, descb)
  call ddla_check(st, 'z pgemm residual')
  st = ddla_memcpy(h, c_loc(zr(1, 1)), zr_p, bytes_z(lldb, bcols))
  call ddla_check(st, 'download zR')
  st = ddla_synchronize(h)
  call ddla_check(st, 'sync z residual')
  rsum = 0.0_c_double
  do j = 1, bcols
     do i = 1, mloc
        rsum(1) = rsum(1) + real(zr(i, j))**2 + aimag(zr(i, j))**2
        rsum(2) = rsum(2) + real(zb(i, j))**2 + aimag(zb(i, j))**2
     end do
  end do
  call mpi_allreduce(rsum(1), resid2, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
  call mpi_allreduce(rsum(2), bnorm2, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
  rel = sqrt(resid2 / max(bnorm2, 1.0d-300))
  if (rel > TOL) then
     write (*, '(A,I0,A,ES9.2)') 'rank ', me, ': z pgesv relative residual ', rel
     nfail = nfail + 1
  end if

  write (*, '(A,I0,A,ES9.2)') 'sec5 done r', me, ' rel ', rel; flush(6)
  ! ---- 6. pgeqrf smoke -----------------------------------------------------
  seed = 19000 + me
  do j = 1, ncola
     do i = 1, mloc
        ha(i, j) = rnd(seed) - 0.5_c_double
     end do
  end do
  st = ddla_memcpy(h, pa, c_loc(ha(1, 1)), bytes_d(llda, ncola))
  call ddla_check(st, 'upload A (qr)')
  allocate (tau(max(ncola + 8, 1)))
  tau = 0.0_c_double
  info = -1_c_int
  st = ddla_d_pgeqrf(h, n, n, pa, desca, tau, info)
  call ddla_check(st, 'pgeqrf')
  if (info /= 0_c_int) then
     write (*, '(A,I0,A,I0)') 'rank ', me, ': d pgeqrf info ', info
     nfail = nfail + 1
  end if
  tval = 0.0_c_double
  do i = 1, ncola + 8
     tval = max(tval, abs(tau(i)))
  end do
  call mpi_allreduce(tval, rmax, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
  if (rmax > 2.0_c_double .or. rmax /= rmax) then
     write (*, '(A,I0,A,ES9.2)') 'rank ', me, ': d pgeqrf |tau| out of range ', rmax
     nfail = nfail + 1
  end if

  ! QL of [1,2,3]^T stores -sqrt(14) at the final row of L.
  ! This also distinguishes the QL C entry from QR.
  ha = 0.0_c_double
  do j = 1, ncola
     global_col = ((j-1)/nb*npcol + mypc)*nb + mod(j-1, nb) + 1
     do i = 1, mloc
        global_row = ((i-1)/mb*nprow + mypr)*mb + mod(i-1, mb) + 1
        if (global_col == 1 .and. global_row <= 3) ha(i,j) = real(global_row, c_double)
     end do
  end do
  st = ddla_memcpy(h, pa, c_loc(ha(1,1)), bytes_d(llda, ncola))
  call ddla_check(st, 'upload A (ql)')
  tau = 0.0_c_double
  st = ddla_d_pgeqlf(h, 3, 1, pa, desca, tau, info)
  call ddla_check(st, 'pgeqlf')
  if (info /= 0_c_int) nfail = nfail + 1
  st = ddla_memcpy(h, c_loc(ha(1,1)), pa, bytes_d(llda, ncola))
  call ddla_check(st, 'download A (ql)')
  st = ddla_synchronize(h)
  call ddla_check(st, 'sync ql')
  do j = 1, ncola
     global_col = ((j-1)/nb*npcol + mypc)*nb + mod(j-1, nb) + 1
     do i = 1, mloc
        global_row = ((i-1)/mb*nprow + mypr)*mb + mod(i-1, mb) + 1
        if (global_col == 1 .and. global_row == 3) then
           if (.not. (abs(ha(i,j) + sqrt(14.0_c_double)) < TOL)) nfail = nfail + 1
        end if
     end do
  end do

  write (*, '(A,I0)') 'sec6 done r', me; flush(6)
  ! ---- teardown ------------------------------------------------------------
  st = ddla_free(h, pa); call ddla_check(st, 'free A')
  st = ddla_free(h, pa0); call ddla_check(st, 'free A0')
  st = ddla_free(h, pb); call ddla_check(st, 'free B')
  st = ddla_free(h, pb0); call ddla_check(st, 'free B0')
  st = ddla_free(h, px); call ddla_check(st, 'free X')
  st = ddla_free(h, pr); call ddla_check(st, 'free R')
  st = ddla_free(h, pm); call ddla_check(st, 'free M')
  st = ddla_free(h, pa2); call ddla_check(st, 'free A2')
  st = ddla_free(h, pa20); call ddla_check(st, 'free A20')
  st = ddla_free(h, pb2); call ddla_check(st, 'free B2')
  st = ddla_free(h, pb20); call ddla_check(st, 'free B20')
  st = ddla_free(h, px2); call ddla_check(st, 'free X2')
  st = ddla_free(h, pr2); call ddla_check(st, 'free R2')
  st = ddla_free(h, za_p); call ddla_check(st, 'free zA')
  st = ddla_free(h, za0_p); call ddla_check(st, 'free zA0')
  st = ddla_free(h, zb_p); call ddla_check(st, 'free zB')
  st = ddla_free(h, zb0_p); call ddla_check(st, 'free zB0')
  st = ddla_free(h, zx_p); call ddla_check(st, 'free zX')
  st = ddla_free(h, zr_p); call ddla_check(st, 'free zR')
  st = ddla_destroy(h)
  call ddla_check(st, 'ddla_destroy')
  if (c_associated(h)) then
     write (*, '(A,I0)') 'rank ', me, ': handle still associated after destroy'
     nfail = nfail + 1
  end if

  call mpi_allreduce(nfail, i, 1, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD, ierr)
  if (me == 0) then
     if (i == 0) then
        write (*, '(A)') 'test_fortran_api passed'
     else
        write (*, '(A,I0,A)') 'test_fortran_api FAILED with ', i, ' failing checks'
     end if
  end if
  call mpi_finalize(ierr)
  if (i /= 0) error stop 1

contains

  ! deterministic uniform [0,1) from a small integer LCG (per rank)
  function rnd(s) result(v)
    integer(c_int64_t), intent(inout) :: s
    real(c_double) :: v
    s = mod(s, 1000000007_c_int64_t) * 16807_c_int64_t + 7_c_int64_t
    v = dble(abs(mod(s, 1000000007_c_int64_t))) / 1000000007.0_c_double
  end function rnd

  ! 1-based global index of 1-based local index il (ScaLAPACK INDXL2G, the
  ! formula of ddla_desc.h)
  function l2g(il, blk, iproc, isrc, nprocs) result(g)
    integer(c_int), intent(in) :: il, blk, iproc, isrc, nprocs
    integer(c_int) :: g
    g = nprocs * blk * ((il - 1) / blk) + mod(il - 1, blk) &
        + mod(nprocs + iproc - isrc, nprocs) * blk + 1
  end function l2g

  function bytes_d(ld, cols) result(b)
    integer(c_int), intent(in) :: ld, cols
    integer(c_int64_t) :: b
    b = int(8, c_int64_t) * int(ld, c_int64_t) * int(max(cols, 1), c_int64_t)
  end function bytes_d

  function bytes_z(ld, cols) result(b)
    integer(c_int), intent(in) :: ld, cols
    integer(c_int64_t) :: b
    b = int(16, c_int64_t) * int(ld, c_int64_t) * int(max(cols, 1), c_int64_t)
  end function bytes_z

end program test_fortran_api
