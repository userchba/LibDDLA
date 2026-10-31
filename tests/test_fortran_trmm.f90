! In-place local TRMM through each typed C/Fortran binding, on every available backend.
program test_fortran_trmm
  use iso_c_binding
  use ieee_arithmetic
  use ddla
  implicit none
  include 'mpif.h'
  type(c_ptr) :: h
  integer(c_int) :: backend, st
  integer :: ierr, rank, failures, total
  failures = 0
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  do backend = DDLA_BACKEND_CPU, DDLA_BACKEND_GPU
    if (ddla_backend_available(backend) == 0) cycle
    st = ddla_init_backend(h, backend)
    call check(st)
    st = ddla_set(h, int(MPI_COMM_WORLD,c_int), 0, 0, 'R')
    call check(st)
    call check_s(h)
    call check_d(h)
    call check_c(h)
    call check_z(h)
    st = ddla_destroy(h)
    call check(st)
  end do
  call MPI_Allreduce(failures,total,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,ierr)
  if (rank == 0 .and. total == 0) print *, 'test_fortran_trmm passed'
  if (total /= 0) call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  call MPI_Finalize(ierr)
contains
  subroutine check(status)
    integer(c_int), intent(in) :: status
    if (status /= DDLA_STATUS_SUCCESS) then
      print *, 'Fortran TRMM binding status:', status
      call MPI_Abort(MPI_COMM_WORLD,1,ierr)
    end if
  end subroutine
  subroutine check_s(handle)
    type(c_ptr), intent(in) :: handle
    type(c_ptr) :: pa, pb
    real(c_float), target :: a(5,3), b(5,2)
    real(c_float) :: want(3,2)
    real(c_float) :: alpha, zero
    real(c_float) :: base(3,2), poison
    integer(c_int) :: status
    integer :: i, j
    poison = ieee_value(0.0_c_float,ieee_quiet_nan)
    a = poison
    a(1,1)=2; a(1,2)=1; a(2,2)=3
    a(1,3)=-1; a(2,3)=2; a(3,3)=4
    b = -3.25
    do j=1,2
      do i=1,3
        b(i,j) = real(i+3*(j-1),c_float)
      end do
    end do
    base(:,1) = [1,12,12]
    base(:,2) = [7,27,24]
    want = 2.0_c_float*base
    alpha = 2.0_c_float
    zero = 0.0_c_float
    status=ddla_malloc(handle,pa,int(15*4,c_int64_t)); call check(status)
    status=ddla_malloc(handle,pb,int(10*4,c_int64_t)); call check(status)
    status=ddla_memcpy(handle,pa,c_loc(a),int(15*4,c_int64_t)); call check(status)
    status=ddla_memcpy(handle,pb,c_loc(b),int(10*4,c_int64_t)); call check(status)
    status=ddla_s_trmm(handle,'L','U','N','N',3,2,alpha,pa,5,pb,5); call check(status)
    status=ddla_memcpy(handle,c_loc(b),pb,int(10*4,c_int64_t)); call check(status)
    status=ddla_synchronize(handle); call check(status)
    if (any(.not.ieee_is_finite(real(b(1:3,:))))) failures=failures+1
    if (maxval(abs(b(1:3,:)-want)) > 1.0e-5) failures=failures+1
    if (any(b(4:5,:) /= -3.25)) failures=failures+1
    b(1:3,:)=poison
    status=ddla_memcpy(handle,pb,c_loc(b),int(10*4,c_int64_t)); call check(status)
    status=ddla_s_trmm(handle,'L','U','N','N',3,2,zero,c_null_ptr,5,pb,5); call check(status)
    status=ddla_memcpy(handle,c_loc(b),pb,int(10*4,c_int64_t)); call check(status)
    status=ddla_synchronize(handle); call check(status)
    if (any(b(1:3,:) /= 0)) failures=failures+1
    if (any(b(4:5,:) /= -3.25)) failures=failures+1
    status=ddla_free(handle,pa); call check(status)
    status=ddla_free(handle,pb); call check(status)
  end subroutine check_s
  subroutine check_d(handle)
    type(c_ptr), intent(in) :: handle
    type(c_ptr) :: pa, pb
    real(c_double), target :: a(5,3), b(5,2)
    real(c_double) :: want(3,2)
    real(c_double) :: alpha, zero
    real(c_double) :: base(3,2), poison
    integer(c_int) :: status
    integer :: i, j
    poison = ieee_value(0.0_c_double,ieee_quiet_nan)
    a = poison
    a(1,1)=2; a(1,2)=1; a(2,2)=3
    a(1,3)=-1; a(2,3)=2; a(3,3)=4
    b = -3.25
    do j=1,2
      do i=1,3
        b(i,j) = real(i+3*(j-1),c_double)
      end do
    end do
    base(:,1) = [1,12,12]
    base(:,2) = [7,27,24]
    want = 2.0_c_double*base
    alpha = 2.0_c_double
    zero = 0.0_c_double
    status=ddla_malloc(handle,pa,int(15*8,c_int64_t)); call check(status)
    status=ddla_malloc(handle,pb,int(10*8,c_int64_t)); call check(status)
    status=ddla_memcpy(handle,pa,c_loc(a),int(15*8,c_int64_t)); call check(status)
    status=ddla_memcpy(handle,pb,c_loc(b),int(10*8,c_int64_t)); call check(status)
    status=ddla_d_trmm(handle,'L','U','N','N',3,2,alpha,pa,5,pb,5); call check(status)
    status=ddla_memcpy(handle,c_loc(b),pb,int(10*8,c_int64_t)); call check(status)
    status=ddla_synchronize(handle); call check(status)
    if (any(.not.ieee_is_finite(real(b(1:3,:))))) failures=failures+1
    if (maxval(abs(b(1:3,:)-want)) > 1.0e-12) failures=failures+1
    if (any(b(4:5,:) /= -3.25)) failures=failures+1
    b(1:3,:)=poison
    status=ddla_memcpy(handle,pb,c_loc(b),int(10*8,c_int64_t)); call check(status)
    status=ddla_d_trmm(handle,'L','U','N','N',3,2,zero,c_null_ptr,5,pb,5); call check(status)
    status=ddla_memcpy(handle,c_loc(b),pb,int(10*8,c_int64_t)); call check(status)
    status=ddla_synchronize(handle); call check(status)
    if (any(b(1:3,:) /= 0)) failures=failures+1
    if (any(b(4:5,:) /= -3.25)) failures=failures+1
    status=ddla_free(handle,pa); call check(status)
    status=ddla_free(handle,pb); call check(status)
  end subroutine check_d
  subroutine check_c(handle)
    type(c_ptr), intent(in) :: handle
    type(c_ptr) :: pa, pb
    complex(c_float_complex), target :: a(5,3), b(5,2)
    complex(c_float_complex) :: want(3,2)
    type(ddla_complex_float) :: alpha, zero
    real(c_float) :: base(3,2), poison
    integer(c_int) :: status
    integer :: i, j
    poison = ieee_value(0.0_c_float,ieee_quiet_nan)
    a = poison
    a(1,1)=2; a(1,2)=1; a(2,2)=3
    a(1,3)=-1; a(2,3)=2; a(3,3)=4
    b = -3.25
    do j=1,2
      do i=1,3
        b(i,j) = cmplx(real(i+3*(j-1),c_float),real(i+3*(j-1),c_float),kind=c_float)
      end do
    end do
    base(:,1) = [1,12,12]
    base(:,2) = [7,27,24]
    want = cmplx(0.75_c_float*base,0.25_c_float*base,kind=c_float)
    alpha = ddla_complex_float(0.5_c_float,-0.25_c_float)
    zero = ddla_complex_float(0.0_c_float,0.0_c_float)
    status=ddla_malloc(handle,pa,int(15*8,c_int64_t)); call check(status)
    status=ddla_malloc(handle,pb,int(10*8,c_int64_t)); call check(status)
    status=ddla_memcpy(handle,pa,c_loc(a),int(15*8,c_int64_t)); call check(status)
    status=ddla_memcpy(handle,pb,c_loc(b),int(10*8,c_int64_t)); call check(status)
    status=ddla_c_trmm(handle,'L','U','N','N',3,2,alpha,pa,5,pb,5); call check(status)
    status=ddla_memcpy(handle,c_loc(b),pb,int(10*8,c_int64_t)); call check(status)
    status=ddla_synchronize(handle); call check(status)
    if (any(.not.ieee_is_finite(real(b(1:3,:))))) failures=failures+1
    if (maxval(abs(b(1:3,:)-want)) > 1.0e-5) failures=failures+1
    if (any(b(4:5,:) /= -3.25)) failures=failures+1
    b(1:3,:)=poison
    status=ddla_memcpy(handle,pb,c_loc(b),int(10*8,c_int64_t)); call check(status)
    status=ddla_c_trmm(handle,'L','U','N','N',3,2,zero,c_null_ptr,5,pb,5); call check(status)
    status=ddla_memcpy(handle,c_loc(b),pb,int(10*8,c_int64_t)); call check(status)
    status=ddla_synchronize(handle); call check(status)
    if (any(b(1:3,:) /= 0)) failures=failures+1
    if (any(b(4:5,:) /= -3.25)) failures=failures+1
    status=ddla_free(handle,pa); call check(status)
    status=ddla_free(handle,pb); call check(status)
  end subroutine check_c
  subroutine check_z(handle)
    type(c_ptr), intent(in) :: handle
    type(c_ptr) :: pa, pb
    complex(c_double_complex), target :: a(5,3), b(5,2)
    complex(c_double_complex) :: want(3,2)
    type(ddla_complex_double) :: alpha, zero
    real(c_double) :: base(3,2), poison
    integer(c_int) :: status
    integer :: i, j
    poison = ieee_value(0.0_c_double,ieee_quiet_nan)
    a = poison
    a(1,1)=2; a(1,2)=1; a(2,2)=3
    a(1,3)=-1; a(2,3)=2; a(3,3)=4
    b = -3.25
    do j=1,2
      do i=1,3
        b(i,j) = cmplx(real(i+3*(j-1),c_double),real(i+3*(j-1),c_double),kind=c_double)
      end do
    end do
    base(:,1) = [1,12,12]
    base(:,2) = [7,27,24]
    want = cmplx(0.75_c_double*base,0.25_c_double*base,kind=c_double)
    alpha = ddla_complex_double(0.5_c_double,-0.25_c_double)
    zero = ddla_complex_double(0.0_c_double,0.0_c_double)
    status=ddla_malloc(handle,pa,int(15*16,c_int64_t)); call check(status)
    status=ddla_malloc(handle,pb,int(10*16,c_int64_t)); call check(status)
    status=ddla_memcpy(handle,pa,c_loc(a),int(15*16,c_int64_t)); call check(status)
    status=ddla_memcpy(handle,pb,c_loc(b),int(10*16,c_int64_t)); call check(status)
    status=ddla_z_trmm(handle,'L','U','N','N',3,2,alpha,pa,5,pb,5); call check(status)
    status=ddla_memcpy(handle,c_loc(b),pb,int(10*16,c_int64_t)); call check(status)
    status=ddla_synchronize(handle); call check(status)
    if (any(.not.ieee_is_finite(real(b(1:3,:))))) failures=failures+1
    if (maxval(abs(b(1:3,:)-want)) > 1.0e-12) failures=failures+1
    if (any(b(4:5,:) /= -3.25)) failures=failures+1
    b(1:3,:)=poison
    status=ddla_memcpy(handle,pb,c_loc(b),int(10*16,c_int64_t)); call check(status)
    status=ddla_z_trmm(handle,'L','U','N','N',3,2,zero,c_null_ptr,5,pb,5); call check(status)
    status=ddla_memcpy(handle,c_loc(b),pb,int(10*16,c_int64_t)); call check(status)
    status=ddla_synchronize(handle); call check(status)
    if (any(b(1:3,:) /= 0)) failures=failures+1
    if (any(b(4:5,:) /= -3.25)) failures=failures+1
    status=ddla_free(handle,pa); call check(status)
    status=ddla_free(handle,pb); call check(status)
  end subroutine check_z
end program test_fortran_trmm
