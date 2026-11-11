! Exercise all typed four-flag TRTRM C/Fortran bindings on every available backend.
program test_fortran_trtrm
  use iso_c_binding
  use ieee_arithmetic
  use ddla
  implicit none
  include 'mpif.h'
  type(c_ptr) :: h
  integer(c_int) :: backend, status
  integer :: provided
  integer :: ierr, rank, failures, total
  failures = 0
  call MPI_Init_thread(MPI_THREAD_FUNNELED, provided, ierr)
  if (ierr /= MPI_SUCCESS) stop 1
  if (provided < MPI_THREAD_FUNNELED) call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  do backend = DDLA_BACKEND_CPU, DDLA_BACKEND_GPU
    if (ddla_backend_available(backend) == 0) cycle
    status = ddla_init_backend(h, backend); call check(status)
    status = ddla_set(h, int(MPI_COMM_WORLD,c_int), 0, 0, 'R'); call check(status)
    call check_s(h)
    call check_d(h)
    call check_c(h)
    call check_z(h)
    status = ddla_destroy(h); call check(status)
  end do
  call MPI_Allreduce(failures,total,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD,ierr)
  if (rank == 0 .and. total == 0) print *, 'test_fortran_trtrm passed'
  if (total /= 0) call MPI_Abort(MPI_COMM_WORLD,1,ierr)
  call MPI_Finalize(ierr)
contains
  subroutine check(st)
    integer(c_int), intent(in) :: st
    if (st /= DDLA_STATUS_SUCCESS) then
      print *, 'Fortran TRTRM binding status:', st
      call MPI_Abort(MPI_COMM_WORLD,1,ierr)
    end if
  end subroutine
  subroutine check_s(handle)
    type(c_ptr), intent(in) :: handle
    type(c_ptr) :: pa, pb, pc
    real(c_float), target :: a(5,3), b(5,3), c(5,3)
    real(c_float) :: oa(3,3), ob(3,3), product(3,3), want(5,3), scale
    real(c_float) :: alpha, beta
    real(c_float) :: poison
    character :: ua, ub, ta, tb
    character, parameter :: uplos(2) = ['U','L'], trans(3) = ['N','T','C']
    integer :: ia, ib, xa, xb, i, j
    logical :: upper_a, upper_b
    integer(c_int) :: st
    poison = ieee_value(0.0_c_float,ieee_quiet_nan)
    scale = 0.5_c_float
    alpha = scale
    beta = 0.25_c_float
    st=ddla_malloc(handle,pa,int(15*4,c_int64_t)); call check(st)
    st=ddla_malloc(handle,pb,int(15*4,c_int64_t)); call check(st)
    st=ddla_malloc(handle,pc,int(15*4,c_int64_t)); call check(st)
    do ia=1,2
      do ib=1,2
        do xa=1,3
          do xb=1,3
            ua=uplos(ia); ub=uplos(ib); ta=trans(xa); tb=trans(xb)
            a=poison; b=poison; c=-3.25; oa=0; ob=0
            do j=1,3
              do i=1,3
                if ((ua == 'U' .and. i <= j) .or. (ua == 'L' .and. i >= j)) then
                  a(i,j)=real(i+2*j,c_float)
                  oa(i,j)=a(i,j)
                end if
                if ((ub == 'U' .and. i <= j) .or. (ub == 'L' .and. i >= j)) then
                  b(i,j)=real(3*i-j,c_float)
                  ob(i,j)=b(i,j)
                end if
              end do
            end do
            if (ta /= 'N') oa=transpose(oa)
            if (tb /= 'N') ob=transpose(ob)
            product=scale*matmul(oa,ob) + 0.25_c_float*c(1:3,:)
            upper_a=(ua == 'U') .neqv. (ta /= 'N')
            upper_b=(ub == 'U') .neqv. (tb /= 'N')
            want=c
            do j=1,3
              do i=1,3
                if ((upper_a .neqv. upper_b) .or. (upper_a .and. i<=j) .or. &
                    (.not.upper_a .and. i>=j)) want(i,j)=product(i,j)
              end do
            end do
            st=ddla_memcpy(handle,pa,c_loc(a),int(15*4,c_int64_t)); call check(st)
            st=ddla_memcpy(handle,pb,c_loc(b),int(15*4,c_int64_t)); call check(st)
            st=ddla_memcpy(handle,pc,c_loc(c),int(15*4,c_int64_t)); call check(st)
            st=ddla_s_trtrm(handle,ua,ta,ub,tb,3,alpha,pa,5,pb,5,beta,pc,5); call check(st)
            st=ddla_memcpy(handle,c_loc(c),pc,int(15*4,c_int64_t)); call check(st)
            st=ddla_synchronize(handle); call check(st)
            if (any(.not.ieee_is_finite(real(c)))) failures=failures+1
            if (maxval(abs(c-want)) > 1.0e-4) failures=failures+1
          end do
        end do
      end do
    end do
    st=ddla_free(handle,pa); call check(st)
    st=ddla_free(handle,pb); call check(st)
    st=ddla_free(handle,pc); call check(st)
  end subroutine check_s
  subroutine check_d(handle)
    type(c_ptr), intent(in) :: handle
    type(c_ptr) :: pa, pb, pc
    real(c_double), target :: a(5,3), b(5,3), c(5,3)
    real(c_double) :: oa(3,3), ob(3,3), product(3,3), want(5,3), scale
    real(c_double) :: alpha, beta
    real(c_double) :: poison
    character :: ua, ub, ta, tb
    character, parameter :: uplos(2) = ['U','L'], trans(3) = ['N','T','C']
    integer :: ia, ib, xa, xb, i, j
    logical :: upper_a, upper_b
    integer(c_int) :: st
    poison = ieee_value(0.0_c_double,ieee_quiet_nan)
    scale = 0.5_c_double
    alpha = scale
    beta = 0.25_c_double
    st=ddla_malloc(handle,pa,int(15*8,c_int64_t)); call check(st)
    st=ddla_malloc(handle,pb,int(15*8,c_int64_t)); call check(st)
    st=ddla_malloc(handle,pc,int(15*8,c_int64_t)); call check(st)
    do ia=1,2
      do ib=1,2
        do xa=1,3
          do xb=1,3
            ua=uplos(ia); ub=uplos(ib); ta=trans(xa); tb=trans(xb)
            a=poison; b=poison; c=-3.25; oa=0; ob=0
            do j=1,3
              do i=1,3
                if ((ua == 'U' .and. i <= j) .or. (ua == 'L' .and. i >= j)) then
                  a(i,j)=real(i+2*j,c_double)
                  oa(i,j)=a(i,j)
                end if
                if ((ub == 'U' .and. i <= j) .or. (ub == 'L' .and. i >= j)) then
                  b(i,j)=real(3*i-j,c_double)
                  ob(i,j)=b(i,j)
                end if
              end do
            end do
            if (ta /= 'N') oa=transpose(oa)
            if (tb /= 'N') ob=transpose(ob)
            product=scale*matmul(oa,ob) + 0.25_c_double*c(1:3,:)
            upper_a=(ua == 'U') .neqv. (ta /= 'N')
            upper_b=(ub == 'U') .neqv. (tb /= 'N')
            want=c
            do j=1,3
              do i=1,3
                if ((upper_a .neqv. upper_b) .or. (upper_a .and. i<=j) .or. &
                    (.not.upper_a .and. i>=j)) want(i,j)=product(i,j)
              end do
            end do
            st=ddla_memcpy(handle,pa,c_loc(a),int(15*8,c_int64_t)); call check(st)
            st=ddla_memcpy(handle,pb,c_loc(b),int(15*8,c_int64_t)); call check(st)
            st=ddla_memcpy(handle,pc,c_loc(c),int(15*8,c_int64_t)); call check(st)
            st=ddla_d_trtrm(handle,ua,ta,ub,tb,3,alpha,pa,5,pb,5,beta,pc,5); call check(st)
            st=ddla_memcpy(handle,c_loc(c),pc,int(15*8,c_int64_t)); call check(st)
            st=ddla_synchronize(handle); call check(st)
            if (any(.not.ieee_is_finite(real(c)))) failures=failures+1
            if (maxval(abs(c-want)) > 1.0e-12) failures=failures+1
          end do
        end do
      end do
    end do
    st=ddla_free(handle,pa); call check(st)
    st=ddla_free(handle,pb); call check(st)
    st=ddla_free(handle,pc); call check(st)
  end subroutine check_d
  subroutine check_c(handle)
    type(c_ptr), intent(in) :: handle
    type(c_ptr) :: pa, pb, pc
    complex(c_float_complex), target :: a(5,3), b(5,3), c(5,3)
    complex(c_float_complex) :: oa(3,3), ob(3,3), product(3,3), want(5,3), scale
    type(ddla_complex_float) :: alpha, beta
    real(c_float) :: poison
    character :: ua, ub, ta, tb
    character, parameter :: uplos(2) = ['U','L'], trans(3) = ['N','T','C']
    integer :: ia, ib, xa, xb, i, j
    logical :: upper_a, upper_b
    integer(c_int) :: st
    poison = ieee_value(0.0_c_float,ieee_quiet_nan)
    scale = cmplx(0.5_c_float,-0.25_c_float,kind=c_float)
    alpha = ddla_complex_float(real(scale),aimag(scale))
    beta = ddla_complex_float(0.25_c_float,0.0_c_float)
    st=ddla_malloc(handle,pa,int(15*8,c_int64_t)); call check(st)
    st=ddla_malloc(handle,pb,int(15*8,c_int64_t)); call check(st)
    st=ddla_malloc(handle,pc,int(15*8,c_int64_t)); call check(st)
    do ia=1,2
      do ib=1,2
        do xa=1,3
          do xb=1,3
            ua=uplos(ia); ub=uplos(ib); ta=trans(xa); tb=trans(xb)
            a=poison; b=poison; c=-3.25; oa=0; ob=0
            do j=1,3
              do i=1,3
                if ((ua == 'U' .and. i <= j) .or. (ua == 'L' .and. i >= j)) then
                  a(i,j)=cmplx(real(i+2*j,c_float),real(2*i-j,c_float),kind=c_float)
                  oa(i,j)=a(i,j)
                end if
                if ((ub == 'U' .and. i <= j) .or. (ub == 'L' .and. i >= j)) then
                  b(i,j)=cmplx(real(3*i-j,c_float),real(i+j,c_float),kind=c_float)
                  ob(i,j)=b(i,j)
                end if
              end do
            end do
            if (ta /= 'N') oa=transpose(oa)
            if (tb /= 'N') ob=transpose(ob)
            if (ta == 'C') oa=conjg(oa)
            if (tb == 'C') ob=conjg(ob)
            product=scale*matmul(oa,ob) + 0.25_c_float*c(1:3,:)
            upper_a=(ua == 'U') .neqv. (ta /= 'N')
            upper_b=(ub == 'U') .neqv. (tb /= 'N')
            want=c
            do j=1,3
              do i=1,3
                if ((upper_a .neqv. upper_b) .or. (upper_a .and. i<=j) .or. &
                    (.not.upper_a .and. i>=j)) want(i,j)=product(i,j)
              end do
            end do
            st=ddla_memcpy(handle,pa,c_loc(a),int(15*8,c_int64_t)); call check(st)
            st=ddla_memcpy(handle,pb,c_loc(b),int(15*8,c_int64_t)); call check(st)
            st=ddla_memcpy(handle,pc,c_loc(c),int(15*8,c_int64_t)); call check(st)
            st=ddla_c_trtrm(handle,ua,ta,ub,tb,3,alpha,pa,5,pb,5,beta,pc,5); call check(st)
            st=ddla_memcpy(handle,c_loc(c),pc,int(15*8,c_int64_t)); call check(st)
            st=ddla_synchronize(handle); call check(st)
            if (any(.not.ieee_is_finite(real(c)))) failures=failures+1
            if (maxval(abs(c-want)) > 1.0e-4) failures=failures+1
          end do
        end do
      end do
    end do
    st=ddla_free(handle,pa); call check(st)
    st=ddla_free(handle,pb); call check(st)
    st=ddla_free(handle,pc); call check(st)
  end subroutine check_c
  subroutine check_z(handle)
    type(c_ptr), intent(in) :: handle
    type(c_ptr) :: pa, pb, pc
    complex(c_double_complex), target :: a(5,3), b(5,3), c(5,3)
    complex(c_double_complex) :: oa(3,3), ob(3,3), product(3,3), want(5,3), scale
    type(ddla_complex_double) :: alpha, beta
    real(c_double) :: poison
    character :: ua, ub, ta, tb
    character, parameter :: uplos(2) = ['U','L'], trans(3) = ['N','T','C']
    integer :: ia, ib, xa, xb, i, j
    logical :: upper_a, upper_b
    integer(c_int) :: st
    poison = ieee_value(0.0_c_double,ieee_quiet_nan)
    scale = cmplx(0.5_c_double,-0.25_c_double,kind=c_double)
    alpha = ddla_complex_double(real(scale),aimag(scale))
    beta = ddla_complex_double(0.25_c_double,0.0_c_double)
    st=ddla_malloc(handle,pa,int(15*16,c_int64_t)); call check(st)
    st=ddla_malloc(handle,pb,int(15*16,c_int64_t)); call check(st)
    st=ddla_malloc(handle,pc,int(15*16,c_int64_t)); call check(st)
    do ia=1,2
      do ib=1,2
        do xa=1,3
          do xb=1,3
            ua=uplos(ia); ub=uplos(ib); ta=trans(xa); tb=trans(xb)
            a=poison; b=poison; c=-3.25; oa=0; ob=0
            do j=1,3
              do i=1,3
                if ((ua == 'U' .and. i <= j) .or. (ua == 'L' .and. i >= j)) then
                  a(i,j)=cmplx(real(i+2*j,c_double),real(2*i-j,c_double),kind=c_double)
                  oa(i,j)=a(i,j)
                end if
                if ((ub == 'U' .and. i <= j) .or. (ub == 'L' .and. i >= j)) then
                  b(i,j)=cmplx(real(3*i-j,c_double),real(i+j,c_double),kind=c_double)
                  ob(i,j)=b(i,j)
                end if
              end do
            end do
            if (ta /= 'N') oa=transpose(oa)
            if (tb /= 'N') ob=transpose(ob)
            if (ta == 'C') oa=conjg(oa)
            if (tb == 'C') ob=conjg(ob)
            product=scale*matmul(oa,ob) + 0.25_c_double*c(1:3,:)
            upper_a=(ua == 'U') .neqv. (ta /= 'N')
            upper_b=(ub == 'U') .neqv. (tb /= 'N')
            want=c
            do j=1,3
              do i=1,3
                if ((upper_a .neqv. upper_b) .or. (upper_a .and. i<=j) .or. &
                    (.not.upper_a .and. i>=j)) want(i,j)=product(i,j)
              end do
            end do
            st=ddla_memcpy(handle,pa,c_loc(a),int(15*16,c_int64_t)); call check(st)
            st=ddla_memcpy(handle,pb,c_loc(b),int(15*16,c_int64_t)); call check(st)
            st=ddla_memcpy(handle,pc,c_loc(c),int(15*16,c_int64_t)); call check(st)
            st=ddla_z_trtrm(handle,ua,ta,ub,tb,3,alpha,pa,5,pb,5,beta,pc,5); call check(st)
            st=ddla_memcpy(handle,c_loc(c),pc,int(15*16,c_int64_t)); call check(st)
            st=ddla_synchronize(handle); call check(st)
            if (any(.not.ieee_is_finite(real(c)))) failures=failures+1
            if (maxval(abs(c-want)) > 1.0e-12) failures=failures+1
          end do
        end do
      end do
    end do
    st=ddla_free(handle,pa); call check(st)
    st=ddla_free(handle,pb); call check(st)
    st=ddla_free(handle,pc); call check(st)
  end subroutine check_z
end program test_fortran_trtrm
