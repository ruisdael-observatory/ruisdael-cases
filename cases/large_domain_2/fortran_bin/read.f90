program read_python
    implicit none

    integer(kind=8), allocatable :: a(:,:,:)
    integer(kind=8), allocatable :: b(:,:)
    integer :: itot=2, jtot=3, ktot=4
    integer :: i, j, k

    allocate(a(itot, jtot, ktot))
    allocate(b(itot, jtot))

    open(1, file="bla1.bin", form="unformatted", status="unknown", action="read", access="stream")
    read(1) a

    open(2, file="bla2.bin", form="unformatted", status="unknown", action="read", access="stream")
    read(2) b

    do i=1, itot
        do j=1, jtot
            do k=1, ktot
                print*, i, j, k, a(i,j,k)
            end do
        end do
    end do

    do i=1, itot
        do j=1, jtot
            print*, i, j, b(i,j)
        end do
    end do

    close(1)
    close(2)
    deallocate(a,b)

end program read_python
