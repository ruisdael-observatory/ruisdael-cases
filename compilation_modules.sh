# source this file

# modules for compiling and running DALES
# note potential version conflict with the netCDF versions from the python set and this set

module load 2019
module load netCDF-Fortran/4.4.4-foss-2018b
module load CMake/3.12.1-GCCcore-7.3.0
module unload OpenMPI/3.1.1-GCC-7.3.0-2.30
module load OpenMPI/3.1.4-GCC-7.3.0-2.30
module load Hypre/2.14.0-foss-2018b
module load FFTW/3.3.8-gompi-2018b
export SYST=gnu-fast

# cd ~/dales
# mkdir build-ruisdael
# cd build-ruisdael
# cmake ..  -DUSE_HYPRE=True -DHYPRE_LIB=/sw/arch/RedHatEnterpriseServer7/EB_production/2019/software/Hypre/2.14.0-foss-2018b/lib/libHYPRE.a -DUSE_FFTW=True -DFFTW_LIB=/sw/arch/RedHatEnterpriseServer7/EB_production/2019/software/FFTW/3.3.8-gompi-2018b/lib/libfftw3.a -DFFTW_INCLUDE_DIR=/sw/arch/RedHatEnterpriseServer7/EB_production/2019/software/FFTW/3.3.8-gompi-2018b/include
# make -j 8
