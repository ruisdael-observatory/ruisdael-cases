# Ruisdael large domain setup with boundary nudging 

Based on the KNMI_testbed, [nudge_boundary_HARMONIE case](https://github.com/julietbravo/KNMI_testbed/tree/master/cases/nudge_boundary_HARMONIE) .

## Data

On Cartesius the required data is in `/projects/0/einf170/Harmonie_boundaries`.
```
data_path = '/projects/0/einf170/Harmonie_boundaries'
```
See the case documentation for details.

## DALES version

The boundary nudging code required for this case is included in the [rusisdael](https://github.com/dalesteam/dales/tree/ruisdael) branch of DALES .

## Running on Cartesius

Create boundary and initial conditions for DALES:
```
cd ruisdael-cases

source setup_env_cartesius.sh 
# first time, run also the commented lines in the script to set up a Python environment

cd cases/large_domain_2/
# edit create_boundaries.py, set output directory
# mkdir <output directory>
python create_boundaries.py
# even if the script fails when plotting, it has already created the output files
```

```
corners
(52.64  3.86)  (52.53  6.41)
(51.60  3.77)  (51.50  6.26)
```

```
# prepare emission data and boundary conditions. For now, copy Marco's setup
# note: requires the grids to match
cp /scratch/shared/mdbruine/dales/runs/ruisdael/co2*   /projects/0/einf170/janssonf/run_large_domain_2/
cp /scratch/shared/mdbruine/dales/runs/ruisdael/lbcsv* /projects/0/einf170/janssonf/run_large_domain_2/

# data for LSM model, from git repository
cp ~/dales/data/van_genuchten_parameters.nc ./
```


second try, 2b. 9 Nodes = 12 x 18 tasks :
```
RUNDIR=/projects/0/einf170/janssonf/run_large_domain_2b/
mkdir $RUNDIR 
python create_boundaries.py
cp ~/dales/data/van_genuchten_parameters.nc $RUNDIR 
# new LBC for co2
cp /scratch/shared/mdbruine/dales/boundaries/ruisdael_procx12_procy18_2017081906/lbcsv* $RUNDIR
# same emission files as before (not tiled)
cp /scratch/shared/mdbruine/dales/runs/ruisdael/co2*  $RUNDIR
cp namoptions.001 testbed.job $RUNDIR
# add column in scalar.inp.001

# copy radiation data
cp ../run_large_domain_2/rrtmg_*.nc ../run_large_domain_2/backrad.inp.001.nc $RUNDIR/
```


Run the job:
```
cp namoptions.001 testbed.job <output directory>
cd <output directory>
sbatch testbed.job
```


Notes:

running with FFTW - needs FFTW support compiled into DALES.
scalar.inp.001 needs one extra column when nsv=3. 
Generated in create_boundaries.py. TODO: make configurable.

got this error message from emissions routine.
NetCDF: Error initializing for parallel access
STOP NetCDF error in modemission. See outputfile for more information.
Trying without MPI-aware netcdf open, seems to work.


Turning on radiation. Did not have it on before, in last ruisdael run ?
iradiation = 4      ! 4 = RRTMG 
timerad    = 60                
-> also needs rrtmg*.nc files and backrad.
-> need xtime=6   ! the starting time, also for emissions

Now using imicro=2 - maybe use 6 instead.
Using albedo 0.3, 0.17 would be more typical.
-- changed in 2b run


consider adding consecutive in job script


