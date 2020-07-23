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

cd cases/large_domain_1/
# edit create_boundaries.py, set output directory
# mkdir <output directory>
python create_boundaries.py
# even if the script fails when plotting, it has already created the output files

```

Run the job:
```
cp namoptions.001 testbed.job <output directory>
cd <output directory>
sbatch testbed.job
```





