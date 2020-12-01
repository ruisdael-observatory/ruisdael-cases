# Ruisdael small domain setup with uniform forcing

Based on the KNMI_testbed, [cabauw case](https://github.com/julietbravo/KNMI_testbed/tree/master/cases/cabauw).


## DALES version

The nudging code required for this case is included in the [rusisdael](https://github.com/dalesteam/dales/tree/ruisdael) branch of DALES.
The new land surface model is in there and used in this case.

The DALES version used for this case with the KNMI testbed is here:
```
git clone https://github.com/dalesteam/dales.git
cd dales
git checkout ruisdael
```

## Data

It still uses the 'old' LES forcing data: 

On Cartesius the required data is in `/projects/0/einf170/LES_forcings/`
```
path = '/projects/0/einf170/LES_forcings/extracted/'
path_e5 = '/projects/0/einf170/LES_forcings/'
```
Unpack LES_forcings_*.tar.gz as necessary: `tar -C extracted -xzf LES_forcings_201705.tar.gz`

The main script used to drive the testbed is `create_input.py`. Modify the script so the paths point to the LES forcings (`path`), ERA5 soil data (`path_e5`), and output directory for the experiments (`path_out`). The variable `iloc` defines which location is used:
    
    # Single column:
    0  = FINO1,        lat=54.01, lon=6.59
    1  = Goeree,       lat=51.93, lon=3.67
    2  = Europlatform, lat=52.00, lon=3.27
    3  = K13,          lat=53.22, lon=3.22
    4  = HKZ,          lat=52.30, lon=4.10
    5  = P11B,         lat=52.36, lon=3.34
    6  = F3-FB-1,      lat=54.85, lon=4.70
    7  = Cabauw,       lat=51.97, lon=4.90
    8  = Loobos,       lat=52.17, lon=5.74
    9  = Lutjewad,     lat=53.40, lon=6.35
    10 = Schiphol,     lat=52.31, lon=4.76
    11 = Rotterdam,    lat=51.91, lon=4.47
    
    # 10x10 & 30x30 km mean:
    12 / 24 = FINO1
    13 / 25 = Goeree
    14 / 26 = Europlatform
    15 / 27 = K13
    16 / 28 = HKZ
    17 / 29 = P11B
    18 / 30 = F3-FB-1
    19 / 31 = Cabauw
    20 / 32 = Loobos
    21 / 33 = Lutjewad
    22 / 34 = Schiphol
    23 / 35 = Rotterdam

Before running the script, the `dales4` executable should be copied to the `cases/cabauw` directory. Running the `create_input.py` script should now create the full case setup in the `path_out` directory, including the `run_DALES.sh` script.

## Running on the BULL

Create boundary and initial conditions for DALES on workstation:
```
cd ruisdael-cases

cd cases/small_domain_atKNMI/                                                                          
# edit create_input.py, set output directory                                                 

python create_input.py                                                                       

```
Copy over to the BULL
....... to add something smart .....

Run the job:
```
cd <output directory>
sbatch run_DALES.sh
```

## References

DOWA/KNMI technical report (https://www.dutchoffshorewindatlas.nl/publications/reports/2019/12/10/knmi-report---downscaling-harmonie-arome-with-large-eddy-simulation)
