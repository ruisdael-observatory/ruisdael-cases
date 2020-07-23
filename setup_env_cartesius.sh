# run this script with 'source'

module load 2019
module load Python/3.6.6-foss-2018b
module load Cartopy/0.15.1-foss-2018b-Python-3.6.6
module load netcdf4-python/1.4.1-foss-2018b-Python-3.6.6
module load Tkinter/3.6.6-foss-2018b-Python-3.6.6

# setup, first time:
#python3 -m venv --system-site-packages testbed_env
#source testbed_env/bin/activate
#pip install matplotlib==2.2.5 numba xarray "dask[complete]"
source testbed_env/bin/activate


