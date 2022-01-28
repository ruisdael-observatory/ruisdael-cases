Land-surface/soil input DALES for large domain Ruisdael runs
-------

### NOTES:
1. For now, the soil in LES is initialised from ERA5 data, as we are having some issues with the HARMONIE soil data. The required ERA5 data is automatically downloaded by `create_dales_input.py`, and uses the CDS api, and thus requires a working CDS environment: https://cds.climate.copernicus.eu/api-how-to. The soil initialisation from ERA5 requires a few _hacks_ to solve issues, mostly close to the coast line. Ideally, this should all be replaced by a soil initialisation from HARMONIE-AROME.
3. On Cartesius, the required land use and soil datasets are stored in `/projects/0/einf170/landuse_soil`.

### Overview files:
- `create_dales_input.py`: Main script, which generates the DALES LSM input. The domain settings should match the settings in `create_boundaries.py`.
- `bofek2012.py`: Helper script which reads `BOFEK2012_profielen_versie2_1.csv`, which is used to translate the BOFEK soil ID's to physical properties.
- `era5_soil.py`: Helper script to download and read/parse ERA5 soil data.
- `interpolate.py`: Various interpolation routines.
- `lsm_input_dales.py`: Helper script with a data structure for the required DALES LSM input, including methods to write the input to binaries (input DALES) or NetCDF.
- `spatial_transforms.py`: Definition of the HARMONIE projection in `pyproj`.
- `vegetation_properties.py`: Lookup tables with vegetation properties, including translation table from Top10NL land use types to IFS land use types.

### Setup on Snellius 2022

```
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
module load pyproj/3.1.0-GCCcore-10.3.0

# one-time installation of Python modules
pip install cdsapi
pip install numpy xarray netcdf4 matplotlib # some are already present
pip install numba "dask[complete]" progress

# edit create_dales_input.py script to specify domain and location of output files
python3 ./create_dales_input.py
```
