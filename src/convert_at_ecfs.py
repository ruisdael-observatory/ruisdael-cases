from __future__ import print_function

import pandas as pd
import datetime
import os
import subprocess
import tarfile
import sys
import pickle

from read_and_convert_DDH import *

# execute('module load ecfs')
ecfs_bin = '/usr/local/apps/ecfs/2.2.4/bin'

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def create_cycle_dir(path, date):
    year_dir  = '{0}/{1:04d}'.format(path,      date.year)
    month_dir = '{0}/{1:02d}'.format(year_dir,  date.month)
    day_dir   = '{0}/{1:02d}'.format(month_dir, date.day)
    cycle_dir = '{0}/{1:02d}'.format(day_dir,   date.hour)

    mkdir(year_dir )
    mkdir(month_dir)
    mkdir(day_dir  )
    mkdir(cycle_dir)

def execute(task):
    rc = subprocess.call(task, shell=True, executable='/bin/bash')

    if rc == 0:
        return True
    else:
        return False

def archive_ecfs(filename, date):
    ecfs_path = 'ec:/nkbs/DOWA_LES_forcings/{0:04d}/{1:02d}/{2:02d}/{3:02d}/'.format(date.year, date.month, date.day, date.hour)
    return execute('{}/ecp -p {} {}'.format(ecfs_bin, filename, ecfs_path))

if __name__ == '__main__':

    name = sys.argv[1]

    tmp_path = '/scratch/ms/nl/nkbs/DOWA/LES_forcing/'

    if name == 'ptD':
        start = datetime.datetime(year=2017, month=1, day=1)
        end   = datetime.datetime(year=2018, month=1, day=1)
        pname = '/home/ms/nl/nkbs/models/KNMI_testbed/src/DDH_processed_ptD.pickle'
        ecfs_path = 'ec:/nkl/harmonie/DOWA/DOWA_40h12tg2_fERA5/ptD_2015-2017/'
    elif name == 'ptE':
        start = datetime.datetime(year=2016, month=1, day=1)
        end   = datetime.datetime(year=2017, month=1, day=1)
        pname = '/home/ms/nl/nkbs/models/KNMI_testbed/src/DDH_processed_ptE.pickle'
        ecfs_path = 'ec:/nkl/harmonie/DOWA/DOWA_40h12tg2_fERA5/ptE_2016/'

    if os.path.exists(pname):
        # Read Dataframe from disk
        df = pd.read_pickle(pname)
    else:
        # Create new Dataframe to monitor post-processing
        dates = pd.date_range(start, end, freq='3H')
        column_names = ['copied_DDH', 'copied_soil', 'converted_nc', 'saved_ecfs']
        df = pd.DataFrame(index=dates, columns=column_names)
        df['copied_DDH']   = False   # Copied DDH tar from ECFS
        df['copied_soil']  = False   # Copied soil NetCDF from ECFS
        df['converted_nc'] = False   # Converted DDH to NetCDF
        df['saved_ecfs']   = False   # Saved NetCDF file in ECFS

    # --------------------------
    # DDH tars
    # --------------------------
    # Process only rows which haven't been copied
    to_copy = df.loc[df['copied_DDH'] == False]

    for index in to_copy.index:
        file_name = 'DDH_{0:04d}{1:02d}{2:02d}{3:02d}.tar.gz'.format(index.year, index.month, index.day, index.hour)
        ecfs_file = '{0}/{1:04d}/{2:02d}/{3:02d}/{4:02d}/{5}'.format(ecfs_path, index.year, index.month, index.day, index.hour, file_name)
        tmp_dir   = '{0}/{1:04d}/{2:02d}/{3:02d}/{4:02d}/'.format(tmp_path, index.year, index.month, index.day, index.hour)
        tmp_file  = '{0}/{1}'.format(tmp_dir, file_name)

        print('Processing DDH {} : '.format(index), end='')

        if os.path.exists(tmp_file):
            # Set download to finised
            df.loc[index, 'copied_DDH'] = True
            # Unpack
            execute('tar -xzvf {0}/{1} -C {2} >& /dev/null'.format(tmp_dir, file_name, tmp_dir))
            print('found local...')

        elif execute('{}/els {} >& /dev/null'.format(ecfs_bin, ecfs_file)):
            # File exists on ECFS! Create local directory (if it doesn't exists)
            create_cycle_dir(tmp_path, index)

            # Copy DDH tar file to tmp
            if execute('{0}/ecp {1} {2}'.format(ecfs_bin, ecfs_file, tmp_dir)):
                # Copy was success! Unpack...
                # DONT USE PYTHONS UNTAR!!
                #tar = tarfile.open('{0}/{1}'.format(tmp_dir, file_name))
                #tar.extractall(path=tmp_dir)
                #tar.close()
                execute('tar -xzvf {0}/{1} -C {2} >& /dev/null'.format(tmp_dir, file_name, tmp_dir))

                # Set download to finised
                df.loc[index, 'copied_DDH'] = True

                print('copied and unpacked DDH tar...')
        else:
            print('DDH tar not available on ECFS, stopping')
            break

    # Write Dataframe back to disk
    df.to_pickle(pname)

    # --------------------------
    # Convert to NetCDF
    # --------------------------
    to_convert = df.loc[(df['copied_DDH'] == True) & (df['converted_nc'] == False)]

    for index in to_convert.index:
        print('Converting {} to NetCDF'.format(index))

        tmp_dir  = '{0}/{1:04d}/{2:02d}/{3:02d}/{4:02d}/'.format(tmp_path, index.year, index.month, index.day, index.hour)
        out_file = '{0}/LES_forcing_{1:04d}{2:02d}{3:02d}{4:02d}.nc'.format(tmp_dir, index.year, index.month, index.day, index.hour)

        if not os.path.exists(out_file):
            # Read DDH files, and convert to NetCDF
            data = Read_DDH_files(tmp_path, index, 180, 10, quiet=True, add_soil=False)
            data.to_netcdf(out_file, add_domain_info=True)

        df.loc[index, 'converted_nc'] = True

    # Write Dataframe back to disk
    df.to_pickle(pname)

    # --------------------------
    # Archive in ECFS
    # --------------------------
    to_copy = df.loc[(df['converted_nc'] == True) & (df['saved_ecfs'] == False)]

    for index in to_copy.index:
        print('Copying {} to ECFS : '.format(index), end='')

        tmp_dir = '{0}/{1:04d}/{2:02d}/{3:02d}/{4:02d}/'.format(tmp_path, index.year, index.month, index.day, index.hour)
        nc_file = '{0}/LES_forcing_{1:04d}{2:02d}{3:02d}{4:02d}.nc'.format(tmp_dir, index.year, index.month, index.day, index.hour)

        # Copy to ECFS
        if archive_ecfs(nc_file, index):
            print('success!')
            df.loc[index, 'saved_ecfs'] = True
        else:
            print('copy failed, stopping')
            break

    # Write Dataframe back to disk
    df.to_pickle(pname)



























    # --------------------------
    # Soil NetCDF files
    # --------------------------
    # Process only rows which haven't been copied
    #to_copy = df.query('copied_soil == False')

    #for index in to_copy.index:
    #    # NetCDF files are gathered in daily files..
    #    if index.hour == 0:
    #        print('Processing soil {} : '.format(index), end='')

    #        # Keep track of wheter all copies were a success
    #        great_success = True

    #        for patch in ['P01']:
    #            # Soil moisture
    #            for level in ['L01', 'L02', 'L03']:
    #                file_name = 'wsa_{0}.{1}.sfx.NETHERLANDS.DOWA_40h12tg2_fERA5_ptD.{2:04d}{3:02d}{4:02d}.nc'.format(level, patch, index.year, index.month, index.day)
    #                ecfs_file = '{0}/{1:04d}/{2:02d}/{3:02d}/{4:02d}/{5}'.format(ecfs_path, index.year, index.month, index.day, index.hour, file_name)
    #                tmp_dir   = '{0}/{1:04d}/{2:02d}/{3:02d}/{4:02d}/'.format(tmp_path, index.year, index.month, index.day, index.hour)
    #                tmp_file  = '{0}/{1}'.format(tmp_dir, file_name)

    #                if execute('els {} >& /dev/null'.format(ecfs_file)):
    #                    # File exists on ECFS! Create local directory (if it doesn't exists)
    #                    create_cycle_dir(tmp_path, index)

    #                    # Copy file to tmp
    #                    if not execute('ecp {0} {1}'.format(ecfs_file, tmp_dir)):
    #                        great_success = False
    #                else:
    #                    great_success = False

    #            # Soil temperature
    #            for level in ['L01', 'L02']:
    #                file_name = 'tg_{0:}.{1:}.sfx.NETHERLANDS.DOWA_40h12tg2_fERA5_ptD.{2:04d}{3:02d}{4:02d}.nc'.format(level, patch, index.year, index.month, index.day)
    #                ecfs_file = '{0}/{1:04d}/{2:02d}/{3:02d}/{4:02d}/{5}'.format(ecfs_path, index.year, index.month, index.day, index.hour, file_name)
    #                tmp_dir   = '{0}/{1:04d}/{2:02d}/{3:02d}/{4:02d}/'.format(tmp_path, index.year, index.month, index.day, index.hour)

    #                if execute('els {} >& /dev/null'.format(ecfs_file)):
    #                    # File exists on ECFS! Create local directory (if it doesn't exists)
    #                    create_cycle_dir(tmp_path, index)

    #                    # Copy file to tmp
    #                    if not execute('ecp {0} {1}'.format(ecfs_file, tmp_dir)):
    #                        great_success = False
    #                else:
    #                    great_success = False

    #        # Flag as success if al files were copied
    #        if great_success:
    #            print('copied files!')
    #            df.loc[index, 'copied_soil'] = True
    #        else:
    #            print('one or more soil NetCDFs not available on ECFS, stopping')
    #            break

    ## Write Dataframe back to disk
    #df.to_pickle(pname)
