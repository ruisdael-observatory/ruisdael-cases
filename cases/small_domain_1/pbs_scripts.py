"""
Some tools to create run scripts for the ECMWF PBS load balancer
"""

def hours_to_walltime(hours):
    hh = int(hours)
    mm = int((hours-hh)*60)
    return '{0:02d}:{1:02d}:00'.format(hh, mm)


def create_runscript(job_name, ntasks, walltime, work_dir, expnr,
        threads_per_task=1, mem_per_task='1GB', hyperthreads=1):
    """
    Create PBS script for forecast
    Arguments:
        job_name : Job name (as shown in queue)
        ntasks   : Number of MPI tasks
        walltime : Wall time limit in hours
        work_dir : Work directory of forecast
        expnr    : DALES experiment number
    """

    walltime = hours_to_walltime(walltime)

    f = open('run.PBS', 'w')
    f.write('#!/bin/ksh\n')
    f.write('#PBS -S /usr/bin/ksh\n')
    f.write('#PBS -q np\n')
    f.write('#PBS -N {}\n'.format(job_name))
    f.write('#PBS -m a\n')
    f.write('#PBS -l walltime={}\n\n'.format(walltime))

    f.write('#PBS -l EC_total_tasks={}\n'.format(ntasks))
    f.write('#PBS -l EC_threads_per_task={}\n'.format(threads_per_task))
    f.write('#PBS -l EC_memory_per_task={}\n'.format(mem_per_task))
    f.write('#PBS -l EC_hyperthreads={}\n\n'.format(hyperthreads))

    f.write('prgenvswitchto intel\n')
    f.write('module load netcdf4\n\n')

    # Switch to working directory
    f.write('cd {}\n\n'.format(work_dir))

    f.write('aprun -n {0} ./dales4 namoptions.{1:03d} > dales4.out'.format(ntasks, expnr))

    f.close()


def create_postscript(job_name, walltime, work_dir, expnr, itot, jtot, ktot, nprocx, nprocy):
    """
    Create PBS script for postprocessing
    Arguments:
        job_name  : Job name (as shown in queue)
        walltime  : Wall time limit in hours
        work_dir  : Work directory
        expnr     : DALES experiment number
        {ijk}tot  : Grid size DALES
        nproc{xy} : Number of MPI tasks in forecast
    """

    walltime = hours_to_walltime(walltime)

    f = open('post.PBS', 'w')
    f.write('#!/bin/ksh\n')
    f.write('#PBS -S /usr/bin/ksh\n')
    f.write('#PBS -q ns\n')
    f.write('#PBS -N {}\n'.format(job_name))
    f.write('#PBS -m a\n')
    f.write('#PBS -l walltime={}\n\n'.format(walltime))

    f.write('module load python\n\n')

    # Switch to working directory
    f.write('cd {}\n\n'.format(work_dir))

    settings = '{} {} {} {} {} {}'.format(expnr, nprocx, nprocy, itot, jtot, ktot)

    f.write('python mergecross.py crossxy lwp {}\n'.format(settings))
    f.write('python mergecross.py crossxy rwp {}\n'.format(settings))

    f.close()


if __name__ == '__main__':

    # Only executed when script is called directly (for testing)
    create_runscript ('job_1',  96, 24, '/home/bla', 1)
    create_postscript('post_1', 24, '/home/bla', 1, 192, 192, 160, 12, 8)
