"""
Some tools to create run scripts for the KNMI SLURM load balancer
   each node has 28 cores
"""

def hours_to_walltime(hours):
    hh = int(hours)
    mm = int((hours-hh)*60)
    return '{0:02d}:{1:02d}:00'.format(hh, mm)


def create_runscript(job_name, ntasks, nnodes, walltime, work_dir, expnr):
    """
    Create SLURM script for forecast
    Arguments:
        job_name : Job name (as shown in queue)
        ntasks   : Number of MPI tasks
        nnodes   : Number of nodes
        walltime : Wall time limit in hours
        work_dir : Work directory of forecast
        expnr    : DALES experiment number
    """

    walltime = hours_to_walltime(walltime)

    f = open('run_DALES.sh', 'w')
    f.write('#!/bin/sh -x\n')
    f.write('#SBATCH --job-name={}\n'.format(job_name))
    f.write('#SBATCH --time="{}"\n\n'.format(walltime))

    f.write('#SBATCH -n{}\n'.format(ntasks))
    f.write('#SBATCH -N{}\n'.format(nnodes))

    f.write('#env\n')
    f.write('#module list\n')
    f.write('ulimit -s unlimited\n\n')

    # Switch to working directory
    f.write('cd {}\n\n'.format(work_dir))

    f.write('srun -N{0} -n{1} dales4 namoptions.{2:03d} > dales.out'.format(nnodes, ntasks, expnr))

    f.close()


if __name__ == '__main__':

    # Only executed when script is called directly (for testing)
    create_runscript ('job_1',  96, 24, '/home/bla', 1)
    create_postscript('post_1', 24, '/home/bla', 1, 192, 192, 160, 12, 8)
