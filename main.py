# -*- coding: utf-8 -*-
import sys
from argparse import ArgumentParser, FileType
from collections import defaultdict

from mesh_process import Region, SpanAverage, Probes
from functions.spectra import Spectra


def config_file():
    # This is configuration part to pass parameters to the main part of the code.
    arg = {}

    # Directory that stores mesh and solution
    # For mesh, give full directory and mesh name
    # For solution, give full directory and prefix of solution, i.e.
    arg['mesh'] = '/scratch/zhenyang/compute/pyfr/Naca0012trip/cylinder_trip/new_geo/run/mesh.pyfrm'
    arg['soln'] = '/scratch/zhenyang/compute/pyfr/Naca0012trip/cylinder_trip/new_geo/run/naca0012'

    arg['mesh'] = '/scratch/zhenyang/compute/pyfr/Naca0012trip/Re2e5/run/mesh.pyfrm'
    arg['soln'] = '/scratch/zhenyang/compute/pyfr/Naca0012trip/Re2e5/run/naca0012'



    # Output directory
    #arg['odir'] = '/scratch/zhenyang/compute/pyfr/Naca0012trip/cylinder_trip/new_geo/run'

    # Time series (if single snapshot, keep start time equals end time)
    arg['series_time'] = [10005, 10025, 5]   # [t_start, t_end, dt]
    arg['series_time'] = [15000, 15007.5, 2.5]   # [t_start, t_end, dt]

    # Functions
    #func['region'] = True


    return arg

def main():

    arg = config_file()

    # Parallel sorting the whole time series
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    #Region(arg).get_wall_O_grid()
    SpanAverage(arg).spanavg(comm)
    #import numpy as np
    #Probes(arg).mainproc(np.array([[100,100,-9],[100,20,-6]]))

    #Spectra().process_data_cylinder2()


if __name__ == "__main__":
    main()
