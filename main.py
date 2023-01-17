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

    # Time series (if single snapshot, keep start time equals end time)
    arg['series_time'] = [10005, 10010.1, 5]   # [t_start, t_end, dt]

    # Functions
    #func['region'] = True


    return arg

def main():

    arg = config_file()


    #Region(arg).get_wall_O_grid()
    SpanAverage(arg).reorder()
    #import numpy as np
    #Probes(arg).mainproc(np.array([[100,100,-6],[200,200,-6]]))

    #Spectra().process_data()


if __name__ == "__main__":
    main()
