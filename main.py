# -*- coding: utf-8 -*-
import sys
from argparse import ArgumentParser, FileType
from collections import defaultdict

from mesh_process import Region, SpanAverage, Probes
#from functions.spectra import Spectra

from pyfr.inifile import Inifile


def config_file():
    # This is configuration part to pass parameters to the main part of the code.
    cfg = Inifile.load(sys.argv[1])
    arg = {}

    # Directory that stores mesh and solution
    # For mesh, give full directory and mesh name
    # For solution, give full directory and prefix of solution, i.e.
    dir = cfg.get('directory','dir')
    mname = cfg.get('directory','mesh_name')
    sheader = cfg.get('directory','soln_header')
    odir = cfg.get('directory','outdir')

    arg['mesh'] = f'{dir}/{mname}'
    arg['soln'] = f'{dir}/{sheader}'
    arg['odir'] = f'{odir}'

    # Time series (if single snapshot, keep start time equals end time)
    tstart = cfg.getfloat('time-series','tstart')
    tend = cfg.getfloat('time-series','tend')
    dt = cfg.getfloat('time-series','dt')
    arg['series_time'] = [tstart, tend, dt]

    return arg, cfg

def main():

    arg, cfg = config_file()


    #Region(arg).get_wall_O_grid()
    if 'func-spanavg' in cfg.sections():
        fname = 'func-spanavg'
        SpanAverage(arg, cfg, fname).spanavg()
    #import numpy as np
    #Probes(arg).mainproc(np.array([[100,100,-9],[100,20,-6]]))

    #Spectra().process_data_cylinder2()


if __name__ == "__main__":
    main()
