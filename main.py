# -*- coding: utf-8 -*-
import sys
from argparse import ArgumentParser, FileType
from collections import defaultdict

#from mesh_process import Region, SpanAverage, Probes
#from functions.spectra import Spectra
from feature.region import Region
#from feature.spanproc import SpanBase
from feature.spanavg import SpanavgBase

#from feature.probes import Probes
#from feature.probes_bounding_box import Probes
#from feature.probes_bounding_box_ele import Probes
#from feature.probes_modified import Probes
from feature.probes_cloest import Probes

from feature.grad import Gradient
from feature.gatherup import GU2, GU_timeseries_slice2d, GU_timeseries_no_dp, GU_timeseries_grad_no_dp, GU_timeseries_stations_no_dp

from feature.Q_criterion import Q_criterion



#from functions.duppts import Duplicate_Pts
from functions.bl import BL, BL_Coeff, BL_Coeff_hotmap
from functions.duppts import Duplicate_Pts
from functions.spod_field import SPOD
from functions.wavenumtrans import Wavenumber_Trans
from functions.directivity import Directivity
from functions.correlation import Correlation


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
    fmat = cfg.get('time-series','fmat')
    arg['series_time'] = [tstart, tend, dt, fmat]

    return arg, cfg

def main():

    arg, cfg = config_file()

    if 'func-spanavg' in cfg.sections():
        fname = 'func-spanavg'
        #Region(arg, cfg, fname).get_wall_O_grid()
        #SpanBase(arg, cfg, fname).spanfft()
        SpanavgBase(arg, cfg, fname).main_proc()
    if 'func-probes' in cfg.sections():
        fname = 'func-probes'
        Probes(arg, cfg, fname).mainproc()
    if 'func-gradient' in cfg.sections():
        fname = 'func-gradient'
        Gradient(arg, cfg, fname).gradproc()

    #Spectra().process_data_cylinder2()
    if 'feature-bl' in cfg.sections():
        fname = 'feature-bl'
        #Duplicate_Pts(arg).main_proc()
        #BL(arg, cfg, fname).main_proc()
        #BL_Coeff(arg, cfg, fname).main_proc()
        BL_Coeff_hotmap(arg, cfg, fname).main_proc()

    if 'func-Q-criterion' in cfg.sections():
        fname = 'func-Q-criterion'
        Q_criterion(arg, cfg, fname).main_proc()

    if 'func-gu' in cfg.sections():
        #GU2(arg).main_proc()
        #GU_timeseries_slice2d(arg).main_proc()
        #GU_timeseries_no_dp(arg).main_proc()
        GU_timeseries_grad_no_dp(arg).main_proc()
        #GU_timeseries_stations_no_dp(arg).main_proc()

    if 'feature-spod' in cfg.sections():
        fname = 'feature-spod'
        SPOD(arg, cfg, fname).main_proc()

    if 'feature-directivity' in cfg.sections():
        fname = 'feature-directivity'
        Directivity(arg, cfg, fname).main_proc()

    if 'feature-wavenumber-trans' in cfg.sections():
        fname = 'feature-wavenumber-trans'
        Wavenumber_Trans(arg, cfg, fname).main_proc()

    if 'feature-correlation' in cfg.sections():
        fname = 'feature-correlation'
        Correlation(arg, cfg, fname).main_proc()

    if 'func-slices2d' in cfg.sections():
        from feature.slices2d import Slices
        Slices(arg).main_proc()

    if 'func-slices3d' in cfg.sections():
        from feature.slices3d import Slices
        Slices(arg).main_proc()


if __name__ == "__main__":
    main()
