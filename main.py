# -*- coding: utf-8 -*-
import sys
from argparse import ArgumentParser, FileType

#from interpo import Interpo
#from BL.boundary_layer import Boundary_layer
#from BL.Re50k import Boundary_layer_Re50k
#from BL.Re50k_corrolation import Corrolation_Re50k

from mesh_process import Region, SpanAverage, Probes


def main():


    meshname = '/scratch/zhenyang/compute/pyfr/Naca0012trip/cylinder_trip/new_geo/run/mesh.pyfrm'
    solnname = '/scratch/zhenyang/compute/pyfr/Naca0012trip/cylinder_trip/new_geo/run/naca0012'
    series_time = [10005, 10010.1, 5]

    sys.argv = [meshname,solnname, series_time]


    #Region(sys.argv).get_wall_O_grid()
    #SpanAverage(sys.argv).average()
    import numpy as np
    Probes(sys.argv).mainproc(np.array([[100,100,-6],[200,200,-6]]))



if __name__ == "__main__":
    main()
