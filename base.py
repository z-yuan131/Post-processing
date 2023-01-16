# -*- coding: utf-8 -*-
from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader
from pyfr.util import memoize, subclass_where
from pyfr.shapes import BaseShape
from pyfr.quadrules import get_quadrule

import numpy as np
import h5py
from collections import defaultdict


class Base(object):
    def __init__(self, avg):

        # Time series
        self.get_time_series(avg[2])

        # Define mesh name and solution name
        self.name = name = [avg[0],f'{avg[1]}_{self.time[0]}.pyfrs']

        # solution dirctory
        self.solndir = avg[1]

        self.mesh = NativeReader(name[0])
        self.soln = NativeReader(name[1])

        # Check solution and mesh are compatible
        if self.mesh['mesh_uuid'] != self.soln['mesh_uuid']:
            raise RuntimeError('Solution "%s" was not computed on mesh "%s"' %
                               (name[0], name[1]))

        # Load the configuration and stats files
        self.cfg = Inifile(self.soln['config'])
        self.stats = Inifile(self.soln['stats'])
        self.order = self.cfg.getint('solver','order')
        self.dtype = np.dtype(self.cfg.get('backend','precision')).type

        # Data file prefix (defaults to soln for backwards compatibility)
        self.dataprefix = self.stats.get('data', 'prefix', 'soln')
        #self.dataprefix = 'tavg'

        # Get element types and array shapes
        self.mesh_inf = self.mesh.array_info('spt')
        self.soln_inf = self.soln.array_info(self.dataprefix)

        # Get the number of elements of each type in each partition
        self.mesh_part = self.mesh.partition_info('spt')

        # Dimensions
        self.ndims = next(iter(self.mesh_inf.values()))[1][2]
        self.nvars = next(iter(self.soln_inf.values()))[1][1]

        # Ops
        #self.get_ops()

        # Constants
        self._constants = self.cfg.items_as('constants', float)

        # If using Sutherland's Law as viscous correction
        self._viscorr = self.cfg.get('solver', 'viscosity-correction', 'none')

    def get_time_series(self, time):

        tt = np.arange(time[0], time[1], time[2])
        self.time = list()
        for i in range(len(tt)):
            self.time.append("{:.1f}".format(tt[i]))

    # Operators
    def _get_ops(self, nspts, etype, upts, nupts, order):

        svpts = self._get_std_ele(etype, nspts, order)
        mesh_op = self._get_mesh_op_vis(etype, nspts, svpts)

        # Convert vis points to solution pts
        mesh_op = self._get_mesh_op_sln(etype, nupts, upts) @ mesh_op
        return mesh_op

    def _get_vis_op(self, nspts, etype, order):
        svpts = self._get_std_ele(etype, nspts, order)
        mesh_op = self._get_mesh_op_vis(etype, nspts, svpts)
        return mesh_op

    def _get_shape(self, name, nspts, cfg):
        shapecls = subclass_where(BaseShape, name=name)
        return shapecls(nspts, cfg)

    def _get_std_ele(self, name, nspts, order):
        return self._get_shape(name, nspts, self.cfg).std_ele(order)

    def _get_mesh_op_vis(self, name, nspts, svpts):
        shape = self._get_shape(name, nspts, self.cfg)
        return shape.sbasis.nodal_basis_at(svpts).astype(self.dtype)

    def _get_mesh_op_sln(self, name, nspts, upts):
        shape = self._get_shape(name, nspts, self.cfg)
        return shape.sbasis.nodal_basis_at(upts).astype(self.dtype)

    def _get_npts(self, name, order):
        return self._get_shape(name, 0, self.cfg).nspts_from_order(order)

    def _get_order(self, name, nspts):
        return self._get_shape(name, nspts, self.cfg).order_from_nspts(nspts)


    