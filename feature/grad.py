from feature.region import Region

import numpy as np
from collections import defaultdict
import h5py

from pyfr.readers.native import NativeReader
from pyfr.quadrules import get_quadrule
from pyfr.util import subclass_where
from pyfr.shapes import BaseShape

class Gradient(Region):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv, icfg, fname)

        from pyfr.solvers.base import BaseSystem
        # System and elements classes
        self.systemscls = subclass_where(
            BaseSystem, name=self.cfg.get('solver', 'system')
        )
        self.elementscls = self.systemscls.elementscls

    def _get_eles(self):
        _, mesh_wall = self.get_boundary()
        return mesh_wall

    def gradproc(self):
        # Prepare for MPI process
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            mesh_wall = self._get_eles()

            mesh = self._pre_proc_mesh(mesh_wall)
            self._flash_to_disk(mesh)

        else:
            mesh = None
            mesh_wall = None


        # Boardcast pts and eles information
        mesh = comm.bcast(mesh, root=0)
        mesh_wall = comm.bcast(mesh_wall, root=0)

        # Get time series
        time = self.get_time_series_mpi(rank, size)

        soln_op = self._get_op_sln(mesh_wall)
        for t in time:
            self._proc_soln(t, mesh_wall, soln_op, mesh)


    def _proc_soln(self, time, mesh_wall, soln_op, mesh):
        soln = f'{self.solndir}{time}.pyfrs'
        soln = self._load_snapshot(soln, mesh_wall, soln_op)

        for etype in soln:
            soln[etype] = self._pre_proc_fields_grad(etype, mesh[etype], soln[etype])

        self._flash_to_disk(soln, time)

    def _flash_to_disk(self, array, t = []):
        if t:
            f = h5py.File(f'{self.dir}/grad_{t}.s', 'w')
            for etype in array:
                f[f'{etype}'] = array[etype]
            f.close()
        else:
            f = h5py.File(f'{self.dir}/grad.m', 'w')
            for etype in array:
                f[f'{etype}'] = array[etype]
            f.close()


    def _load_snapshot(self, name, mesh_wall, soln_op):
        soln = defaultdict()
        f = h5py.File(name,'r')
        for k in mesh_wall:
            _, etype, part = k.split('_')
            name = f'{self.dataprefix}_{etype}_{part}'
            sln = np.array(f[name])[...,mesh_wall[k]]
            sln = np.einsum('ij, jkl -> ikl',soln_op[etype],sln)
            sln = self._pre_proc_fields_soln(sln)
            try:
                soln[etype] = np.append(soln[etype], sln, axis = -1)
            except KeyError:
                soln[etype] = sln
        f.close()
        return soln

    def _get_op_sln(self, mesh_wall):
        soln_op = {}
        for key in mesh_wall:
            _, etype, part = key.split('_')
            # Operator
            if etype not in soln_op:
                name = f'{self.dataprefix}_{etype}_{part}'
                nspts = self.soln[name].shape[0]
                soln_op[etype] = self._get_soln_op(etype, nspts)
        return soln_op




    def _pre_proc_mesh(self, mesh_wall):
        mesh = {}
        for key, eles in mesh_wall.items():
            _, etype, part = key.split('_')
            nspts = self.mesh[key].shape[0]
            # Operator
            mesh_op_vis = self._get_vis_op(nspts, etype, self.order)
            msh = self.mesh[key][:,mesh_wall[key]]
            msh = np.einsum('ij, jkl -> ikl', mesh_op_vis, msh)

            try:
                mesh[etype] = np.append(mesh[etype], msh, axis = 1)
            except KeyError:
                mesh[etype] = msh
        return mesh

    def _pre_proc_fields_soln(self, soln):
        # Convert from conservative to primitive variables
        return np.array(self.elementscls.con_to_pri(soln, self.cfg))

    def _pre_proc_fields_grad(self, name, mesh, soln):
        # Reduce solution size since only velocity gradient is interested
        soln = soln[:,1:self.ndims+1]

        # Dimensions
        nupts, nvars = soln.shape[:2]

        # Get the shape class
        basiscls = subclass_where(BaseShape, name=name)

        # Construct an instance of the relevant elements class
        eles = self.elementscls(basiscls, mesh, self.cfg)

        # Get the smats and |J|^-1 to untransform the gradient
        smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)
        rcpdjac = eles.rcpdjac_at_np('upts')

        # Gradient operator
        gradop = eles.basis.m4.astype(self.dtype)

        # Evaluate the transformed gradient of the solution
        gradsoln = gradop @ soln.reshape(nupts, -1)
        gradsoln = gradsoln.reshape(self.ndims, nupts, nvars, -1)

        # Untransform
        gradsoln = np.einsum('ijkl,jkml->mikl', smat*rcpdjac, gradsoln,
                             dtype=self.dtype, casting='same_kind')
        gradsoln = gradsoln.reshape(nvars*self.ndims, nupts, -1)

        return np.append(soln, gradsoln.swapaxes(0,1), axis = 1)


    def main(self):
        # Calculate node locations of VTU elements
        vpts = mesh_vtu_op @ mesh.reshape(nspts, -1)
        vpts = vpts.reshape(nsvpts, -1, self.ndims)

        # Pre-process the solution
        soln = self._pre_proc_fields(name, mesh, soln).swapaxes(0, 1)

        # Interpolate the solution to the vis points
        vsoln = soln_vtu_op @ soln.reshape(len(soln), -1)
        vsoln = vsoln.reshape(nsvpts, -1, neles).swapaxes(0, 1)
