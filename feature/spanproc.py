from feature.region import Region

import numpy as np
from collections import defaultdict
import h5py

from pyfr.readers.native import NativeReader
from pyfr.quadrules import get_quadrule

import matplotlib.pyplot as plt
#import matplotlib.tri as tri

"""
Idea here:
Use Region to extrat the region that we interested. Note here only mesh_wall
is kind of usedful since it has all information about element ids we need.

second, use the function in the Spanavg ( change the name) to re order the
elements relationship that we need for doing span average or FFT in span. Note,
avgspan and fftspan could be two distinct useful functions. More importantly,
it would be better to find a way to output the results from reordering.
Otherwise it could be a problem in loading snapshot if mpi failed.

third, mpi process for loading and process all snapshots. The key point here is
to evenly distribute all snapshots that needs to output.

After all of these, the total memory requirement shoudl be much much smaller
and could be able to process locally for plotting etc.


"""


class SpanBase(Region):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv, icfg, fname)
        self.mode = icfg.get(fname, 'mode', 'mesh')
        self.tol = icfg.getfloat(fname, 'tol', 1e-6)
        self.outfreq = icfg.getfloat(fname, 'outfreq', 1)
        self._linmap = lambda n: {'hex': np.array([0, n-1, n**2-n, n**2-1,
                                n**2*(n-1), (n-1)*(n**2+1), n**3-n, n**3-1])}

    def _get_eles(self):
        f = h5py.File(f'{self.dir}/region.m','r')
        mesh_wall = defaultdict()
        for key in f:
            mesh_wall[key] = list(f[key])
        f.close()
        return mesh_wall

    def _ele_reorder(self):
        mesh_wall = self._get_eles()

        # Pre-process mesh
        mesh, lookup = self.mesh_proc(mesh_wall)

        # Collect one periodic boundary
        amin = min([np.min(msh[:,-1]) for msh in mesh])
        index = [np.where(abs(msh[:,-1] - amin) < self.tol)[0] for msh in mesh]

        #peid = []
        """
        for id, eids in enumerate(index):
            if len(eids) > 0:
                _, etype, part = lookup[id].split('_')

                con, _ = self.load_connectivity(self, part)

                uid = mesh_wall[lookup[id]][eids]
                uid = zip(etype, uid)

                for l, r in con:
                    if r in uid and l in uid:
                        con_new.append()
        """

        #npeid = np.sum([len(index[id]) for id in range(len(mesh)) if len(index[id]) > 0])

        # Find adjacent elements in the periodic direction
        zeleid = defaultdict(list)
        n = 0
        for idp, eids in enumerate(index):
            if len(eids) > 0:
                for idm, msh in enumerate(mesh):
                    pts = mesh[idp][eids]
                    dists = [np.linalg.norm(msh[:,:2] - pt[:2], axis=1)
                                                                for pt in pts]
                    for peidx,dist in enumerate(dists):
                        iidx = list(np.where(dist < self.tol)[0])
                        if len(iidx) > 0:
                            zeleid[n+peidx].append((idm, iidx))

            n += len(eids)

        return zeleid, mesh_wall, lookup

    def spanfft(self):
        # Prepare for MPI process
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            zeleid, mesh_wall, lookup = self._ele_reorder()

        self.mesh_avg(zeleid, mesh_wall, lookup)


    def mesh_avg(self, zeleid, mesh_wall, lookup):
        ele_type = {'hex': 'quad'}
        soln_pts = {'quad': 'gauss-legendre'}

        mesh = []
        idlist = []
        sortid = []
        for key in lookup:
            mesh.append(self.mesh[key][:,mesh_wall[key]])

        n = self.meshord + 1
        mesh_avg = np.zeros([n**2,self.ndims,len(zeleid)])

        for id, item in zeleid.items():
            for idx, (kid, eids) in enumerate(item):
                if idx == 0:
                    msh = mesh[kid][:,eids]
                else:
                    msh = np.append(msh, mesh[kid][:,eids], axis = 1)

            idx, index1 = self.mesh_sort(msh)
            idlist.append(idx)
        if len(index1) != 1:
            raise RuntimeError('much more complicated case, not ready yet')
        else:
            index = self.highorderproc(index1[0], self.meshord)

        for id, item in zeleid.items():
            for idx, (key, eid) in enumerate(item):
                if idx == 0:
                    msh = mesh[key][:,eid]
                else:
                    msh = np.append(msh, mesh[key][:,eid], axis = 1)

            if len(idlist[id]) > 0:
                mt = msh[:,idlist[id]]
                msh[:,idlist[id]] = mt[index]

            msh = msh.reshape(n**2,-1,self.ndims, order = 'F')
            sortid.append(np.argsort(np.mean(msh[:,:,-1],axis = 0)))

            # Average
            mesh_avg[:,:,id] = np.mean(msh, axis = 1)

        # use strict soln pts set if not exit
        etype = ele_type['hex']
        try:
            self.cfg.get(f'solver-elements-{etype}', 'soln-pts')
        except:
            self.cfg.set(f'solver-elements-{etype}', 'soln-pts', soln_pts[etype])

        # Get operator
        mesh_op = self._get_mesh_op(etype, mesh_avg.shape[0])
        mesh_avg = np.einsum('ij,jkl -> ikl', mesh_op, mesh_avg)

        # Flash ro disk
        #self._flash_to_disk(self.dir, mesh_avg)
        f = h5py.File(f'{self.dir}/spanavg.m', 'w')
        f['mesh'] = mesh_avg
        for id in zeleid:
            for kid, eids in zeleid[id]:
                f[f'zeleid/{id}/{kid}'] = eids

        f['idlist'] = np.array(idlist)
        f['index'] = index1[0]
        f.close()

        return idlist, index1


    def mesh_sort(self, mesh):
        npts, neles = mesh.shape[:2]

        # Check if all elements in z direction are facing the same direction
        loc = [np.linalg.norm(mesh[0,0,:2] - mesh[0,i,:2], axis = 0) < self.tol
                                        for i in range(neles)]
        if not all(loc):
            idx = []
            _map = self._linmap(self.meshord+1)['hex']
            mesh = mesh[_map]
            msh = mesh[:,0]
            index = []
            nid = -1
            for id, k in enumerate(loc):
                if not k:
                    msh[:,-1] += np.min(mesh[:,id,-1]) - np.min(msh[:,-1])
                    dlist = [np.linalg.norm(msh[j] - mesh[:,id], axis = 1)
                                        < self.tol for j in range(len(_map))]
                    dlist = [k for d in dlist for k, x in enumerate(d) if x]

                    if dlist not in index:
                        nid += 1
                        index.append(dlist)

                    idx.append(id)

            return idx, index

    def highorderproc(self, index, order):
        print('Just tabulate it until I found a way to write it efficiently')
        n = order + 1
        for i in range(n**2):
            if i == 0:
                oindex = np.arange(n**3-n,n**3,1)
            else:
                oindex = np.append(oindex, np.arange(n**3-n*(i+1),n**3-n*i,1))

        return oindex


    def mesh_proc(self, mesh_wall):
        # Corner points map for hex type of elements
        n = self.meshord + 1
        _map = self._linmap(n)

        mesh = []
        lookup = []
        for k in mesh_wall:
            _, etype, part = k.split('_')
            msh = self.mesh[k][:,mesh_wall[k]]
            mesh.append(np.mean(msh[_map[etype]],axis = 0))
            lookup.append(k)
        return mesh, lookup

    def _load_snapshot(self, name, region):
        soln = []
        f = h5py.File(name, 'r')
        for k in region:
            _, etype, part = k.split('_')
            kk = f'{self.dataprefix}_{etype}_{part}'
            soln.append(np.array(f[kk])[...,region[k]].swapaxes(1,-1))
        f.close()
        return soln
