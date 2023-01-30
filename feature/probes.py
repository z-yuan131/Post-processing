from base import Base

import numpy as np
from collections import defaultdict
import h5py

from pyfr.readers.native import NativeReader
from pyfr.quadrules import get_quadrule
from pyfr.util import subclasses
from pyfr.shapes import BaseShape

import matplotlib.pyplot as plt


class Probes(Base):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv)
        self.exactloc = icfg.getbool(fname, 'exactloc', False)
        npts = icfg.getint(fname, 'npts', 1)
        # List of points to be sampled and format
        pointA = icfg.getliteral(fname, 'samp-ptsA')
        pointB = icfg.getliteral(fname, 'samp-ptsB', None)
        self.fmt = icfg.get(fname, 'format', 'primitive')

        self.mode = icfg.get(fname, 'mode', 'mesh')

        self.preprocpts(pointA, pointB, npts)
        print(self.pts)


    def preprocpts(self, pta, ptb, npts):
        if ptb == None:
            self.pts = [np.array(pt) for pt in pta]
        else:
            if len(pta) != len(ptb):
                raise RuntimeError('Length of points lists should match.')
            self.pts = []
            for id in range(len(pta)):
                lt = [np.linspace(pta[id][i],ptb[id][i], npts)
                                                for i in range(self.ndims)]
                if self.ndims == 3:
                    self.pts += [np.array([x,y,z]) for x,y,z in zip(*lt)]
                else:
                    self.pts += [np.array([x,y]) for x,y in zip(*lt)]


    def pre_process2(self):
        self.mesh_bg = list()
        self.mesh_bg_vis = list()
        self.mesh_trans = list()
        self.lookup = list()

        # Use a strictly interior point set
        qrule_map = {
            'quad': 'gauss-legendre',
            'tri': 'williams-shunn',
            'hex': 'gauss-legendre',
            'pri': 'williams-shunn~gauss-legendre',
            'pyr': 'gauss-legendre',
            'tet': 'shunn-ham'
        }


        for key in self.mesh:
            if 'spt' in key.split('_'):
                _,etype,part = key.split('_')
                soln_name = f'{self.dataprefix}_{etype}_{part}'
                nupts = self.soln[soln_name].shape[0]

                # Get Operators
                nspts = self.mesh[key].shape[0]
                upts = get_quadrule(etype, qrule_map[etype], nupts).pts

                mesh_op = self._get_ops_interp(nspts, etype, upts, nupts, self.order)
                mesh_op_vis = self._get_vis_op(nspts, etype, self.order)

                self.mesh_bg.append(np.einsum('ij, jkl -> ikl',mesh_op,self.mesh[key]))
                self.mesh_bg_vis.append(np.einsum('ij, jkl -> ikl',mesh_op_vis,self.mesh[key]))
                self.mesh_trans.append(upts)
                self.lookup.append((etype,part))

    def data_process(self, option, elepts = []):
        # Use a strictly interior point set
        qrule_map = {
            'quad': 'gauss-legendre',
            'tri': 'williams-shunn',
            'hex': 'gauss-legendre',
            'pri': 'williams-shunn~gauss-legendre',
            'pyr': 'gauss-legendre',
            'tet': 'shunn-ham'
        }

        if option == 'bg':
            mesh = list()
            mesh_trans = list()
            self.lookup = list()
            for key in self.mesh:
                if 'spt' in key.split('_'):
                    _,etype,part = key.split('_')
                    # create lookup dictionary
                    self.lookup.append((etype,part))
                    # Get Operators
                    soln_name = f'{self.dataprefix}_{etype}_{part}'
                    nupts = self.soln[soln_name].shape[0]
                    nspts = self.mesh[key].shape[0]
                    upts = get_quadrule(etype, qrule_map[etype], nupts).pts
                    mesh_op = self._get_ops_interp(nspts, etype, upts, nupts, self.order)

                    mesh.append(np.einsum('ij, jkl -> ikl',mesh_op,self.mesh[key]))
                    mesh_trans.append(upts)
            return mesh, mesh_trans

        elif option == 'vis':
            mesh = list()
            for id, (etype, part) in enumerate(self.lookup):
                if not elepts[id]:
                    mesh.append([])
                    continue
                key = f'spt_{etype}_{part}'
                nspts = self.mesh[key].shape[0]
                # Operator
                mesh_op_vis = self._get_vis_op(nspts, etype, self.order)
                mesh.append(np.einsum('ij, jkl -> ikl',mesh_op_vis,self.mesh[key]))
            return mesh


    def mainproc(self):
        # Prepare for MPI process
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Only root rank is enough to process data
        if self.mode == 'mesh':
            if rank == 0:
                ptsinfo = self.procpts(self.pts)
                self.dump_to_file(ptsinfo)
            return 0
        else:
            if rank == 0:
                solninfo, intop, ploc, lookup = self.load_ptsinfo()
            else:
                solninfo = None
                lookup = None
                intop = None

        # Boardcast pts and eles information
        solninfo = comm.bcast(solninfo, root=0)
        lookup = comm.bcast(lookup, root=0)
        intop = comm.bcast(intop, root=0)

        comm.barrier()

        # Get time series
        time = self.get_time_series_mpi(rank, size)
        if len(time) > 0:
            soln_op = self._get_op_sln(solninfo, lookup)
            for t in time:
                self.procsoln(t, solninfo, lookup, soln_op, intop)

        print(rank, time)



    def procsoln(self, time, solninfo, lookup, soln_op, intop):
        soln = f'{self.solndir}{time}.pyfrs'
        soln = self._load_snapshot(soln, solninfo, lookup, soln_op)
        print(len(soln))


    def _load_snapshot(self, name, solninfo, lookup, soln_op):
        soln = []
        f = h5py.File(name,'r')
        for k in solninfo:
            etype,part = lookup[k].split('_')
            key = f'{self.dataprefix}_{etype}_{part}'
            sln = np.array(f[key])[...,solninfo[k]]
            sln = np.einsum('ij, jkl -> ikl',soln_op[k],sln)
            soln.append(sln)
        f.close()
        return soln

    def _get_op_sln(self, solninfo, lookup):
        soln_op = {}
        for k in solninfo:
            etype,part = lookup[k].split('_')
            key = f'{self.dataprefix}_{etype}_{part}'
            nspts = self.soln[key].shape[0]
            # Operator
            soln_op[k] = self._get_soln_op(etype, nspts)
        return soln_op

    def load_ptsinfo(self):
        f = h5py.File('probes.m','r')
        lookup = {}
        for i in f:
            if i == 'trank':
                trank = np.array(f[i])
            if i == 'eid':
                eid = np.array(f[i])
            if i == 'ploc':
                ploc = np.array(f[i])
            if i == 'op':
                op = np.array(f[i])
            if i == 'lookup':
                for id in np.array(f[i]):
                    idx = int(np.array(f[f'{i}/{id}']))
                    lookup[idx] = id
        f.close()

        solninfo = defaultdict(list)
        intop = defaultdict(list)
        ploc = defaultdict(list)
        for itrank, ieid, iploc, iop in zip(trank, eid, ploc, op):
            solninfo[itrank].append(ieid)
            intop[itrank].append(iop)
            ploc[itrank].append(ploc)

        return solninfo, intop, ploc, lookup

        #lookup = [(v.split('_')[0], v.split('_')[1]) for k,v in lookup.items()]
        #return trank, lookup, [(itrank, ieid, iop) for itrank, ieid, iop
        #                                            in zip(trank, eid, op)]


    def dump_to_file(self, ptsinfo):
        f = h5py.File(f'{self.dir}/probes.m','w')
        if self.exactloc == True:
            trank = []
            eid = []
            ploc = []
            op = []
            for _trank, _eid, _ploc, _op in ptsinfo:
                trank.append(_trank)
                eid.append(_eid)
                ploc.append(_ploc)
                op.append(_op)

            f[f'trank'] = np.array(trank)
            f[f'eid'] = np.array(eid)
            f[f'ploc'] = np.array(ploc)
            f[f'op'] = np.array(op)
            for id, (etype, part) in enumerate(self.lookup):
                f[f'lookup/{etype}_{part}'] = id
        f.close()


    def procpts(self, pts):

        mesh, mesh_trans = self.data_process('bg')

        outpts = list()

        # Input to algorithm to do finding closest point process
        closest = self._closest_pts(mesh, pts)
        del mesh

        # Sample points we're responsible for, grouped by element type + part
        elepts = [[] for i in range(len(self.mesh_inf))]

        # If we are looking for the exact locations
        if self.exactloc:
            # For each sample point find our nearest search location
            for i, (dist, trank, (uidx, eidx)) in enumerate(closest):
                elepts[trank].append((i, eidx, mesh_trans[trank][uidx]))
            del mesh_trans

            #raise NotImplementedError('refinement needs some process before output')
            # Refine
            #outpts.append(self._refine_pts(elepts, pts).reshape(nupts,
            #                                neles, self.nvars).swapaxes(1,2))

            ourpts = self._refine_pts(elepts, pts)

            # Perform the sampling and interpolation
            #samples = [op @ solns[et][:, :, ei] for et, ei, _, op in self._ourpts]
            return ourpts
        else:
            # For each sample point find our nearest search location
            for i, (dist, trank, (uidx, eidx)) in enumerate(closest):
                elepts[trank].append((i, eidx, uidx, trank))

            return elepts

    def load_snapshot(self, name):
        soln = defaultdict()
        f = h5py.File(name, 'r')
        for i in f:
            kk = i.split('_')
            if self.dataprefix in kk:
                soln[i] = np.array(f[i])

        f.close()

        return soln

    def _closest_pts(self, epts, pts):
        # Use brute force to find closest pts
        yield from self._closest_pts_bf(epts, pts)
        """More methods are on the way"""

    def _closest_pts_bf(self, epts, pts):
        for p in pts:
            # Compute the distances between each point and p
            dists = [np.linalg.norm(e - p, axis=2) for e in epts]

            # Get the index of the closest point to p for each element type and mpi rank
            amins = [np.unravel_index(np.argmin(d), d.shape) for d in dists]

            # Dereference to get the actual distances
            dmins = [d[a] for d, a in zip(dists, amins)]

            # Find the minimum across all element types and mpi ranks
            yield min(zip(dmins, range(len(epts)), amins))



    def _refine_pts(self, elepts, pts):
        # Use visualization points to do refinement to improve stability of the code
        elelist = self.data_process('vis', elepts)
        ptsinfo = []

        # Mapping from nupts to element type
        ordplusone = self.order + 1
        etype_map = {
            ordplusone**2: 'quad',
            ordplusone*(ordplusone+1)/2: 'tri',
            ordplusone**3: 'hex',
            ordplusone**2*(ordplusone+1)/2: 'pri',
            ordplusone*(ordplusone+1)*(2*ordplusone+1)/6: 'pyr',
            ordplusone*(ordplusone+1)*(ordplusone+2)/6: 'tet'
        }

        # Get basis class map
        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}

        # Loop over all the points for each element type
        for etype, (eles, epts) in enumerate(zip(elelist, elepts)):
            if not epts:
                continue

            idx, eidx, tlocs = zip(*epts)
            spts = eles[:, eidx, :]
            eletype = etype_map.get(len(spts))
            plocs = [pts[i] for i in idx]

            # Use Newton's method to find the precise transformed locations
            basis = basismap[eletype](len(spts), self.cfg)
            ntlocs, nplocs = self._plocs_to_tlocs(basis.sbasis, spts, plocs,
                                             tlocs)

            # Form the corresponding interpolation operators
            ops = basis.ubasis.nodal_basis_at(ntlocs)

            # Do interpolation on the corresponding elements
            #new_sln = np.einsum('ij,jki -> ik', ops, soln[:, :, eidx])
            # Append index and solution of each point
            #ptsinfo.extend(info for info in zip(idx, new_sln))

            # Append to the point info list
            ptsinfo.extend(
                (*info, etype) for info in zip(idx, eidx, nplocs, ops)
            )

        # Resort to the original index
        ptsinfo.sort()

        # Strip the index, move etype to the front, and return
        return [(etype, *info) for idx, *info, etype in ptsinfo]
        #return np.array([new_sln for idx, new_sln in ptsinfo])


    def _plocs_to_tlocs(self, sbasis, spts, plocs, tlocs):
        plocs, itlocs = np.array(plocs), np.array(tlocs)

        # Set current tolerance
        tol = 10e-12

        # Evaluate the initial guesses
        iplocs = np.einsum('ij,jik->ik', sbasis.nodal_basis_at(itlocs), spts)

        # Iterates
        kplocs, ktlocs = iplocs.copy(), itlocs.copy()

        # Apply maximum ten iterations of Newton's method
        for k in range(10):
            # Get Jacobian operators
            jac_ops = sbasis.jac_nodal_basis_at(ktlocs)
            # Solve from ploc to tloc
            kjplocs = np.einsum('ijk,jkl->kli', jac_ops, spts)
            ktlocs -= np.linalg.solve(kjplocs, kplocs - plocs)
            # Transform back to ploc
            ops = sbasis.nodal_basis_at(ktlocs)
            np.einsum('ij,jik->ik', ops, spts, out=kplocs)

            # Apply check routine
            kdists = np.linalg.norm(plocs - kplocs, axis=1)
            index = np.where(kdists > tol)[0]
            if len(index) == 0:
                break
            if k == 9:
                """Currently only precise location is acceptable"""
                raise RuntimeError(f'warning: failed to apply Newton Method, tol: {tol}, dist: {kdists}')
                # Compute the initial and final distances from the target location
                idists = np.linalg.norm(plocs - iplocs, axis=1)

                # Replace any points which failed to converge with their initial guesses
                closer = np.where(idists < kdists)
                ktlocs[closer] = itlocs[closer]
                kplocs[closer] = iplocs[closer]


        return ktlocs, kplocs





    def sort_time_series(self, elepts, time):
        lookupid = defaultdict(list)
        Npts = 0
        for epts, lookup in zip(elepts,self.lookup):
            if not epts:
                continue
            i, eidx, uidx, etype = zip(*epts)

            lookupid[f'{lookup[0]}_{lookup[1]}'].append(uidx)
            lookupid[f'{lookup[0]}_{lookup[1]}'].append(eidx)
            Npts += len(eidx)

        # Get information from solution field
        soln_pts = np.zeros([Npts,self.nvars,len(time)])

        for t in range(len(time)):
            print(t)
            soln = f'{self.solndir}{time[t]}.pyfrs'
            soln = self.load_snapshot(soln)
            sln = []
            for kdx, idx in lookupid.items():
                name = f'{self.dataprefix}_{kdx}'

                #print(idx)
                if len(sln) > 0:
                    sln =  np.append(sln, soln[name][idx[0],:,idx[1]].reshape(-1,self.nvars), axis = 0)
                else:
                    sln = soln[name][idx[0],:,idx[1]].reshape(-1,self.nvars)

            soln_pts[...,t] = sln

        return soln_pts












    def _flash_to_disk(self, outpts):
        f = h5py.File('soln_pts.s','w')
        for pts in range(len(outpts)):
            f[f'soln_{pts}'] = outpts[pts]
        f.close()




    #"""
    def _flash_to_disk(self, outpts):
        # Prepare data for output
        self.cfg.set('solver','order',self.nwmsh_ord)

        import h5py

        f = h5py.File(f'{self.dir_name}','w')
        f['config'] = self.cfg.tostr()
        f['mesh_uuid'] = self.uuid
        f['stats'] = self.stats.tostr()

        for id, (etype, part) in enumerate(self.mesh_new_info):
            name = f'{self.dataprefix}_{etype}_{part}'
            f[name] = outpts[id]
        f.close()
    #"""
