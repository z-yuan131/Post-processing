from base import Base

import numpy as np
from collections import defaultdict
import h5py

from pyfr.readers.native import NativeReader
from pyfr.quadrules import get_quadrule
from pyfr.util import subclasses
from pyfr.shapes import BaseShape
from pyfr.mpiutil import get_comm_rank_root, mpi

class Probes(Base):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv)
        self.exactloc = icfg.getbool(fname, 'exactloc', True)
        npts = icfg.getliteral(fname, 'npts', [1])
        # List of points to be sampled and format
        pointA = icfg.getliteral(fname, 'samp-ptsA')
        pointB = icfg.getliteral(fname, 'samp-ptsB', None)
        self.fmt = icfg.get(fname, 'format', 'primitive')

        self.mode = icfg.get(fname, 'mode', 'mesh')

        self.preprocpts(pointA, pointB, npts)
        print(len(self.pts))
        if self.exactloc == False:
            raise NotImplementedError('not finished yet, use exact location instead.')

    def preprocpts(self, pta, ptb, npts):
        if ptb == None:
            self.pts = [np.array(pt) for pt in pta]
        else:
            if len(pta) != len(ptb):
                raise RuntimeError('Length of points lists should match.')
            self.pts = []
            for id in range(len(pta)):
                lt = [np.linspace(pta[id][i],ptb[id][i], npts[id])
                                                for i in range(self.ndims)]
                if self.ndims == 3:
                    self.pts += [np.array([x,y,z]) for x,y,z in zip(*lt)]
                else:
                    self.pts += [np.array([x,y]) for x,y in zip(*lt)]

    def data_process(self, option, argv = []):
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
            lookup = list()

            comm, rank, root = get_comm_rank_root()
            size = comm.Get_size()
            if size > len(list(self.mesh.keys())):
                raise RuntimeError('Use less mpi ranks.')

            if rank == 0:
                kk = [key for key in self.mesh if 'spt' in key.split('_')]
            else:
                kk = None
            kk = comm.bcast(kk, root = 0)
            from itertools import cycle
            for r, k in zip(cycle(np.arange(size)), kk):
                if r == rank:
                    _,etype,part = k.split('_')
                    # create lookup dictionary
                    lookup.append((etype,part))

            for etype,part in lookup:
                # Get Operators
                soln_name = f'{self.dataprefix}_{etype}_{part}'
                key = f'spt_{etype}_{part}'
                nupts = self.soln[soln_name].shape[0]
                nspts = self.mesh[key].shape[0]
                upts = get_quadrule(etype, qrule_map[etype], nupts).pts
                mesh_op = self._get_ops_interp(nspts, etype, upts, nupts, self.order)

                mesh.append(np.einsum('ij, jkl -> ikl',mesh_op,self.mesh[key]))
                mesh_trans.append(upts)

            return mesh, mesh_trans, lookup

        elif option == 'bg2':
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
            for id, key in enumerate(argv):
                etype, part = key.split('_')
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
            ptsinfo = self.procpts(self.pts, comm)
            return 0
        else:
            if len(self.time) < size:
                raise RuntimeError('The number of Mpi rank is more than time series.')
            if rank == 0:
                 ptsinfo, lookup = self.load_ptsinfo()
                 if self.exactloc:
                     ptsinfo = self._refine_pts(ptsinfo, lookup)
            else:
                ptsinfo = None
                lookup = None

        # Boardcast pts and eles information
        ptsinfo = comm.bcast(ptsinfo, root=0)
        lookup = comm.bcast(lookup, root=0)

        comm.barrier()

        # Get time series
        time = self.get_time_series_mpi(rank, size)

        # Process information
        eidinfo, intopinfo, soln_op, plocs = self.procinf(ptsinfo, lookup)
        pts = {}
        for t in time:
            pts[t] = self.procsoln(t, eidinfo, intopinfo, lookup, soln_op)

        # Collect all points
        pts = comm.gather(pts, root = 0)

        # Dump to h5 file
        if rank == 0:
            self.dump_to_h5(pts, plocs)

    def dump_to_h5(self, pts, ploc):
        soln = np.zeros([len(self.time), len(self.pts), self.nvars])
        for tpt in pts:
            for t, pt in tpt.items():
                index = self.time.index(t)
                soln[index] = np.array(pt)

        # Make it primitive varibles
        if self.fmt == 'primitive':
            soln = soln.swapaxes(0,-1)
            soln = np.array(self._con_to_pri(soln)).swapaxes(0,-1)
        f = h5py.File(f'{self.dir}/probes.s','w')
        f['soln'] = soln
        f['pts'] = np.array(ploc)
        f.close()

    def _con_to_pri(self, cons):
        rho, E = cons[0], cons[-1]

        # Divide momentum components by rho
        vs = [rhov/rho for rhov in cons[1:-1]]

        # Compute the pressure
        gamma = self.cfg.getfloat('constants', 'gamma')
        p = (gamma - 1)*(E - 0.5*rho*sum(v*v for v in vs))

        return [rho] + vs + [p]

    def procinf(self, ptsinfo, lookup):
        # Process information
        kids, eids, plocs, intops = zip(*ptsinfo)
        eidinfo = defaultdict(list)
        intopinfo = defaultdict(list)
        for kid, eid, intop in zip(kids, eids, intops):
            eidinfo[kid].append(eid)
            intopinfo[kid].append(intop)

        soln_op = self._get_op_sln(kids, lookup)
        return eidinfo, intopinfo, soln_op, plocs

    def _get_op_sln(self, kids, lookup):
        kids = list(set(kids))
        soln_op = {}
        for k in kids:
            etype,part = lookup[k].split('_')
            key = f'{self.dataprefix}_{etype}_{part}'
            nspts = self.soln[key].shape[0]
            # Operator
            soln_op[k] = self._get_soln_op(etype, nspts)
        return soln_op

    def procsoln(self, time, eidinfo, intops, lookup, soln_op):
        soln = f'{self.solndir}{time}.pyfrs'
        soln = self._load_snapshot(soln, eidinfo, lookup, soln_op)

        return [intops[k][id] @ soln[k][...,id] for k in intops for id in range(len(intops[k]))]




    def _load_snapshot(self, name, eidinfo, lookup, soln_op):
        soln = defaultdict()
        f = h5py.File(name,'r')
        for k in soln_op:
            etype,part = lookup[k].split('_')
            key = f'{self.dataprefix}_{etype}_{part}'
            sln = np.array(f[key])[...,eidinfo[k]]
            sln = np.einsum('ij, jkl -> ikl',soln_op[k],sln)
            try:
                soln[k] = np.append(soln[k], sln, axis = -1)
            except KeyError:
                soln[k] = sln
        f.close()
        return soln

    def _refine_pts(self, ptsinfo, lookup):
        # Use visualization points to do refinement to improve stability of the code
        elelist = self.data_process('vis', lookup)

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

        ptsinfon = []
        # Loop over all the points for each element type
        for etype, (eles, epts) in enumerate(zip(elelist, ptsinfo)):

            idx, eidx, tlocs = zip(*epts)
            spts = eles[:, eidx, :]
            eletype = etype_map.get(len(spts))
            plocs = [self.pts[i] for i in idx]

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
            ptsinfon.extend(
                (*info, etype) for info in zip(idx, eidx, nplocs, ops)
            )

        # Resort to the original index
        ptsinfon.sort()

        # Strip the index, move etype to the front, and return
        return [(etype, *info) for idx, *info, etype in ptsinfon]
        #return np.array([new_sln for idx, new_sln in ptsinfo])


    def _plocs_to_tlocs(self, sbasis, spts, plocs, tlocs):
        plocs, itlocs = np.array(plocs), np.array(tlocs)

        # Set current tolerance
        tol = 5e-9

        # Evaluate the initial guesses
        iplocs = np.einsum('ij,jik->ik', sbasis.nodal_basis_at(itlocs), spts)

        # Iterates
        kplocs, ktlocs = iplocs.copy(), itlocs.copy()

        # Output array
        oplocs, otlocs = iplocs.copy(), itlocs.copy()

        indexp = [*range(len(kplocs))]
        # Apply maximum ten iterations of Newton's method
        for k in range(100):
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
            # Get the points satisfied criterion
            index = np.where(kdists < tol)[0]
            index = [id for id in index if id in indexp]
            indexp = list(set(indexp) - set(index))
            #print(indexp, kdists)
            if len(index) != 0:
                oplocs[index], otlocs[index] = kplocs[index], ktlocs[index]
            if len(indexp) == 0:
                break
            if k == 99:
                """Currently only precise location is acceptable"""
                raise RuntimeError(f'warning: failed to apply Newton Method, tol: {tol}, dist: {kdists}, loc: {plocs[indexp]}')
                # Compute the initial and final distances from the target location
                idists = np.linalg.norm(plocs - iplocs, axis=1)

                # Replace any points which failed to converge with their initial guesses
                closer = np.where(idists < kdists)
                otlocs[closer] = itlocs[closer]
                oplocs[closer] = iplocs[closer]


        return otlocs, oplocs


    def load_ptsinfo(self):
        f = h5py.File(f'{self.dir}/probes.m','r')
        eleinfo = {}
        tloc = {}
        id = {}

        for i in f:
            eleinfo[i] = np.array(f[f'{i}/eid'])
            tloc[i] = np.array(f[f'{i}/tloc'])
            id[i] = np.array(f[f'{i}/id'])
        f.close()

        ptsinfo = []
        lookup = []
        for key, eids in eleinfo.items():
            lookup.append(key)
            pts = []
            for _id, _eid, _tloc in zip(id[key], eids, tloc[key]):
                pts.append((_id, _eid, _tloc))
            ptsinfo.append(pts)

        return ptsinfo, lookup


    def procpts(self, pts, comm):
        # Get mpi info
        comm, rank, root = get_comm_rank_root()

        mesh, mesh_trans, lookup = self.data_process('bg')

        outpts = list()

        # Input to algorithm to do finding closest point process
        closest = self._closest_pts(mesh, pts)
        del mesh

        # Sample points we're responsible for, grouped by element type + part
        elepts = [[] for i in range(len(lookup))]


        # If we are looking for the exact locations
        if self.exactloc:
            # For each sample point find our nearest search location
            for i, (dist, trank, (uidx, eidx)) in enumerate(closest):
                # Reduce over the distance
                _, mrank = comm.allreduce((dist, rank), op=mpi.MINLOC)

                if rank == mrank:
                    elepts[trank].append((i, eidx, mesh_trans[trank][uidx]))
            del mesh_trans

            elepts = comm.gather(elepts, root = 0)
            lookup = comm.gather(lookup, root = 0)
            if rank == 0:
                ptsinfo = {}
                for id, elept in enumerate(elepts):
                    for pt, lp in zip(elept, lookup[id]):
                        if pt:
                            ptsinfo[lp] = pt

                self.dump_to_file(ptsinfo)
        else:
            raise NotImplementedError()
            # For each sample point find our nearest search location
            for i, (dist, trank, (uidx, eidx)) in enumerate(closest):
                elepts[trank].append((i, eidx, uidx, trank))

            return elepts

    def dump_to_file(self, ptsinfo):
        f = h5py.File(f'{self.dir}/probes.m','w')
        for key in ptsinfo:
            k = f'{key[0]}_{key[1]}'
            eid = []
            id = []
            tloc = []
            for _id, _eid, _tloc in ptsinfo[key]:
                eid.append(_eid)
                tloc.append(_tloc)
                id.append(_id)

            f[f'{k}/eid'] = np.array(eid)
            f[f'{k}/id'] = np.array(id)
            f[f'{k}/tloc'] = np.array(tloc)
        f.close()


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
