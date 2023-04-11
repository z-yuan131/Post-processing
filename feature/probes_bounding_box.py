from base import Base

import numpy as np
from collections import defaultdict
import h5py

from pyfr.readers.native import NativeReader
from pyfr.quadrules import get_quadrule
from pyfr.util import subclasses, subclass_where
from pyfr.shapes import BaseShape
from pyfr.mpiutil import get_comm_rank_root #, mpi

import matplotlib.pyplot as plt

class Probes(Base):
    name = None
    def __init__(self, argv, icfg, fname):
        super().__init__(argv)
        self.exactloc = icfg.getbool(fname, 'exactloc', True)
        # List of points to be sampled and format
        self.fmt = icfg.get(fname, 'format', 'primitive')
        self.mode = icfg.get(fname, 'mode', 'mesh')
        self.porder = icfg.getint(fname, 'porder', self.order)
        self.nmsh_dir = icfg.get(fname, 'new-mesh-dir')

        if not self.nmsh_dir:
            raise ValueError('Directory and file name is needed for the target mesh')

        if self.exactloc == False:
            raise NotImplementedError('not finished yet, use exact location instead.')


    def preprocpts_elewise(self):
        # Use a strictly interior point set
        qrule_map = {
            'quad': 'gauss-legendre',
            'tri': 'williams-shunn',
            'hex': 'gauss-legendre',
            'pri': 'williams-shunn~gauss-legendre',
            'pyr': 'gauss-legendre',
            'tet': 'shunn-ham'
        }
        f = h5py.File(self.nmsh_dir, 'r')
        self.kshp, mesh, nele = [], [], []
        comm, rank, root = get_comm_rank_root()
        for k,v in f.items():
            if 'spt' in k.split('_'):
                if f'p{rank}' == k.split('_')[-1]:
                    dim = v.shape[-1]
                    # Raise mesh to aimed polynomial order and interpolate it to upts
                    etype = k.split('_')[1]
                    nspts = v.shape[0]
                    nupts = self._get_npts(etype, self.porder+1)
                    upts = get_quadrule(etype, qrule_map[etype], nupts).pts
                    mesh_op = self._get_ops_interp(nspts, etype, upts, nupts, self.porder)
                    vv = np.einsum('ij, jkl -> ikl',mesh_op,v)
                    try:
                        self.pts = np.append(vv.reshape(-1, dim), self.pts, axis = 0)
                    except:
                        self.pts = vv.reshape(-1, dim)
                    self.kshp.append((k, vv.shape))

        f.close()
        print(rank, len(self.pts))

    def preprocpts_ptswise(self):
        # Load pts file
        pts = np.loadtxt(self.nmsh_dir, delimiter =' ')

        # maunpulate some points
        #"""
        bdpts = np.array([[86.59443292, -1.86316528, -3.0001]])
        for bdpt in bdpts:
            index = np.argmin(np.linalg.norm(pts - bdpt, axis=1))
            cloest = np.argsort(np.linalg.norm(pts - pts[index], axis=1))[1]
            pts[index] = (pts[index] + pts[cloest])/2
        #"""

        # Extrude in span direction
        z = np.array([0,0,11.999])
        z0 = np.array([0,0,0.0001])
        pts = np.array([np.linspace(pt -z0, pt - z, 200) for pt in pts]).reshape(-1, self.ndims)

        comm, rank, root = get_comm_rank_root()
        size = comm.Get_size()
        # Let each rank get a batch of points
        npts_rank = len(pts) // size
        if rank == size - 1:
            self.pts = pts[rank*npts_rank:]
        else:
            self.pts = pts[rank*npts_rank:(rank+1)*npts_rank]

        #if rank == 3:
        #    self.pts += np.array([0.001, 0.0001, 0])

        self.kshp = list(('hex', pts.shape))
        print(rank, len(self.pts))


    def mainproc(self):
        # Prepare for MPI process
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Only root rank is enough to process data
        if self.mode == 'mesh':
            self.preprocpts_ptswise()
            lookup, gbox, lbox = self.preprocmesh()
            self.procpts(lookup, gbox, lbox)
            return 0

        else:
            # Load new mesh informations
            #if rank == 0:
            #    self.nmesh = NativeReader(self.nmsh_dir)
            # Load interpolation info
            ptsinfo, lookup = self.load_ptsinfo()
            # Check the exact location
            ptsinfo = self._refine_pts(ptsinfo, lookup)



            # Process information
            eidinfo, intopinfo, soln_op, plocs = self.procinf(ptsinfo, lookup)
            pts = comm.gather(plocs, root=0)
            if rank == 0:
                pts = np.concatenate(pts, axis = 0)
                self.dump_to_h5_ptswise(pts)

            for t in self.time:
                print(rank, t)
                self.procsoln_ptswise(t, eidinfo, intopinfo, lookup, soln_op)

    def procsoln_ptswise(self, time, eidinfo, intops, lookup, soln_op):
        soln = f'{self.solndir}{time}.pyfrs'
        comm, rank, root = get_comm_rank_root()
        for i in range(comm.Get_size()):
            if rank == i:
                soln = self._load_snapshot(soln, eidinfo, lookup, soln_op)
            comm.barrier()

        sln = [intops[k][id] @ soln[k][...,id] for k in intops for id in range(len(intops[k]))]

        # Make it primitive varibles
        if self.fmt == 'primitive':
            sln = np.array(sln).swapaxes(0,-1)
            sln = np.array(self._con_to_pri(sln)).swapaxes(0,-1)

        # Gather and reshape to original shape
        soln = comm.gather(sln, root=0)

        if rank == 0:
            soln = np.concatenate(soln, axis = 0)
            self.dump_to_h5_ptswise(soln, time)

    def procsoln_elewise(self, time, eidinfo, intops, lookup, soln_op):
        soln = f'{self.solndir}{time}.pyfrs'
        soln = self._load_snapshot(soln, eidinfo, lookup, soln_op)

        sln = [intops[k][id] @ soln[k][...,id] for k in intops for id in range(len(intops[k]))]

        # Reshape to original shape
        #print(self.kshp)
        tNpt, soln = 0, {}
        for k, shp in self.kshp:
            Npt = shp[0] * shp[1]
            soln[k] = np.array(sln)[tNpt:tNpt+Npt].reshape(shp[0], shp[1], self.nvars)
            tNpt += Npt

        self.dump_to_h5(soln, time)


    def _load_snapshot(self, name, eidinfo, lookup, soln_op):
        soln = defaultdict()
        f = h5py.File(name,'r')
        for k in soln_op:
            etype,part = lookup[k].split('_')[1:]
            key = f'{self.dataprefix}_{etype}_{part}'
            sln = np.array(f[key])[...,eidinfo[k]]
            sln = np.einsum('ij, jkl -> ikl',soln_op[k],sln)
            try:
                soln[k] = np.append(soln[k], sln, axis = -1)
            except KeyError:
                soln[k] = sln
        f.close()
        return soln

    def dump_to_h5_ptswise(self, var, time = []):
        if len(time) == 0:
            f = h5py.File(f'{self.dir}/interp.pyfrs','w')
            f['mesh'] = var
        else:
            f = h5py.File(f'{self.dir}/interp.pyfrs','a')
            f[f'soln_{time}'] = var
        f.close()

    def dump_to_h5_elewise(self, soln, time):
        # Get mpi info
        comm, rank, root = get_comm_rank_root()

        if rank == 0:
            f = h5py.File(f'{self.dir}/interp_{time}.pyfrs','w')
            f['mesh_uuid'] = self.nmesh['mesh_uuid']
            self.cfg.set('solver','order',self.porder)
            f['config'] = self.cfg.tostr()
            f['stats'] = self.stats.tostr()
            f.close()

        for i in range(comm.Get_size()):
            if i == rank:

                f = h5py.File(f'{self.dir}/interp_{time}.pyfrs','a')

                for k, v in soln.items():
                    # Make it primitive varibles
                    if self.fmt == 'primitive':
                        v = v.swapaxes(0,-1)
                        v = np.array(self._con_to_pri(v)).swapaxes(0,-1)
                    prefix, etype, part = k.split('_')
                    f[f'soln_{etype}_{part}'] = v.swapaxes(1,-1)
                f.close()
            comm.Barrier()


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
            etype,part = lookup[k].split('_')[1:]
            key = f'{self.dataprefix}_{etype}_{part}'
            nspts = self.soln[key].shape[0]
            # Operator
            soln_op[k] = self._get_soln_op(etype, nspts)
        return soln_op



    def _refine_pts(self, elepts, lookup):
        # Mapping from nupts to element type
        etype_map = lambda x: {
            (x+1)**2: 'quad',
            (x+1)*(x+2)/2: 'tri',
            (x+1)**3: 'hex',
            (x+1)**2*(x+2)/2: 'pri',
            (x+1)*(x+2)*(2*(x+1)+1)/6: 'pyr',
            'tet': (x+1)*(x+2)*(x+3)/6,
            'pyr': (x+1)*(x+2)*(2*(x+1)+1)/6,
            'hex': (x+1)**3,
            'pri': (x+1)**2*(x+2)/2
        }
        # Use strict interior points
        qrule_map = {
            'quad': 'gauss-legendre',
            'tri': 'williams-shunn',
            'hex': 'gauss-legendre',
            'pri': 'williams-shunn~gauss-legendre',
            'pyr': 'gauss-legendre',
            'tet': 'shunn-ham'
        }
        qrule_map_v = {
            'quad': 'gauss-legendre',
            'tri': 'williams-shunn',
            'hex': 'gauss-legendre-lobatto',
            'pri': 'alpha-opt~gauss-legendre-lobatto', #'williams-shunn~gauss-legendre-lobatto',
            'pyr': 'gauss-legendre-lobatto',
            'tet': 'alpha-opt'
        }

        # Get basis class map
        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}

        ptsinfo = []
        # Loop over all the points for each element type
        for erank, eptsinfo in elepts.items():
            # Get information
            pidx, eidx, plocs = zip(*eptsinfo)

            # Load relevant elements
            spts = self.data_process(eidx, lookup[erank], 'vis')

            _, etype, part = lookup[erank].split('_')

            # Highest possible polynomial order in quadrature map
            #qrule_porder_map = {'quad': 13, 'tri': 8, 'hex': 8, 'pri': 8,
            #                    'pyr': 6, 'tet': 6}

            # Update closest point in the reference domain
            #upts = get_quadrule(etype, qrule_map_v[etype], len(spts)).pts
            upts = np.array(self._get_std_ele(etype, len(spts), self.order))

            # Update element coord in physical domain
            #mesh_op_vis = self._get_vis_op(len(ispts), etype, order)
            #spts = np.einsum('ij, jkl -> ikl', mesh_op_vis, spts)
            #mesh_op = self._get_ops_interp(len(spts), etype, upts, nupts, order)
            #spts = np.einsum('ij, jkl -> ikl', mesh_op, spts)

            # Update closest points index
            #uidx = [np.argsort(np.linalg.norm(spts[:,id] - plocs[id], axis = 1))[:2] for id in range(len(plocs))]
            #tlocs = np.mean(upts[np.array(uidx)], axis = 1)
            uidx = [np.argmin(np.linalg.norm(spts[:,id] - plocs[id], axis = 1)) for id in range(len(plocs))]
            tlocs = upts[uidx]

            #from pyfr.polys import get_polybasis
            #sbasis = get_polybasis(etype, order + 1, upts)
            basis = basismap[etype](len(spts), self.cfg)
            ntlocs, nplocs, index = self._plocs_to_tlocs(basis.sbasis, spts,
                                                    plocs, tlocs)

            if len(index) != 0:
                print(etype, index)
                raise RuntimeError('Newton iteration is failed')

            #index = self._check_converg(ntlocs, index, basis, tlocs)
            #idd = [*range(len(ntlocs))]
            #idd = [id for id in idd if id not in index]

            # Form the corresponding interpolation operators
            ops = basis.ubasis.nodal_basis_at(ntlocs)
            #ops = basis.ubasis.nodal_basis_at(ntlocs[idd])

            # Append to the point info list
            ptsinfo.extend(
                (*info, erank) for info in zip(pidx, eidx, plocs, ops)
            )

        # Resort to the original index
        ptsinfo.sort()

        # Strip the index, move etype to the front, and return
        return [(erank, *info) for pidx, *info, erank in ptsinfo]

    def _check_converg(self, ntlocs, index, basis, tlocs):
        # Get indeces that finished Newton-iteration
        idx = [*range(len(tlocs))]
        idx = [id for id in idx if id not in index]
        gtlocs = ntlocs[idx]

        # Check transformed position to make sure they are inside the bounding box
        #shapecls = subclass_where(Probes, name=basis.name)
        #idx = shapecls.std_ele_box(gtlocs, basis)
        idx = self.std_ele_box(gtlocs, basis)

        return np.sort(idx + index)


    def _plocs_to_tlocs(self, sbasis, spts, plocs, tlocs):
        plocs, itlocs = np.array(plocs), np.array(tlocs)

        # Set current tolerance
        """
            Note here, for those points near the element boundaries, one has to
            consider about Hessian matrix (second order derivatives).
            For convergen analysis, convergence in reference space is also
            important, L1 norm could be added for this purpose.
        """
        tol = 3e-8

        # Evaluate the initial guesses
        iplocs = np.einsum('ij,jik->ik', sbasis.nodal_basis_at(itlocs), spts)

        # Iterates
        kplocs, ktlocs = iplocs.copy(), itlocs.copy()

        # Output array
        oplocs, otlocs = iplocs.copy(), itlocs.copy()

        indexp = [*range(len(kplocs))]
        # Apply maximum ten iterations of Newton's method
        for k in range(50):
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
            # Get the points satisfied criterion in both physical and refernece domain
            index = np.where(kdists < tol)[0]
            index = [id for id in index if id in indexp]
            indexp = list(set(indexp) - set(index))

            if len(index) != 0:
                oplocs[index], otlocs[index] = kplocs[index], ktlocs[index]

            if len(indexp) == 0:
                return otlocs, oplocs, []
            if k == 49:
                print(np.max(kdists[indexp]))
                return otlocs, oplocs, indexp




    def load_ptsinfo(self):
        # Get mpi info
        comm, rank, root = get_comm_rank_root()
        f = h5py.File(f'{self.dir}/probes.m','r')

        lookup, ptsinfo, self.pts, pid = [], defaultdict(list), [], []
        for i in f:
            if i == f'{rank}':
                for key, ptid, eid, pt in np.array(f[i])[['f0','f1','f2','f3']].astype('U14, i4, i4, (3,)f8'):
                    if key not in lookup:
                        lookup.append(key)
                    ptsinfo[lookup.index(key)].append((ptid, eid, pt))
                    #self.pts.append(pt)
                    #pid.append(ptid)
                #self.kshp = np.array(f[f'{i}_info'])[['f0','f1']].astype('U14, (3,)i4').tolist()
        f.close()

        #index = np.argsort(pid)
        #self.pts = np.array(self.pts)[index]
        #print(len(self.pts))

        return ptsinfo, [str(key) for key in lookup]

    def preprocmesh(self):
        # In this section, bounding boxes will be created
        lookup, gbox, lbox = [], [], []
        for k, v in self.mesh.items():
            if 'spt' in k.split('_'):
                lookup.append(k)
                gbox.append([(np.min(v[...,i]),np.max(v[...,i])) for i in range(self.ndims)])
                lbox.append([(np.min(v[...,i], axis = 0),np.max(v[...,i], axis = 0)) for i in range(self.ndims)])

        return lookup, gbox, lbox

    def procpts(self, lookup, gbox, lbox):
        # Get mpi info
        comm, rank, root = get_comm_rank_root()

        # Get global location
        pid, gid = self._global_check(gbox)

        # Get local location
        eleinfo = self._local_check(pid, lbox, lookup)

        # Load corresponding mesh into memory
        ptsinfo, mesh_trans, lookup = self.data_process(eleinfo, lookup)

        # Input to algorithm to do finding the ownership
        ptsinfo = self._owner_check(ptsinfo, lookup, mesh_trans)

        # Write interpolation information to disk
        self.dump_to_file(ptsinfo, lookup)



    def dump_to_file(self, ptsinfo, lookup):
        # Get mpi info
        comm, rank, root = get_comm_rank_root()
        pts = []
        for erank, v in ptsinfo.items():
            for ptid, eid in v:
                pts.append((lookup[erank], ptid, eid, self.pts[ptid]))

        pts = np.array(pts, dtype='S14, i4, i4, (3,)f8')

        for i in range(comm.Get_size()):
            if i == rank:
                if i == 0:
                    f = h5py.File(f'{self.dir}/probes.m','w')
                else:
                    f = h5py.File(f'{self.dir}/probes.m','a')
                f[f'{rank}'] = pts
                #f[f'{rank}_info'] = np.array(self.kshp, dtype='S, (3,)i4')
                f.close()
            comm.barrier()

    def _owner_check(self, pts, lookup, mesh_trans):
        # Get basis class map
        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}

        ptsinfo, eleinfo, pidinfo = defaultdict(list), defaultdict(list), defaultdict(list)
        #classfy it with erank
        for pid, info in pts.items():
            p = self.pts[pid]
            for erank, e, eidx in info:


                # Compute the distances between each point and p
                dists = np.linalg.norm(e - p, axis=-1)
                for ide, dist in enumerate(dists.T):

                    # Get the index of the closest point to p for each element
                    idp = np.unravel_index(np.argmin(dist), dist.shape)

                    spts, tloc = e[:,ide], mesh_trans[erank][idp]
                    eleinfo[erank].append((spts, tloc, p, spts[idp]))
                    pidinfo[erank].append((pid, eidx[ide]))

        pt_temp = []
        for erank, info in eleinfo.items():
            spts, tlocs, ipts, iplocs = zip(*info)
            pid = pidinfo[erank]

            spts, iplocs = np.array(spts).swapaxes(0,1), np.array(iplocs)
            ipts = np.array(ipts)

            # Relevent classes and basises
            etype = lookup[erank].split('_')[1]
            basis = basismap[etype](len(spts), self.cfg)

            idx = self._newton_check(spts, tlocs, ipts, iplocs, basis)
            for id, (pid, eid) in enumerate(pidinfo[erank]):
                if id in idx and pid not in pt_temp:
                    ptsinfo[erank].append((pid, eid))
                    pt_temp.append(pid)

        # Check if all points have gone through this procedure
        comm, rank, root = get_comm_rank_root()
        for id in pts:
            if id not in pt_temp:
                print(rank, id, self.pts[id])
                raise RuntimeError('No suitbale ownership')

        return ptsinfo


    def _newton_check(self, spts, tlocs, pts, iplocs, basis):
        sbasis = basis.sbasis
        # Get Jacobian operators
        jac_ops = sbasis.jac_nodal_basis_at(tlocs)

        # Solve from ploc to tloc
        kjplocs = np.einsum('ijk,jkl->kli', jac_ops, spts)

        tlocs -= np.linalg.solve(kjplocs, iplocs - pts)

        # Apply check routine: all points out of bounding box will be thrown away
        return self.std_ele_box(tlocs, basis)

    def std_ele_box(self, tlocs, basis):
        tol = 1e-8
        kind, proj, norm = zip(*basis.faces)

        tlocs = tlocs[:,None,:] - np.array(norm)
        inner = np.einsum('ijk,jk -> ij', tlocs, np.array(norm))

        return [id for id in range(len(inner)) if all(inner[id] < tol)]


    def _local_check(self, pid, lbox, lookup):
        #ptsinfo, eleinfo = defaultdict(list), defaultdict(list)
        ptsinfo, eleinfo = [], defaultdict(list)
        for id, gid in pid.items():
            for gidx in gid:
                llbox = lbox[gidx]
                index = np.arange(len(llbox[0][0]))
                for did, (amin, amax) in enumerate(llbox):
                    idx = np.argsort(amin[index])
                    idxm = np.searchsorted(amin[index], self.pts[id,did], sorter = idx, side = 'right')
                    idx = idx[:idxm]
                    idy = np.argsort(amax[index[idx]])
                    idxm = np.searchsorted(amax[index[idx]], self.pts[id,did], sorter = idy, side = 'left')
                    idy = idy[idxm:]
                    index = index[idx[idy]]

                    if len(index) == 0:
                        break

                if len(index) != 0:
                    ptsinfo.append(id)
                    eleinfo[gidx].append((id,index))

            if id not in ptsinfo:
                print(self.pts[id])
                raise RuntimeError

            # For checking purpose
            #if id == 100:
            #    break


        return eleinfo



    def _global_check(self, gbox):
        pid, gid = defaultdict(list), []
        for id, p in enumerate(self.pts):
            for idg, rg in enumerate(gbox):
                if p[0] >= rg[0][0] and p[0] <= rg[0][1]:
                    if p[1] >= rg[1][0] and p[1] <= rg[1][1]:
                        if p[2] >= rg[2][0] and p[2] <= rg[2][1]:
                            pid[id].append(idg)
                            if idg not in gid:
                                gid.append(idg)
        return pid, gid








    def data_process(self, eleinfo, lookup, op='bg'):
        # Use a strictly interior point set
        qrule_map = {
            'quad': 'gauss-legendre',
            'tri': 'williams-shunn',
            'hex': 'gauss-legendre',
            'pri': 'williams-shunn~gauss-legendre',
            'pyr': 'gauss-legendre',
            'tet': 'shunn-ham'
        }

        qrule_map_v = {
            'quad': 'gauss-legendre',
            'tri': 'williams-shunn',
            'hex': 'gauss-legendre-lobatto',
            'pri': 'alpha-opt~gauss-legendre-lobatto', #'williams-shunn~gauss-legendre-lobatto',
            'pyr': 'gauss-legendre-lobatto',
            'tet': 'alpha-opt'
        }

        if op == 'bg':

            mesh, mesh_trans, lookup_update = [], [], []

            for erank, index in eleinfo.items():
                key = lookup[erank]
                _,etype, part = key.split('_')
                # Get Operators
                soln_name = f'{self.dataprefix}_{etype}_{part}'
                nupts = self.soln[soln_name].shape[0]
                nspts = self.mesh[key].shape[0]
                upts = get_quadrule(etype, qrule_map_v[etype], nupts).pts
                mesh_op = self._get_ops_interp(nspts, etype, upts, nupts, self.order)
                mesh_op = self._get_vis_op(nspts, etype, self.order)

                mesh_temp = np.einsum('ij, jkl -> ikl',mesh_op,self.mesh[key])

                mesh.append([(id, mesh_temp[:,idx], idx) for id, idx in index])

                #vpts = get_quadrule(etype, qrule_map_v[etype], nupts).pts
                mesh_trans.append(upts)
                lookup_update.append(lookup[erank])

            # Strip mesh indices and move pid into the front
            ptsinfo = defaultdict(list)
            for erankid, info in enumerate(mesh):
                for id, msh, idx in info:
                    ptsinfo[id].append((erankid, msh, idx))

            # Sort ptsinfo
            ptsinfo = dict(sorted(ptsinfo.items()))
            #print(ptsinfo.keys())
            #raise RuntimeError
            return ptsinfo, mesh_trans, lookup_update

        elif op == 'vis':
            mesh = self.mesh[lookup][:,eleinfo]
            etype = lookup.split('_')[1]
            nspts = mesh.shape[0]
            # Operator
            mesh_op_vis = self._get_vis_op(nspts, etype, self.order)
            return np.einsum('ij, jkl -> ikl',mesh_op_vis,mesh)

        elif op == 'vis2':
            mesh = self.mesh[lookup][:,eleinfo]
            return mesh

        elif op == 'sln':

            mesh = self.mesh[lookup][:,eleinfo]
            etype, part = lookup.split('_')[1:]
            nspts = mesh.shape[0]
            # Operator
            soln_name = f'{self.dataprefix}_{etype}_{part}'
            nupts = self.soln[soln_name].shape[0]
            upts = get_quadrule(etype, qrule_map[etype], nupts).pts
            mesh_op = self._get_ops_interp(nspts, etype, upts, nupts, self.order)
            return np.einsum('ij, jkl -> ikl',mesh_op,mesh)

        elif op == 'std':

            etype, part = lookup.split('_')[1:]
            # Operator
            soln_name = f'{self.dataprefix}_{etype}_{part}'
            nupts = self.soln[soln_name].shape[0]
            upts = np.array(self._get_std_ele(etype, nupts, self.order))
            return upts[np.array(eleinfo)]

        elif op == 'std2':

            etype, part = lookup.split('_')[1:]
            # Operator
            soln_name = f'{self.dataprefix}_{etype}_{part}'
            nupts = self.soln[soln_name].shape[0]
            upts = get_quadrule(etype, qrule_map_v[etype], nupts).pts
            return upts[np.array(eleinfo)]

class HexShape(Probes):
    name = 'hex'

    def std_ele_box(tlocs, basis):
        idx0 = np.where(tlocs[:,0] < -1)[0]
        idx1 = np.where(tlocs[idx0,0] > 1)[0]
        idx = idx0[idx1]
        idy0 = np.where(tlocs[idx,1] < -1)[0]
        idy1 = np.where(tlocs[idx[idy0],1] > 1)[0]
        idy = idx[idy0[idy1]]
        idz0 = np.where(tlocs[idy,2] < -1)[0]
        idz1 = np.where(tlocs[idy[idz0],2] > 1)[0]
        return idy[idz0[idy1]]

class TetShape(Probes):
    name = 'tet'

    def std_ele_box(tlocs, basis):
        kind, proj, norm = zip(*basis.faces)
        #print(tlocs.shape, np.array(norm).shape)
        # Point - fcenter
        tlocs = tlocs[:,None,:] - np.array(norm)
        inner = np.einsum('ijk,jk -> ij', tlocs, np.array(norm))

        #print(len([id for id in range(len(inner)) if any(inner[id] > 0)]))
        return [id for id in range(len(inner)) if any(inner[id] > 0)]


class PyrShape(Probes):
    name = 'pyr'
