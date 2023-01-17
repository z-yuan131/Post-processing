from base import Base

import numpy as np
from collections import defaultdict
import h5py

from pyfr.readers.native import NativeReader
from pyfr.quadrules import get_quadrule

import matplotlib.pyplot as plt
#import matplotlib.tri as tri

class Region(Base):
    def __init__(self, argv):
        super().__init__(argv)
        self.layers = 20
        self.suffix = ['hex']
        self.AOA = -3
        self.boundary_name = ['wall']

        if self.suffix == None:
            self.suffix = ['quad','tri','hex','tet','pri','pyr']

        mesh_part = self.mesh.partition_info('spt')

        self.suffix_parts = np.where(np.array(mesh_part['hex']) > 0)[0]
        self.suffix_parts = [f'p{i}' for i in self.suffix_parts]
        print(self.suffix_parts)

    def get_boundary(self):
        mesh_wall = defaultdict(list)
        mesh_wall_tag = list()
        con_new = defaultdict(list)
        # Get first level wall mesh
        for key in self.mesh:
            keyword = key.split('_')
            for bname in self.boundary_name:
                if bname in keyword:

                    part = keyword[-1]
                    for etype, eid, fid, pid in self.mesh[key][['f0','f1','f2','f3']].astype('U4,i4,i1,i2'):
                        if etype in self.suffix:
                            mesh_wall[f'spt_{etype}_{part}'].append(eid)

                            # Tag all elements in the set as belonging to the first layer
                            if isinstance(mesh_wall_tag, list):
                                mesh_wall_tag = {part: {(etype, eid): 0}}
                                con_new[f'bcon_{bname}_{part}'].append((etype, eid))
                            elif part in mesh_wall_tag:
                                mesh_wall_tag[part].update({(etype, eid): 0})
                                con_new[f'bcon_{bname}_{part}'].append((etype, eid))
                            else:
                                mesh_wall_tag.update({part: {(etype, eid): 0}})
                                con_new[f'bcon_{bname}_{part}'].append((etype, eid))


        #print(np.sum([len(mesh_wall[i]) for i in mesh_wall]))
        return mesh_wall_tag, mesh_wall, con_new



    def get_wall_O_grid(self):
        # Get O-grid meshes
        mesh_wall_tag, mesh_wall, con_new = self.get_boundary()

        # For single rank process, we'd better to pre load connectivities
        con = defaultdict(list)
        pcon = {}
        for part in self.suffix_parts:

            cont, pcont = self.load_connectivity(part)

            con[part] = cont
            pcon.update(pcont)


        # For wall normal quantities, load the connectivities
        keys = list(mesh_wall.keys())
        for i in range(self.layers):

            for key in keys:
                _, etype, part = key.split('_')


                # Exchange information about recent updates to our set
                if len(pcon) > 0:
                    for p, (pc, pcr, sb) in pcon[part].items():
                        sb[:] = [mesh_wall_tag[part].get(c, -1) == i for c in pc]


                # Growing out by considering inertial partition
                for l, r in con[part]:
                    # Exclude elements which are not interested
                    if not all([r[0] in self.suffix]):
                        continue
                    if mesh_wall_tag[part].get(l, -1) == i and r not in mesh_wall_tag[part]:
                        mesh_wall_tag[part].update({r: i + 1})
                        mesh_wall[f'spt_{r[0]}_{part}'].append(r[1])
                        con_new[f'con_{part}'].append((l,r))


                    elif mesh_wall_tag[part].get(r, -1) == i and l not in mesh_wall_tag[part]:
                        mesh_wall_tag[part].update({l: i + 1})
                        mesh_wall[f'spt_{l[0]}_{part}'].append(l[1])
                        con_new[f'con_{part}'].append((l,r))

                # Grow our element set by considering adjacent partitions
                for p, (pc, pcr, sb) in pcon[part].items():
                    for l, r, b in zip(pc, pcr, sb):
                        if not all([r[0] in self.suffix]):
                            continue
                        try:
                            if b and r not in mesh_wall_tag[f'{p}']:
                                mesh_wall_tag[f'{p}'].update({r: i + 1})
                                mesh_wall[f'spt_{r[0]}_{p}'].append(r[1])
                                con_new[f'con_{part}{p}'].append(l)
                                con_new[f'con_{p}{part}'].append(r)

                        except  KeyError:
                                mesh_wall_tag.update({f'{p}': {r: i + 1}})
                                mesh_wall[f'spt_{r[0]}_{p}'].append(r[1])
                                con_new[f'con_{part}{p}'].append(l)
                                con_new[f'con_{p}{part}'].append(r)


        #print(mesh_wall.keys())
        #plt.figure()
        #for key in mesh_wall:
        #    mesh = self.mesh[key][:,mesh_wall[key],:2]
        #    plt.plot(mesh[...,0],mesh[...,1],'.')
        #plt.show()
        return mesh_wall_tag, mesh_wall, con_new

    def load_connectivity(self, part):

        # Load our inner connectivity arrays
        con = self.mesh[f'con_{part}'].T
        con = con[['f0', 'f1']].astype('U4,i4').tolist()

        pcon = {}
        # Load our partition boundary connectivity arrays
        for p in self.suffix_parts:
            try:
                pc = self.mesh[f'con_{part}{p}']
                pc = pc[['f0', 'f1']].astype('U4,i4').tolist()
                pcr = self.mesh[f'con_{p}{part}']
                pcr = pcr[['f0', 'f1']].astype('U4,i4').tolist()

            except KeyError:
                continue

            try:
                pcon[part].update({p: (pc, pcr, *np.empty((1, len(pc)), dtype=bool))})
            except KeyError:
                pcon.update({part: {p: (pc, pcr, *np.empty((1, len(pc)), dtype=bool))}})

        return con, pcon




class SpanAverage(Region):
    def __init__(self, argv):
        super().__init__(argv)
        self.tol = 1e-6

        raise NotImplementedError("Haven't finished yet")

    def reorder(self):
        mesh_wall_tag, mesh_wall, con_new = self.get_wall_O_grid()

        # Reduce/reorder connectivities

        # Collect one periodic boundary
        amin = min([np.min(self.mesh[key][...,-1]) for key in mesh_wall])
        for key in mesh_wall:
            index = np.where(abs(self.mesh[key][:,mesh_wall[key],-1] - amin) < self.tol)[1]
            index = list(set(index))

            index = np.array(mesh_wall[key])[index]
            """
            try:
                mesh_prio = np.append(mesh_prio,self.mesh[key][0,index,:2],axis = 0)
            except UnboundLocalError:
                mesh_prio = self.mesh[key][0,index,:2]
            """
            print(index)
            raise RuntimeError

        #print(mesh_prio.shape)
        #plt.figure()
        #plt.plot(mesh_prio[:,0],mesh_prio[:,-1],'.')
        #plt.show()
        #raise RuntimeError

        index = defaultdict(list)
        for key in mesh_wall:
            part = key.split('_')[-1]
            #mesh = self.mesh[key]

            # Load internal connectivities
            #con = self.mesh[f'con_{part}'].T
            #con = con[['f0', 'f1']].astype('U4,i4').tolist()

            #for l, r in con:
            #    print(l)
            #    raise RuntimeError

            # A single point of each element is enough for sorting
            tol = 1e-10
            peid = np.array(mesh_wall[key])
            mesh = self.mesh[key][0,peid]

            for id, pt in enumerate(mesh_prio):
                dists = np.linalg.norm(mesh[:,:2] - pt, axis=1)
                eids = np.where(dists < tol)[0]
                if len(eids) > 0:
                    index[id].append((key,peid[eids]))

        return index

    def average(self):
        index = self.reorder()
        soln_avg = np.zeros([(self.order+1)**3,self.nvars,len(index),len(self.time)])

        for t in range(len(self.time)):
            print(t)
            soln = f'{self.solndir}_{self.time[t]}.pyfrs'
            soln = self.load_snapshot(soln)

            for id, item in index.items():
                for idx, (key, eid) in enumerate(item):
                    _, etype, part = key.split('_')
                    name = f'{self.dataprefix}_{etype}_{part}'
                    if idx == 0:
                        length = len(eid)
                        sln = np.sum(soln[name][...,eid], axis = -1)
                    else:
                        length += len(eid)
                        sln += np.sum(soln[name][...,eid], axis = -1)

                #sln = sln.reshape((self.order+1)**2,self.nvars,len(index))
                soln_avg[:,:,id,t] = sln / length

        print(soln_avg.shape)

    def load_snapshot(self, name):
        soln = defaultdict()
        f = h5py.File(name, 'r')
        for i in f:
            kk = i.split('_')
            if 'hex' in kk:
                soln[i] = np.array(f[i])

        f.close()

        return soln




class Probes(Base):
    def __init__(self, argv):
        super().__init__(argv)
        self.exactloc = False

    def pre_process(self):
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

                mesh_op = self._get_ops(nspts, etype, upts, nupts, self.order)
                mesh_op_vis = self._get_vis_op(nspts, etype, self.order)

                self.mesh_bg.append(np.einsum('ij, jkl -> ikl',mesh_op,self.mesh[key]))
                self.mesh_bg_vis.append(np.einsum('ij, jkl -> ikl',mesh_op_vis,self.mesh[key]))
                self.mesh_trans.append(upts)
                self.lookup.append((etype,part))


    def mainproc(self, pts):
        print('-----------------------------\n')

        # Preparation for parallel this code
        #comm = MPI.COMM_WORLD
        #rank = comm.Get_rank()
        #size = comm.Get_size()

        self.pre_process()

        outpts = list()

        # Input to algorithm to do finding closest point process
        closest = self._closest_pts(self.mesh_bg, pts)

        # Sample points we're responsible for, grouped by element type + rank
        elepts = [[] for i in range(len(self.mesh_inf))]

        # If we are looking for the exact locations
        if self.exactloc:
            # For each sample point find our nearest search location
            for i, (dist, etype, (uidx, eidx)) in enumerate(closest):
                elepts[etype].append((i, eidx, self.mesh_trans[etype][uidx]))

            # Refine
            outpts.append(self._refine_pts(elepts, pts).reshape(nupts,
                                            neles, self.nvars).swapaxes(1,2))
        else:
            # For each sample point find our nearest search location
            for i, (dist, etype, (uidx, eidx)) in enumerate(closest):
                elepts[etype].append((i, eidx, uidx, etype))

            # Get time series
            self.sort_time_series(elepts)

        #self._flash_to_disk(outpts)

    def sort_time_series(self, elepts):
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
        soln_pts = np.zeros([Npts,self.nvars,len(self.time)])

        for t in range(len(self.time)):
            print(t)
            soln = f'{self.solndir}_{self.time[t]}.pyfrs'
            soln = self.load_snapshot(soln)
            for kdx, idx in lookupid.items():
                name = f'{self.dataprefix}_{kdx}'

                """An error here: think about pts in the different partitions"""
                soln_pts[...,t] = soln[name][idx[0],:,idx[1]]


        print(soln_pts.shape)

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
        elelist = self.mesh_old_vis
        slnlist = self.soln
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
        for etype, (eles, epts, soln) in enumerate(zip(elelist, elepts, slnlist)):
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
            new_sln = np.einsum('ij,jki -> ik', ops, soln[:, :, eidx])
            # Append index and solution of each point
            ptsinfo.extend(info for info in zip(idx, new_sln))

        # Resort to the original index
        ptsinfo.sort()

        return np.array([new_sln for idx, new_sln in ptsinfo])


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

            # Apply check routine after three iterations of Newton's method
            if k > 2:
                kdists = np.linalg.norm(plocs - kplocs, axis=1)
                index = np.where(kdists > tol)[0]
                if len(index) == 0:
                    break
            if k == 49:
                """Currently only precise location is acceptable"""
                raise RuntimeError('warning: failed to apply Newton Method')
                # Compute the initial and final distances from the target location
                idists = np.linalg.norm(plocs - iplocs, axis=1)

                # Replace any points which failed to converge with their initial guesses
                closer = np.where(idists < kdists)
                ktlocs[closer] = itlocs[closer]
                kplocs[closer] = iplocs[closer]


        return ktlocs, kplocs

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
