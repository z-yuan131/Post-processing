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
        self.layers = 3
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
        mesh_wall_tag, mesh_wall, bcon = self.get_boundary()

        # For single rank process, we'd better to pre load connectivities
        con = defaultdict(list)
        con_new = defaultdict(list)
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
                    if not all([r[0] in self.suffix]) or not all([l[0] in self.suffix]):
                        continue
                    if mesh_wall_tag[part].get(l, -1) == i and r not in mesh_wall_tag[part]:
                        mesh_wall_tag[part].update({r: i + 1})
                        mesh_wall[f'spt_{r[0]}_{part}'].append(r[1])
                        con_new[f'{part}_{part}'].append((l,r))


                    elif mesh_wall_tag[part].get(r, -1) == i and l not in mesh_wall_tag[part]:
                        mesh_wall_tag[part].update({l: i + 1})
                        mesh_wall[f'spt_{l[0]}_{part}'].append(l[1])
                        con_new[f'{part}_{part}'].append((l,r))

                # Grow our element set by considering adjacent partitions
                for p, (pc, pcr, sb) in pcon[part].items():
                    for l, r, b in zip(pc, pcr, sb):
                        if not all([r[0] in self.suffix]):
                            continue
                        try:
                            if b and r not in mesh_wall_tag[f'{p}']:
                                mesh_wall_tag[f'{p}'].update({r: i + 1})
                                mesh_wall[f'spt_{r[0]}_{p}'].append(r[1])
                                con_new[f'{part}_{p}'].append(l)
                                con_new[f'{p}_{part}'].append(r)

                        except  KeyError:
                                mesh_wall_tag.update({f'{p}': {r: i + 1}})
                                mesh_wall[f'spt_{r[0]}_{p}'].append(r[1])
                                con_new[f'{part}_{p}'].append(l)
                                con_new[f'{p}_{part}'].append(r)

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


    def reorder(self):
        mesh_wall_tag, mesh_wall, con = self.get_wall_O_grid()

        # Corner points map for hex type of elements
        n = self.meshord + 1
        _map = {'hex': [0, n-1, n*(n-1), n**2-1, n**2*(n-1), (n**2+1)*(n-1), n**3-n, n**3-1]}

        # Collect one periodic boundary
        eidp = defaultdict()
        amin = min([np.min(self.mesh[key][...,-1]) for key in mesh_wall])
        for key in mesh_wall:
            index = np.where(abs(self.mesh[key][:,mesh_wall[key],-1] - amin) < self.tol)[1]
            if len(index) > 0:
                index = list(set(index))

                etype = key.split('_')[1]
                eidp[key] = np.array(mesh_wall[key])[index]
                mesh = self.mesh[key][:,eidp[key],:2]
                mesh = np.mean(mesh[_map[etype]], axis = 0)
                try:
                    mesh_prio = np.append(mesh_prio,mesh,axis = 0)
                except UnboundLocalError:
                    mesh_prio = mesh

        # Reduce/reorder connectivities
        con_new = defaultdict(list)
        for key in mesh_wall:
            _, etype, part = key.split('_')
            if key not in eidp.keys():
                continue
            for kk in con:
                if kk.split('_')[0] == part:
                    if kk.split('_')[-1] == part:
                        for l,r in con[kk]:
                            if l[1] in eidp[key] and r[1] in eidp[key]:
                                con_new[kk].append((l,r))
                    else:
                        p = kk.split('_')[-1]
                        nkey = f'{_}_{etype}_{p}'
                        nk = f'{p}_{part}'
                        for l,r in zip(con[kk],con[nk]):
                            if l[1] in eidp[key] and r[1] in eidp[nkey]:
                                con_new[kk].append(l)


        # Find adjacent element in periodic direction
        zeleid = defaultdict(list)
        for key in mesh_wall:
            _,etype,part = key.split('_')

            # corner points of each element is enough for sorting
            """This tolerance should depend on mesh size"""
            tol = 1e-5
            eids = np.array(mesh_wall[key])
            mesh = self.mesh[key][:,eids]
            mesh = np.mean(mesh[_map[etype]], axis = 0)

            for id, pt in enumerate(mesh_prio):
                dists = np.linalg.norm(mesh[:,:2] - pt, axis=1)
                index = np.where(dists < tol)[0]
                if len(index) > 0:
                    zeleid[id].append((key, index))

        return zeleid, mesh_wall

    def spancheck(self, meshid, key):
        mesh = self.mesh[key][:,meshid,:2]
        return np.allclose(mesh[:,0],mesh[:,1])


    def spanavg(self):
        zeleid, mesh_wall = self.reorder()

        ele_type = {'hex': 'quad'}
        soln_pts = {'quad': 'gauss-legendre'}

        # Mesh average
        mesh = defaultdict()
        for key in mesh_wall:
            print(key)
            mesh[key] = self.mesh[key][:,mesh_wall[key]]
            npts = self.meshord+1
        mesh_avg = np.zeros([npts**2,self.ndims,len(zeleid)])

        for id, item in zeleid.items():
            for idx, (key, eid) in enumerate(item):
                if idx == 0:
                    length = len(eid)
                    msh = np.sum(mesh[key][:,eid], axis = 1)
                else:
                    length += len(eid)
                    msh += np.sum(mesh[key][:,eid], axis = 1)
                plt.figure()
                plt.plot(mesh[key][:,eid,0],mesh[key][:,eid,1],'.')
                plt.figure()
                plt.plot(mesh[key][:,eid,0],mesh[key][:,eid,-1],'.')
                plt.show()
                raise RuntimeError

            # Note here, reordering bases on the fact
            msh = msh.reshape(npts**2,npts,self.ndims, order = 'F')
            print(msh.shape)
            plt.figure()
            plt.plot(msh[:,0,0],msh[:,0,1],'.')
            plt.figure()
            plt.plot(msh[:,0,0],msh[:,0,-1],'.')
            plt.figure()
            plt.plot(msh[:,0,1],msh[:,0,-1],'.')
            plt.show()
            raise RuntimeError

            mesh_avg[:,:,id] = np.mean(msh, axis = 1) / length



        id  = 100
        plt.figure()
        plt.plot(mesh_avg[:,0,id],mesh_avg[:,1,id],'.')
        plt.figure()
        plt.plot(mesh_avg[:,0,id],mesh_avg[:,-1,id],'.')
        plt.figure()
        plt.plot(mesh_avg[:,1,id],mesh_avg[:,-1,id],'.')
        plt.show()
        raise RuntimeError


        # use strict soln pts set if not exit
        etype = ele_type[key.split('_')[1]]
        try:
            self.cfg.get(f'solver-elements-{etype}', 'soln-pts')
        except:
            self.cfg.set(f'solver-elements-{etype}', 'soln-pts', soln_pts[etype])

        # Get operator
        mesh_op = self._get_mesh_op(etype, mesh_avg.shape[0])
        mesh_avg = np.einsum('ij,jkl -> ikl', mesh_op, mesh_avg)

        # Flash ro disk
        self._flash_to_disk(self.dir, mesh_avg)
        del mesh, mesh_avg, msh, mesh_op

        soln_avg = np.zeros([(self.order+1)**2,self.nvars,len(zeleid),len(self.time)])
        # Solution average
        for t in range(len(self.time)):
            print(t)
            soln = f'{self.solndir}_{self.time[t]}.pyfrs'
            soln = self.load_snapshot(soln, mesh_wall)

            for id, item in zeleid.items():
                for idx, (key, eid) in enumerate(item):
                    if idx == 0:
                        length = len(eid)
                        sln = np.sum(soln[key][...,eid], axis = -1)
                    else:
                        length += len(eid)
                        sln += np.sum(soln[key][...,eid], axis = -1)

                # Note here, reordering bases on the fact
                sln = sln.reshape((self.order+1)**2,(self.order+1),self.nvars, order = 'F')
                soln_avg[:,:,id,t] = np.mean(sln, axis = 1) / length

        # use strict soln pts set if not exit
        etype = ele_type[key.split('_')[1]]
        if self.cfg.get(f'solver-elements-{etype}', 'soln-pts', None) == None:
            self.cfg.set(f'solver-elements-{etype}', 'soln-pts', soln_pts[etype])

        # Get operator
        soln_op = self._get_soln_op(ele_type[key.split('_')[1]], soln_avg.shape[0])
        soln_avg = np.einsum('ij,jklm -> iklm', soln_op, soln_avg)

        # Flash ro disk
        self._flash_to_disk(self.dir, soln_avg)


    def spanfft(self):
        raise NotImplementedError('This function is under development')


    def load_snapshot(self, name, region):
        soln = defaultdict()
        f = h5py.File(name, 'r')
        for k in region:
            _, etype, part = k.split('_')
            kk = f'{self.dataprefix}_{etype}_{part}'
            soln[k] = np.array(f[kk])[...,region[k]]
        f.close()
        return soln

    def _flash_to_disk(self, dir, array):
        if array.shape[1] == self.ndims:
            f = h5py.File(f'{dir}/spanavg.m', 'w')
            f['mesh'] = array
            f.close()
        else:
            f = h5py.File(f'{dir}/spanavg.s', 'w')
            f['solution'] = array
            f['info'] = np.array([self.tst, self.ted, self.dt])
            f.close()




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

                mesh_op = self._get_ops_interp(nspts, etype, upts, nupts, self.order)
                mesh_op_vis = self._get_vis_op(nspts, etype, self.order)

                self.mesh_bg.append(np.einsum('ij, jkl -> ikl',mesh_op,self.mesh[key]))
                self.mesh_bg_vis.append(np.einsum('ij, jkl -> ikl',mesh_op_vis,self.mesh[key]))
                self.mesh_trans.append(upts)
                self.lookup.append((etype,part))


    def mainproc(self, pts):
        print('-----------------------------\n')

        # Parallel sorting the whole time series
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Only root rank is enough to process data
        if rank == 0:
            elepts = self.procpts(pts)
        else:
            elepts = None
            self.lookup = None
        # Wait until everything finished
        comm.barrier()

        # Boardcast pts and eles information
        elepts = comm.bcast(elepts, root=0)
        self.lookup = comm.bcast(self.lookup, root=0)

        # Get time series
        soln_pts = self.sort_time_series(elepts, rank, size)

        soln_pts = comm.gather(soln_pts, root=0)

        if rank == 0:
            self._flash_to_disk(soln_pts)

    def procpts(self, pts):

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

            raise NotImplementedError('refinement needs some process before output')
            # Refine
            outpts.append(self._refine_pts(elepts, pts).reshape(nupts,
                                            neles, self.nvars).swapaxes(1,2))
        else:
            # For each sample point find our nearest search location
            for i, (dist, etype, (uidx, eidx)) in enumerate(closest):
                elepts[etype].append((i, eidx, uidx, etype))

        return elepts


    def sort_time_series(self, elepts, rank, size):
        lookupid = defaultdict(list)
        Npts = 0
        for epts, lookup in zip(elepts,self.lookup):
            if not epts:
                continue
            i, eidx, uidx, etype = zip(*epts)

            lookupid[f'{lookup[0]}_{lookup[1]}'].append(uidx)
            lookupid[f'{lookup[0]}_{lookup[1]}'].append(eidx)
            Npts += len(eidx)

        # Get each rank a period of time
        Ntime = len(self.time)//size + 1
        for i in range(size):
            if i == rank:
                if i+1 == size:
                    time = self.time[i*Ntime:]
                else:
                    time = self.time[i*Ntime:(i+1)*Ntime]

        # Get information from solution field
        soln_pts = np.zeros([Npts,self.nvars,len(time)])

        for t in range(len(time)):
            print(t)
            soln = f'{self.solndir}_{time[t]}.pyfrs'
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
        #print(soln_pts.shape)

        return soln_pts


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
        f = h5py.File('soln_pts.s','w')
        for pts in range(len(outpts)):
            f[f'soln_{pts}'] = outpts[pts]
        f.close()




    """
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
    """
