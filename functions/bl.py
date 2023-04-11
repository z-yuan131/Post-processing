import numpy as np
from collections import defaultdict
import h5py

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import ticker
from matplotlib import colors
from scipy.signal import welch, get_window


from base import Base
from pyfr.util import subclasses
from pyfr.shapes import BaseShape

"""
This code is for processing boundary layer properties
The code can be divided into two parts:
the first part is designed to focus on the boundary layer profiles
the second part is focus on the surface pressure and stress distribution
the first part will use data from the spanfft
the second part will use data from the grad
"""

class BL_base(Base):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv)
        self.AoA = icfg.getfloat(fname, 'AoA', 0)
        self._trip_loc = icfg.getfloat(fname, 'trip-loc', None)
        self.fmt = icfg.get(fname, 'format', 'primitive')
        self.tol = 1e-5
        self.etype = ['hex']

        self.L = 100

        print(self._constants)
        self._rho = rho = self._constants['rhoInf']
        self._Uinf = Uinf = self._constants['uInf']
        self._pinf = pinf = self._constants['pInf']
        self._dyna_p = 0.5*rho*Uinf**2

        self.enpts = {}
        for k in self.mesh_inf:
            if k.split('_')[1] in self.etype:
                self.enpts[k.split('_')[1]] = self.mesh_inf[k][1][0]

    def _load_preproc_wall_mesh(self):
        # Get wall mesh from pyfrm file
        mid = defaultdict(list)
        for k in self.mesh:
            if 'bcon' in k.split('_') and 'wall' in k.split('_'):
                for etype, eid, fid, pid in self.mesh[k][['f0','f1','f2','f3']].astype('U4,i4,i1,i2'):
                    if etype in self.etype:
                        part = k.split('_')[-1]
                        mid[f'spt_{etype}_{part}'].append((eid, fid))

        basismap = {b.name: b for b in subclasses(BaseShape, just_leaf=True)}

        fpts, m0, mesh_op = {}, {}, {}
        for etype in self.etype:
            basis = basismap[etype](self.enpts[etype], self.cfg)
            fpts[etype] = basis.facefpts
            m0[etype] = self._get_interp_mat(basis, etype)
            mesh_op[etype] = self._get_mesh_op(etype, self.enpts[etype])

        for k in mid:
            eid, fid = zip(*mid[k])
            etype = k.split('_')[1]
            # Get elements on the surface
            msh = self.mesh[k][:,list(eid)]
            # Interpolation to same order of solution
            msh = np.einsum('ij, jkl -> ikl',mesh_op[etype],msh)
            # Get face points based on etype
            msh = np.array([m0[etype][fpts[etype][fid[id]]] @ msh[:,id] for id in range(len(fid))])

            try:
                mesh = np.append(mesh, msh, axis = 0)
            except UnboundLocalError:
                mesh = msh

        # Get a slice of mesh and get rid of all duplicated points
        # Set the first point as reference
        mesh = mesh.reshape(-1, self.ndims)
        msh = mesh[0]
        # Get a slice
        index = np.where(abs(mesh[:,-1] - msh[-1]) < self.tol)[0]
        mesh = mesh[index]
        # Use Pandas to get rid of duplicated points
        mesh = self._duplpts_pd(mesh)
        """
        plt.figure()
        plt.plot(mesh[:,0],mesh[:,1],'.')
        plt.show()
        """
        return mesh @ self.rot_map()

    def _ortho_mesh(self, mesh):
        # Split mesh into suction and pressure side
        index = np.where(mesh[:,1] >= 0)[0]
        idx = np.argsort(mesh[index, 0])
        meshu = mesh[index[idx]]

        index = np.where(mesh[:,1] <= 0)[0]
        idx = np.argsort(mesh[index, 0])
        meshl = mesh[index[idx]]

        vect = np.diff(meshu, axis = 0)
        # Normalise tangential vector
        vectu = vect/np.linalg.norm(vect, axis = -1)[:,None]
        # Calculate normal vector
        vecnu = np.cross(vectu,np.array([0,0,-1]))

        vect = np.diff(meshl, axis = 0)
        # Normalise tangential vector
        vectl = vect/np.linalg.norm(vect, axis = -1)[:,None]
        # Calculate normal vector
        vecnl = np.cross(vectl,np.array([0,0,1]))

        mesh = np.append(meshu[:-1], meshl[1:], axis = 0)
        vect = np.append(vectu, vectl, axis = 0)
        vecn = np.append(vecnu, vecnl, axis = 0)

        # Create the mesh by growing out via Chebychev points
        from math import pi
        L = 10*(1-np.cos(np.linspace(0, pi/2, 200)))
        mesh = mesh[...,None] + np.einsum('ij,k ->ijk', vecn, L)

        # Angle vector from cartitian to local coordinate
        xt = vect @ np.array([1,0,0])
        yt = vect @ np.array([0,1,0])

        xn = vecn @ np.array([1,0,0])
        yn = vecn @ np.array([0,1,0])

        return mesh.swapaxes(1,-1), np.append(xt[:,None],yt[:,None], axis = -1).T

    def _get_interp_mat(self, basis, etype):
        iqrule = self._get_std_ele(etype, basis.nupts, self.order)
        iqrule = np.array(iqrule)

        # Use strict solution points for quad and pri or line
        if self.ndims == 3:
            self.cfg.set('solver-elements-tri','soln-pts','williams-shunn')
            self.cfg.set('solver-elements-quad','soln-pts','gauss-legendre')
        else:
            self.cfg.set('solver-elements-line','soln-pts','gauss-legendre')

        m = []
        for id0, (kind, proj, norm) in enumerate(basis.faces):
            npts = basis.npts_for_face[kind](self.order)

            qrule = self._get_std_ele(kind, npts, self.order)
            qrule = np.array(qrule)

            pts = self._proj_pts(proj, qrule)

            # Search for closest point
            m0 = np.zeros([npts, basis.nupts])
            for id, pt in enumerate(pts):
                idx = np.argsort(np.linalg.norm(iqrule - pt, axis = 1))
                m0[id, idx[0]] = 1
            m.append(m0)

        return np.vstack(m)

    def _proj_pts(self, projector, pts):
        pts = np.atleast_2d(pts.T)
        return np.vstack(np.broadcast_arrays(*projector(*pts))).T

    def _duplpts_pd(self, mesh, subset=['x','y']):
        # Use panda to fast drop duplicated points
        import pandas as pd
        df = pd.DataFrame({'x':mesh[:,0], 'y':mesh[:,1], 'z':mesh[:,2]})
        return df.drop_duplicates(subset=subset).values

    def rot_map(self):
        from math import pi
        rot_map = np.array([[np.cos(self.AoA/180*pi),np.sin(self.AoA/180*pi),0],
                [-np.sin(self.AoA/180*pi), np.cos(self.AoA/180*pi), 0],
                [0,0,1]])
        return rot_map[:self.ndims,:self.ndims]

    def stress_tensor(self, du, u):
        c = self._constants

        # Density, pressure
        rho, p = u[0], u[-1]

        # Gradient of velocity
        gradu = du.reshape(self.ndims, self.ndims, -1)

        # Bulk tensor
        bulk = np.eye(self.ndims)[:, :, None]*np.trace(gradu)

        # Viscosity
        mu = c['mu']

        if self._viscorr == 'sutherland':
            cpT = c['gamma']*p/rho/(c['gamma'] - 1)
            Trat = cpT/c['cpTref']
            mu *= (c['cpTref'] + c['cpTs'])*Trat**1.5 / (cpT + c['cpTs'])

        return mu*(gradu + gradu.swapaxes(0, 1) - 2/3*bulk)

    def _reorder_pts(self, mesh, soln):
        #msh = self._duplpts_pd(mesh, ['x','y'])
        index = np.where(abs(mesh[:,-1] - np.max(mesh[:,-1])) < 1e-5)
        msh = mesh[index]

        omesh, osoln = [], []
        for pt in msh:
            index = np.where(np.linalg.norm(pt[:2] - mesh[:,:2], axis = 1) < 1e-3)[0]
            #index2 = np.argsort(mesh[index1,-1])
            #index = index1[index2]
            #print(len(index))
            if len(index) > 30:
                omesh.append(np.mean(mesh[index], axis = 0))
                osoln.append(np.mean(soln[index], axis = 0))
        return np.array(omesh), np.array(osoln)

    def _get_rid_of_dp_pts_cf(self, mesh, soln, index = []):
        mn, sn = [], []
        nvars = soln.shape[-1]
        mesh = mesh.reshape(-1, self.ndims)
        soln = soln.reshape(-1, nvars)
        for k, idx in enumerate(index):
            msh = np.mean(mesh[idx], axis = 0)
            sln = np.mean(soln[idx], axis = 0)

            mn.append(msh)
            sn.append(sln)

        msh = np.stack(mn)
        sln = np.stack(sn)
        return msh, sln

    def _con_to_pri(self, cons):
        rho, E = cons[0], cons[-1]

        # Divide momentum components by rho
        vs = [rhov/rho for rhov in cons[1:-1]]

        # Compute the pressure
        gamma = self.cfg.getfloat('constants', 'gamma')
        p = (gamma - 1)*(E - 0.5*rho*sum(v*v for v in vs))

        return [rho] + vs + [p]


class BL(BL_base):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv, icfg, fname)


    def _load_mesh_soln(self):
        f = h5py.File(f'{self.dir}/spanavg_mean.s','r')
        mesh = np.array(f['mesh'])
        soln = np.array(f['soln'])
        f.close()

        """ Limit the region for plotting and calculation """
        index = np.where(mesh[:,0] < 105)
        mesh, soln = mesh[index], soln[index]
        """ END """

        print(self.fmt)
        if self.fmt != 'primitive':
            soln = np.array(self._con_to_pri(soln.T)).T

        return mesh @ self.rot_map(), soln


    def main_proc(self):
        # Create new mesh which is orthogonal to wall mesh
        mesh_wall = self._load_preproc_wall_mesh()
        # Load original mesh and soln
        mesh, soln = self._load_mesh_soln()

        print(mesh.shape, soln.shape, self._trip_loc)



        #raise RuntimeError
        """ Runtime cp-cf first to get tau"""

        meshw, vecm = self._ortho_mesh(mesh_wall)

        xmesh = []
        # Get location of profiles are plotted
        #xloc = [0.62, 0.75, 0.85, 0.94]
        xloc = np.linspace(0.05, 0.99, 100)
        for x in xloc:
            idx0 = np.where(meshw[:,0,1] > 0)[0]
            idx1 = np.where(meshw[:,0,0] > x*np.max(meshw[:,0,0]))[0][:1]
            xmesh.append(np.mean(meshw[idx0[idx1]], axis = 0))
        xmesh = np.stack(xmesh)
        print(meshw.shape, xmesh.shape)


        xsoln = self._interpolation(mesh, soln, xmesh)
        solnw = self._interpolation(mesh, soln, meshw)
        self._sample_location(meshw, solnw, xmesh, xsoln)
        bledge = self._plot_session_bl_profile(xmesh, xsoln, xloc)

        # Plot field varibles
        self._plot_session_field(mesh, soln, mesh_wall, bledge)

        """
        # Split the region at the tripping location
        mm, info = [], {}
        if self._trip_loc:
            index = np.where(mesh_wall[:,0] < self._trip_loc)[0]
            mm.append(mesh_wall[index])
            index = np.where(mesh_wall[:,0] > self._trip_loc)[0]
            mm.append(mesh_wall[index])

            for id, msh in enumerate(mm):
                meshw, vecm = self._ortho_mesh(msh)
                info[id] = self._interpolation(mesh, soln, meshw, vecm)
        """

        #self._plot_session(info)

    def _plot_session_field(self, mesh, soln, meshw, bledge):
        # Bad points on the wall
        index = [np.argsort(np.linalg.norm(mesh[:,:2] - msh[:2] , axis = 1))[0] for msh in meshw]

        # Seperation bubble
        """
        v = np.linalg.norm(soln[:,[1,2]], axis = 1)
        idx = np.where(v < 1e-3)
        mesh_bubble = mesh[idx]

        sln_amp = np.linalg.norm(soln[:,[1,2,3]], axis = 1)
        soln = np.append(soln, sln_amp[:,None], axis = 1)
        """
        varname = ['rho','u','v','w','p']
        for i in [0,1,2,4]:
            var = soln[:,i]
            levels = np.linspace(np.min(var),np.max(var),40)

            # Wall points
            var[index] = np.NaN

            plt.figure(figsize=(20,5))
            if self._trip_loc:
                for j in range(2):
                    sln = var.copy()
                    if j == 0:
                        idx = np.where(mesh[:,0] > self._trip_loc)[0]
                    else:
                        idx = np.where(mesh[:,0] < self._trip_loc)[0]
                    sln[idx] = np.NaN

                    sln = np.ma.masked_invalid(sln)
                    sln = sln.filled(fill_value=-999)

                    triangle = tri.Triangulation(mesh[:,0],mesh[:,1])
                    plt.tricontourf(triangle, sln.real, levels ,cmap = 'coolwarm') # coldwarm jets


                    #plt.tricontour(triangle, soln[:,-1], levels=[0.0],cmap = 'Reds',linestyles='dashed')
            else:
                sln = var.copy()
                sln = np.ma.masked_invalid(sln)
                sln = sln.filled(fill_value=-999)
                triangle = tri.Triangulation(mesh[:,0],mesh[:,1])
                plt.tricontourf(triangle, sln.real, levels ,cmap = 'coolwarm') # coldwarm jets

            cbar = plt.colorbar()

            # Plot boundary layer edge
            if self._trip_loc:
                idx = np.where(bledge[:,0] > self._trip_loc)[0]
                plt.plot(bledge[idx,0],bledge[idx,1],'r--')
                idx = np.where(bledge[:,0] < self._trip_loc)[0]
                plt.plot(bledge[idx,0],bledge[idx,1],'r--')
            else:
                plt.plot(bledge[idx,0],bledge[idx,1],'r--')

            plt.savefig(f'{self.dir}/figs/{varname[i]}_meanflow.eps')
        plt.show()


    def _sample_location(self, mesh, soln, xmesh, xsoln):
        mesh, xmesh = mesh/100, xmesh/100
        levels = np.linspace(0.0,0.45,30)
        # Mask invalid location, i.e. wall
        soln[:,0] = np.NaN

        # Trip location
        plt.figure()
        if self._trip_loc:
            for i in range(2):
                if i==0:
                    index = np.where(mesh[:,0,0] > self._trip_loc/100)[0]
                else:
                    index = np.where(mesh[:,0,0] < self._trip_loc/100)[0]
                sln = soln.copy()
                sln[index] = np.NaN
                sln = np.ma.masked_invalid(sln)
                sln = sln.filled(fill_value=-999)

                msh = mesh.reshape(-1, self.ndims)
                sln = sln.reshape(-1, self.nvars)
                triangle = tri.Triangulation(msh[:,0],msh[:,1])
                plt.tricontourf(triangle, sln[:,1], levels ,cmap = 'coolwarm') # coldwarm jets
            cbar = plt.colorbar()
        # Probes location
        for i in range(xmesh.shape[0]):
            # Invalid position due to interpolation
            index = np.where(np.isnan(xsoln[i,:,0]) == False)[0]
            plt.plot(xmesh[i,index,0],xmesh[i,index,1],'r--')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        plt.show()


    def _get_stress_tensor(self, msh):
        # Load stress tensor from surface stress calculation
        try:
            f = h5py.File(f'{self.dir}/tau.s','r')
            mesh = np.array(f['mesh'])
            tau = np.array(f['tau'])
            f.close()
        except:
            raise RuntimeError('Run cp cf alogrithm first and copy tau to this directory')

        #tau = np.sum(tau.reshape(9,-1, order = 'F'), axis = 0)
        tau = tau[0,1]

        # Do interpolation
        return np.interp(msh[:,0], mesh[:,0], tau)

    def _plot_session_bl_profile(self, mesh, soln, xloc):
        npts, nlayers, nvars = soln.shape
        vtot = np.linalg.norm(soln[...,1:self.ndims], axis = -1)
        rho = soln[...,0]
        y = np.linalg.norm(mesh[...,:2] - mesh[:,0,:2][:,None], axis = -1)


        """
        tau = self._get_stress_tensor(mesh[:,0,:2])

        # Wall units
        plt.figure()
        for i in range(npts):
            # Drop nan due to interpolation
            index = np.where(np.isnan(vtot[i]) == False)[0]
            ut = np.sqrt(tau[i]/rho[i,index])
            nu = self._constants['mu']/rho[i,index]
            delta_nu = nu/ut
            yplus = y[i,index]/delta_nu
            uplus = vtot[i,index]/ut
            plt.plot(yplus, uplus, label = f'x/c = {xloc[i]}')

        # Log law
        kappa, B = 0.41, 5.2
        index = np.where(yplus > 5)[0]
        plt.plot(yplus[index], (1/kappa)*np.log(yplus[index]) + B, '--', label = 'Log law')
        # Linear law
        index = np.where(yplus < 10)[0]
        plt.plot(yplus[index],yplus[index],'.-',label = 'Linear law')

        plt.legend()
        plt.xscale('log')
        plt.xlabel('$y^+$')
        plt.ylabel('$u^+$')
        plt.show()
        """

        # Mean profile scaled in outer units
        idx = []
        plt.figure()
        for i in range(npts):
            # Drop nan due to interpolation
            index = np.where(np.isnan(vtot[i]) == False)[0]

            deltas = np.trapz(1 - rho[i,index]*vtot[i,index]/(rho[i,index[-1]]*vtot[i,index[-1]]), y[i, index])
            plt.plot(vtot[i,index]/vtot[i,index[-5]], y[i,index]/deltas,label = f'x/c = {xloc[i]}')

            # Get boundary layer edge position
            vedge = vtot[i,index[-5]]
            for j in range(len(y[i])-3):
                if vtot[i,j] > 0.99*vedge:
                    idx.append(mesh[i,j])
                    break
        #plt.legend()
        #plt.xlabel('$\lVert u/U_{\infty} \rVert$')
        plt.xlabel('$||u/U_e||$')
        plt.ylabel('$y/\delta^*$')
        plt.ylim([0,8])
        plt.savefig(f'{self.dir}/figs/velocity_profile.eps')
        plt.show()

        return np.array(idx)



    def _plot_session_bl_profile_berlin(self, mesh, soln, xloc):
        npts, nlayers, nvars = soln.shape
        vtot = np.linalg.norm(soln[...,1:self.ndims], axis = -1)
        rho = soln[...,0]
        y = np.linalg.norm(mesh[...,:2] - mesh[:,0,:2][:,None], axis = -1)

        tau = self._get_stress_tensor(mesh[:,0,:2])

        # Wall units
        plt.figure()
        for i in range(npts):
            # Drop nan due to interpolation
            index = np.where(np.isnan(vtot[i]) == False)[0]
            ut = np.sqrt(np.sum(tau[i])/rho[i,index])
            nu = self._constants['mu']/rho[i,index]
            delta_nu = nu/ut
            yplus = y[i,index]/delta_nu
            uplus = vtot[i,index]/ut
            plt.plot(yplus, uplus, label = f'x/c = {xloc[i]}')

        # Log law
        kappa, B = 0.41, 5.2
        index = np.where(yplus > 5)[0]
        plt.plot(yplus[index], (1/kappa)*np.log(yplus[index]) + B, '--', label = 'Log law')
        # Linear law
        index = np.where(yplus < 10)[0]
        plt.plot(yplus[index],yplus[index],'.-',label = 'Linear law')

        plt.legend()
        plt.xscale('log')
        plt.xlabel('$y^+$')
        plt.ylabel('$u^+$')



        # Mean profile
        plt.figure()
        for i in range(npts):
            plt.plot(vtot[i]/self._constants['uInf'], y[i]/np.max(mesh[:,0,0]),label = f'x/c = {xloc[i]}')
        plt.legend()
        #plt.xlabel('$\lVert u/U_{\infty} \rVert$')
        plt.xlabel('$||u/U_{\infty}||$')
        plt.ylabel('$y/c$')

        # Mean profile scaled in outer units
        plt.figure()
        #deltas = [np.trapz(1 - vtot[i]/np.mean(vtot[i,-5:], axis = -1), y[i]) for i in range(npts)]
        for i in range(npts):
            #plt.plot(vtot[i]/np.mean(vtot[i,-5:], axis = -1), y[i]/deltas[i],label = f'x/c = {xloc[i]}')
            # Drop nan due to interpolation
            index = np.where(np.isnan(vtot[i]) == False)[0]

            deltas = np.trapz(1 - rho[i,index]*vtot[i,index]/(rho[i,index[-1]]*vtot[i,index[-1]]), y[i, index])
            print(deltas)
            plt.plot(vtot[i,index]/vtot[i,index[-1]], y[i,index]/deltas,label = f'x/c = {xloc[i]}')
        plt.legend()
        #plt.xlabel('$\lVert u/U_{\infty} \rVert$')
        plt.xlabel('$||u/U_e||$')
        plt.ylabel('$y/\delta^*$')




        # External results
        import scipy.io
        mat = scipy.io.loadmat(f'{self.dir}/data_BLexp_U30_AoAeff3_xdc94_tripzz0d4.mat')
        print(mat.keys())
        mat['uBL'] = mat['uBL'].reshape(-1)
        mat['dist2wall'] = mat['dist2wall'].reshape(-1)

        plt.figure()
        i = -1
        index = np.where(np.isnan(vtot[i]) == False)[0]
        deltas = np.trapz(1 - rho[i,index]*vtot[i,index]/(rho[i,index[-1]]*vtot[i,index[-1]]), y[i, index])
        plt.plot(vtot[i,index]/vtot[i,index[-1]], y[i,index]/deltas,label = f'LES')
        deltas = np.trapz(1 - mat['uBL']/mat['uBL'][-1], mat['dist2wall'])
        print(mat['dist2wall'],mat['uBL'],deltas)
        plt.plot(mat['uBL']/mat['uBL'][-1], mat['dist2wall']/deltas,'.', label = 'Exp')
        plt.legend()
        #plt.xlabel('$\lVert u/U_{\infty} \rVert$')
        plt.xlabel('$||u/U_e||$')
        plt.ylabel('$y/\delta^*$')


        plt.figure()
        i = -1
        index = np.where(np.isnan(vtot[i]) == False)[0]
        plt.plot(vtot[i,index]/vtot[i,index[-1]], y[i,index]/100,label = f'LES')
        plt.plot(mat['uBL'][1:]/mat['uBL'][-1], mat['dist2wall'][1:]/100 + 0.005,'.', label = 'Exp')
        plt.legend()
        #plt.xlabel('$\lVert u/U_{\infty} \rVert$')
        plt.xlabel('$||u/U_e||$')
        plt.ylabel('$y/c$')


        plt.figure()
        # Y is corrected by density
        i = -1
        index = np.where(np.isnan(vtot[i]) == False)[0]

        Y = [np.trapz(rho[i,index[:id]] / rho[i,index[-1]],y[i,index[:id]]/100) for id in range(1,len(index))]
        Y = np.array([0] + Y)
        plt.plot(vtot[i,index]/vtot[i,index[-1]], Y,label = f'LES')
        plt.plot(mat['uBL'][1:]/mat['uBL'][-1], mat['dist2wall'][1:]/100 + 0.005,'.', label = 'Exp')
        plt.legend()
        #plt.xlabel('$\lVert u/U_{\infty} \rVert$')
        plt.xlabel('$||u/U_e||$')
        plt.ylabel('$\eta$')
        plt.show()


    def _plot_session(self, info):

        for id, inf in info.items():
            if id == 0:
                mesh, soln, meshw = inf['mesh'], inf['soln'], inf['intmesh']
                levels = [np.linspace(np.min(soln[:,i]),np.max(soln[:,i]),50) for i in range(self.nvars)]
            else:
                meshw = inf['intmesh']

            # Bad points on the wall
            index = [np.argsort(np.linalg.norm(mesh[:,:2] - msh[:2] , axis = 1))[0] for msh in meshw[:,0]]
            #plt.figure()
            #plt.plot(mesh[index,0],mesh[index,1],'.')
            soln[index] = np.NaN
            soln = np.ma.masked_invalid(soln)
            soln = soln.filled(fill_value=-999)
            # Bad points on the tripping boundary
            if self._trip_loc:
                id1 = np.where(meshw[:,0,1] > 0)[0]
                id11 = np.argsort(abs(meshw[id1,0,0] - self._trip_loc))[0]
                id2 = np.where(meshw[:,0,1] < 0)[0]
                id22 = np.argsort(abs(meshw[id2,0,0] - self._trip_loc))[0]
                index = [id1[id11],id2[id22]]
                index = [np.argsort(np.linalg.norm(mesh[:,:2] - msh[:2] , axis = 1))[0]  for idx in index for msh in meshw[idx]]
                soln[index] = np.NaN
                soln = np.ma.masked_invalid(soln)
                soln = soln.filled(fill_value=-999)

        plt.figure()
        triangle = tri.Triangulation(mesh[:,0],mesh[:,1])
        plt.tricontourf(triangle, soln[:,0], levels[0] ,cmap = 'coolwarm') # coldwarm jets
        cbar = plt.colorbar()
        plt.show()

    def _interpolation(self, mesh, soln, meshw):
        # Interpolation function
        print(mesh.shape, soln.shape)
        # Use scipy to do linear 2D interpolation
        from scipy.interpolate import LinearNDInterpolator
        interp = LinearNDInterpolator(mesh[:,:2], soln)

        npts, nlayer, ndims = meshw.shape
        msh = meshw.reshape(-1, ndims)
        return interp(msh[:,:2]).reshape(npts, nlayer, soln.shape[-1])




    def _interpolation2(self, mesh, soln, meshw, vecm):
        # Interpolation function
        print(mesh.shape, soln.shape, vecm.shape)
        # Use scipy to do linear 2D interpolation
        from scipy.interpolate import LinearNDInterpolator
        interpU = LinearNDInterpolator(mesh[:,:2], soln[:,1]/soln[:,0])
        interpV = LinearNDInterpolator(mesh[:,:2], soln[:,2]/soln[:,0])

        npts, nlayer, ndims = meshw.shape
        msh = meshw.reshape(-1, ndims)
        U = interpU(msh[:,:2]).reshape(npts, nlayer)
        V = interpV(msh[:,:2]).reshape(npts, nlayer)
        u_t = np.einsum('ij, i -> ij',U,vecm[0]) + np.einsum('ij, i -> ij',V,vecm[1])
        #u_n = np.einsum('ij, i -> ij',U,vecm[2]) + np.einsum('ij, i -> ij',V,vecm[3])

        # Creat a dictionary to store all useful variables
        return {'mesh': mesh, 'soln': soln, 'intut': u_t, 'intmesh':meshw}

        id = 10
        y = ((meshw[id,:,0] - meshw[id,0,0])**2 + (meshw[id,:,1] - meshw[id,0,1])**2)**0.5
        plt.figure()
        plt.plot(u_t[id],y,'.')
        #return u_t, u_n


        index = np.where(msh[:,0] > 0.2)[0]

        levels = np.linspace(-0.01,0.5,60)

        # Get bad points
        u_t[:,0] = np.NaN
        u_t = np.ma.masked_invalid(u_t)
        u_t = u_t.filled(fill_value=-999)
        # On the wall it should be zeros
        #u_t[:,0] = -999
        sln = u_t.reshape(-1)



        plt.figure()
        triangle = tri.Triangulation(msh[:,0],msh[:,1])
        plt.tricontourf(triangle, sln,levels,cmap = 'coolwarm') # coldwarm jets
        cbar = plt.colorbar()

        plt.plot(meshw[:,0,0],meshw[:,0,1],'.')
        plt.plot(meshw[id,:,0],meshw[id,:,1],'.')


        # BL thickness
        blthick = {}
        for ly in range(u_t.shape[1]-1):
            index = np.where(u_t[:,ly]/u_t[:,ly+1] > 0.991)[0]
            for id in index:
                if id not in blthick:
                    blthick[id] = ly
        bl = []
        for id, ly in blthick.items():
            bl.append((meshw[id,0], y[ly]))



        plt.figure()
        triangle = tri.Triangulation(msh[:,0],msh[:,1])
        plt.tricontourf(triangle, sln,levels,cmap = 'coolwarm') # coldwarm jets
        cbar = plt.colorbar()

        plt.plot(meshw[:,0,0],meshw[:,0,1],'.')
        for id, ly in blthick.items():
            plt.plot(meshw[id,ly,0],meshw[id,ly,1],'r.')


        # Original
        plt.figure()
        triangle = tri.Triangulation(mesh[:,0],mesh[:,1])
        plt.tricontourf(triangle, soln[:,1]/soln[:,0],levels,cmap = 'coolwarm') # coldwarm jets
        cbar = plt.colorbar()
        plt.plot(meshw[:,0,0],meshw[:,0,1],'.')

        plt.show()

class BL_Coeff(BL_base):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv, icfg, fname)

    def _load_mesh_soln(self):
        # A strict rule about element type here
        f = h5py.File(f'{self.dir}/grad_mean.s','r')
        mesh = np.array(f['mesh'])
        soln = np.array(f['soln'])
        f.close()

        mesh = mesh /self.L
        self._trip_loc = self._trip_loc / self.L

        return mesh @ self.rot_map(), soln

    def main_proc(self):
        # Load averaged soln and mesh file
        mesh, soln = self._load_mesh_soln()
        print(mesh.shape, soln.shape)

        # Reorder points
        mesh, soln = self._reorder_pts(mesh, soln)
        #mesh, soln = self._get_rid_of_dp_pts_cf(mesh, soln, index)
        #print(mesh.shape, soln.shape)
        """
        index = np.where(abs(mesh[:,-1] + 9) > 0.1)[0]
        idx = [*range(len(mesh))]
        idx = [id for id in idx if id not in index]
        plt.figure()
        #plt.plot(mesh[:,0],mesh[:,1],'.')
        #plt.plot(mesh[index,0],mesh[index,1],'.')
        plt.plot(mesh[idx,0],mesh[idx,1],'.')
        plt.show()
        raise RuntimeError
        """

        cp, cf, mp = defaultdict(list), defaultdict(list), defaultdict(list)
        tau = defaultdict(list)
        for i in range(2):
            # seperate the upper and lower side
            if i == 0:
                index = np.where(mesh[:,1] > 0)[0]
            else:
                index = np.where(mesh[:,1] < 0)[0]
            msh = mesh[index]
            sln = soln[index]

            # Sort along x-axis
            index = np.argsort(msh[:,0])
            msh, sln = msh[index], sln[index]

            # Splite the region before and after the tripping location
            mm, sm = [], []
            if self._trip_loc:
                index = np.where(msh[:,0] > self._trip_loc)[0]
                mm.append(msh[0:index[0]])
                mm.append(msh[index[0]:])
                sm.append(sln[0:index[0]])
                sm.append(sln[index[0]:])
            else:
                mm.append(msh)
                sm.append(sln)


            for msh, sln in zip(mm, sm):
                if len(msh) == 0:
                    continue
                if i == 0:
                    side = 1
                else:
                    side = -1
                _msh, _cp, _cf, _tau = self.cal_cp_cf(msh, sln, side)
                cp[i].append(_cp)
                cf[i].append(_cf)
                tau[i].append(_tau)
                mp[i].append(_msh)

        self._plot_session(mp, cp, cf, tau)

    def _plot_session(self, mesh, cp, cf, tau):
        # Line plots Cp
        plt.figure()
        for k, v in cp.items():
            for msh, _cp in zip(mesh[k],v):
                if k == 0:
                    plt.plot(msh[:,0],_cp,'r')
                else:
                    plt.plot(msh[:,0],_cp,'k-.')
        plt.xlabel('$x/c$')
        plt.ylabel('$-C_p$')
        plt.savefig(f'{self.dir}/figs/cp.eps')
        plt.figure()
        for k, v in cf.items():
            for msh, _cf in zip(mesh[k],v):
                if k == 0:
                    plt.plot(msh[:,0],_cf,'r')
                else:
                    plt.plot(msh[:,0],_cf,'k-.')
            plt.axhline(y = 0.0, color = 'b', linestyle = '--')
        plt.xlabel('$x/c$')
        plt.ylabel('$C_f$')
        plt.savefig(f'{self.dir}/figs/cf.eps')
        plt.show()

        #"""
        f = h5py.File(f'{self.dir}/tau.s','w')
        for k, v in tau.items():
            tt = []
            for msh, _tau in zip(mesh[k],v):
                if tt == []:
                    tt = _tau
                    mmsh = msh
                else:
                    tt = np.append(tt, _tau, axis = -1)
                    mmsh = np.append(mmsh, msh, axis = 0)
            f[f'tau'] = tt
            f[f'mesh'] = mmsh
            break
        f.close()
        #"""

    def cal_cp_cf(self, mesh, soln, side):
        mesh[:,-1] = 0

        # Get normal direction
        vect = mesh[1:] - mesh[:-1]
        vect = np.append(vect, vect[-1][None,:], axis = 0)

        # Be sure that no duplicated points
        index = np.where(np.linalg.norm(vect, axis=1) > 1e-10)[0]
        mesh, soln, vect = mesh[index], soln[index], vect[index]

        # Reshape vectors
        npts, nvars = soln.shape

        # Normalise tangential vector
        vect = vect/np.linalg.norm(vect, axis = -1)[:,None]
        # Calculate normal vector
        vecn = np.cross(vect,np.array([0,0,1]))

        # Calculate stress
        du = soln.T[self.nvars:]
        u = soln.T[:self.nvars]
        tau = self.stress_tensor(du, u)

        # Calculate cf
        cf = np.einsum('ik, jki -> jki', vecn, tau)/self._dyna_p
        cf = tau/self._dyna_p
        # Get wall friction coefficient
        cf = cf[1,0] * side

        # Calculate cp
        cp = (u[-1] - self._pinf)/self._dyna_p

        # Reshape
        return mesh.reshape(npts, self.ndims), cp, cf, tau


    def _duplpts_pd(self, mesh, subset):
        # Use panda to fast drop duplicated points
        import pandas as pd
        df = pd.DataFrame({'x':mesh[:,0], 'y':mesh[:,1], 'z':mesh[:,2]})
        return df.drop_duplicates(subset=subset).values



class BL_Coeff_hotmap(BL_base):
    def __init__(self, argv, icfg, fname):
        super().__init__(argv, icfg, fname)

    def _load_mesh_soln(self):
        # A strict rule about element type here
        f = h5py.File(f'{self.dir}/grad_mean.s','r')
        #f = h5py.File(f'{self.dir}/grad_timesereis2.s','r')
        mesh = np.array(f['mesh'])
        soln = np.array(f['soln'])
        #soln = np.array(f['soln'])[100]
        f.close()

        mesh = mesh /self.L
        self._trip_loc = self._trip_loc / self.L

        return mesh @ self.rot_map(), soln

    def main_proc(self):
        # Load averaged soln and mesh file
        mesh, soln = self._load_mesh_soln()
        print(mesh.shape, soln.shape)

        # Make sure NACA0012 part of aerofoil is used
        index = np.where(mesh[:,0] < 0.98)[0]
        mesh, soln = mesh[index], soln[index]

        mp, cp, cf = defaultdict(list), defaultdict(list), defaultdict(list)
        for i in range(2):
            if i == 0:
                index = np.where(mesh[:,1] > 0)[0]
                msh, sln = mesh[index], soln[index]
            else:
                index = np.where(mesh[:,1] > 0)[0]
                msh, sln = mesh[index], soln[index]

            # sort along x-axis
            index = np.argsort(msh[:,0])
            msh, sln = msh[index], sln[index]

            # Splite the region before and after the tripping location
            mm, sm = [], []
            if self._trip_loc:
                index = np.where(msh[:,0] > self._trip_loc)[0]
                mm.append(msh[0:index[0]])
                mm.append(msh[index[0]:])
                sm.append(sln[0:index[0]])
                sm.append(sln[index[0]:])
            else:
                mm.append(msh)
                sm.append(sln)

            for msh, sln in zip(mm, sm):
                if i == 0:
                    side = -1
                else:
                    side = 1
                _msh, _cp, _cf = self.cal_cp_cf(msh, sln, side)
                cp[i].append(_cp)
                cf[i].append(_cf)
                mp[i].append(_msh)

        self._plot_session(mp, cp, cf)

    def _plot_session(self, mesh, cp, cf):
        # Contour plot skin friction field
        side = ['upper','lower']
        for i in cf:
            # get levels
            abound = []
            for mm, _cf in zip(mesh[i], cf[i]):
                abound += [np.min(_cf),np.max(_cf)]
            levels = np.linspace(np.min(abound),np.max(abound),10)

            plt.figure(figsize=(18,5))
            for mm, _cf in zip(mesh[i], cf[i]):
                levels = np.linspace(np.min(_cf),np.max(_cf),50)

                print(mm.shape, _cf.shape)
                triangle = tri.Triangulation(mm[:,0],mm[:,-1])
                divnorm=colors.TwoSlopeNorm(vmin=np.min(levels), vcenter=0., vmax=np.max(levels))
                plt.tricontourf(triangle, _cf,levels,cmap = 'coolwarm', norm=divnorm) # coldwarm jets

            #plt.ylim([0.55,1])
            plt.xlabel('$x/c$')
            plt.ylabel('$z/c$')

            cbar = plt.colorbar()
            plt.gca().invert_yaxis()
            plt.savefig(f'{self.dir}/figs/cf_hotmap_{side[i]}.eps')
            #plt.tight_layout()
            plt.show()




    def NACA0012_norm_dir(self, mesh, soln):
        #y= +- 0.6*[0.2969*sqrt(x) - 0.1260*x - 0.3516*x2 + 0.2843*x3 - 0.1015*x4]
        dx = np.min([0.01, np.max(mesh[:,0]/1000)])
        if any(mesh[:,1] > 0):
            y = lambda x: 0.6*(0.2969*x**(0.5) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
            dy = y(mesh[:,0] + dx) - y(mesh[:,0])
            vect = np.zeros([3,len(dy)])
            vect[0], vect[1], vect[2] = dx, dy, 0

            #index = np.where(np.linalg.norm(vect, axis = 1) > 1e-10)[0]
            #mesh, soln, vect = mesh[index], soln[index], vect[:,index]
            vect = vect / np.linalg.norm(vect, axis = 0)[None]

            # Calculate normal vector
            vecn = np.cross(vect.T,np.array([0,0,-1]))

        else:
            y = lambda x: -0.6*(0.2969*x**(0.5) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
            dy = y(mesh[:,0] + dx) - y(mesh[:,0])
            vect = np.zeros([3,len(dy)])
            vect[0], vect[1], vect[2] = dx, dy, 0

            #index = np.where(np.linalg.norm(vect, axis = 1) > 1e-10)[0]
            #mesh, soln, vect = mesh[index], soln[index], vect[:,index]

            vect = vect / np.linalg.norm(vect, axis = 0)[None]
            # Calculate normal vector
            vecn = np.cross(vect.T,np.array([0,0,1]))

        return vecn



    def cal_cp_cf(self, mesh, soln, side):
        # Get normal direction from NACA0012 profile
        vecn =  self.NACA0012_norm_dir(mesh, soln)

        # Calculate stress
        du = soln.T[self.nvars:]
        u = soln.T[:self.nvars]
        tau = self.stress_tensor(du, u)

        # Calculate cf
        cf = np.einsum('ij, jki -> jki', vecn, tau)/self._dyna_p

        # Calculate cp
        cp = (u[-1] - self._pinf)/self._dyna_p

        return mesh, cp, cf[1,0]

    def _get_rid_of_dp_pts_cf(self, mesh, soln, index = []):
        mn, sn = [], []
        nvars = soln.shape[-1]
        mesh = mesh.reshape(-1, self.ndims)
        soln = soln.reshape(-1, nvars)
        for k, idx in enumerate(index):
            msh = mesh[idx]
            sln = soln[idx]

            # Use pandas to drop all duplicated points
            mm = self._duplpts_pd(msh,['z'])

            # Average duplicate points
            sln = [np.mean(sln[np.where(np.linalg.norm(pt - msh, axis = -1) < self.tol)[0]], axis = 0) for pt in mm]

            mn.append(mm)
            sn.append(sln)

        return mn, sn

    def _duplpts_pd(self, mesh, subset):
        # Use panda to fast drop duplicated points
        import pandas as pd
        df = pd.DataFrame({'x':mesh[:,0], 'y':mesh[:,1], 'z':mesh[:,2]})
        return df.drop_duplicates(subset=subset).values


class BL_wavenumber_trans(BL_base):
    """
    To do wave number transform, one has to interpolate field alone x direction
    and z direction. The thing about z direction is that our mesh has some very
    small error but could be bad for fft and it can save effort to reorder
    points. Then fft in z direction, fft in x direction, psd in time, we can get
    wavenumber, frequency in x and time repesct to each wavenumber in span.
    Define, kx = alpha, kz = beta and frequency f. Notice here none of them is angular.
    """
    def __init__(self, argv, icfg, fname):
        super().__init__(argv, icfg, fname)

        n = self.order + 1
        self.ele_map = [0, n -1, n*(n-1), n**2 - 1]
        self.chord = 100

    def _pre_proc_mesh(self):
        # A strict rule about element type here
        #f = h5py.File(f'{self.dir}/grad.m','r')
        f = h5py.File(f'./grad.m','r')
        mesh = np.array(f['hex']) / self.chord
        f.close()

        mesh = mesh @ self.rot_map()

        # Center of a element
        elec = np.mean(mesh[self.ele_map], axis = 0)

        """
        # Slite upper and lower side
        mm, idx = [], []
        for i in range(2):
            if i == 0:
                index = np.where(elec[:,1] > 0)[0]
            else:
                index = np.where(elec[:,1] < 0)[0]
            if self._trip_loc:
                index1 = np.where(elec[index,0] > self._trip_loc)[0]

            idx0 = index[index1]
            msh = mesh[:, idx0]




            #mn.append()
            #mm.append(mesh[index[index1]])

        """
        mesh = mesh.reshape(-1, self.ndims)

        # Slite upper and lower side
        mm, idx, mn, sp = [], [], [], []
        for i in range(2):
            if i == 0:
                index = np.where(mesh[:,1] > 0)[0]
            else:
                index = np.where(mesh[:,1] < 0)[0]
            if self._trip_loc:
                index1 = np.where(mesh[index,0] > self._trip_loc/self.chord)[0]

            idx.append(index[index1])
            msh = mesh[index[index1]]
            #plt.figure()
            #plt.plot(msh[:,0],msh[:,1],'.')
            #plt.show()
            # Create a new mesh for linear interpolation
            x = np.linspace(np.min(msh[:,0]) + 1/100,np.max(msh[:,0]) - 1/100, 100)
            z = np.linspace(np.min(msh[:,-1]) + 0.1/100,np.max(msh[:,-1]) - 0.1/100, 100)
            xv, zv = np.meshgrid(x, z)
            mn.append(np.array([xv.reshape(-1), zv.reshape(-1)]).T)
            mm.append(msh)
            sp.append(xv.shape)

        return idx, mm, mn, sp


    def _pre_proc_soln(self, index, mesh, meshn, shape):
        f = h5py.File(f'{self.dir}/field_pressure_timeseries_15285.00_23720.00.s', 'r')
        p = []
        for i in range(2):
            p.append(np.array(f[f'{i}']))
        f.close()

        # Use scipy to do linear 2D interpolation
        from scipy.interpolate import LinearNDInterpolator

        pn = []
        for idx, mm, mn, sp, _p in zip(index, mesh, meshn, shape, p):
            #plt.figure()
            #plt.plot(mn[:,0],mn[:,1],'.')
            #plt.plot(mm[:,0],mm[:,-1],'.')
            #plt.show()
            pp = _p.T
            nt = pp.shape[-1]
            interp = LinearNDInterpolator(mm[:,[0,2]], pp)
            (nx, nz) = sp
            pp = interp(mn).reshape(nx,nz,nt)
            print(np.min(pp))

            pn.append(pp)

        return pn

    def _get_pressure(self, index, mesh):
        pp = defaultdict(list)
        for t in self.time:
            print(t)
            f = h5py.File(f'{self.dir}/grad_{t}.s', 'r')
            soln = np.array(f['hex'])
            f.close()

            nvars = soln.shape[-1]
            soln = soln.reshape(-1, nvars).T
            rho, rhou, E = soln[0], soln[1:self.ndims+1], soln[self.nvars-1]
            p = E - 0.5*np.sum([rhou[i]**2 for i in range(self.ndims)], axis = 0)/rho
            p = p*(self._constants['gamma'] - 1)

            for id, idx in enumerate(index):
                pp[id].append(p[idx])

        f = h5py.File(f'{self.dir}/field_pressure_timeseries_{self.time[0]}_{self.time[-1]}.s', 'w')
        for k, v in pp.items():
            print(k,np.stack(v).shape, mesh[k].shape)
            f[f'{k}'] = np.stack(v)
            f[f'mesh_{k}'] = mesh[k]
        f.close()

    def main_proc(self):
        # Load soln and mesh file
        idx, mm, mn, sp = self._pre_proc_mesh()

        # First, precess pressure data into a file
        #self._get_pressure(idx, mm)
        #return 0

        pn = self._pre_proc_soln(idx, mm, mn, sp)

        for msh, p, s in zip(mn, pn, sp):
            #self._post_proc_field_alpha_f(msh, p, s)
            self._post_proc_field_alpha_beta(msh, p, s)
        #plt.show()

    def _post_proc_field_alpha_beta(self, msh, p, shape):
        # FFT in span
        pp = np.fft.fft(p, axis = 1) #/ (0.5* self._constants['rhoInf']*self._constants['uInf'])
        # FFT in streamwise
        pp = np.fft.fft(pp, axis = 0)
        # PSD in time
        f, pxx = self.psd(pp.conj())

        alpha = np.fft.fftfreq(shape[0], msh[1,0] - msh[0,0])
        alpha = np.fft.fftshift(alpha)
        beta = np.fft.fftfreq(shape[1], msh[0,1] - msh[0,0])
        beta = np.fft.fftshift(beta)
        pxx = np.fft.fftshift(pxx).real

        for ff in range(10):
            pp = pxx[:,:,ff]

            m0, m1 = np.meshgrid(alpha, beta)
            plt.figure()
            levels = np.linspace(np.min(pp), np.max(pp), 10)
            plt.contourf(m0, m1, pp.T, levels, locator = ticker.LogLocator(), cmap = 'binary')

            #plt.plot(np.array([-50*0.3,0]),np.array([50,0]),'--')

            plt.xlabel('$k_x$')
            plt.ylabel('$k_z$')
            #plt.colorbar()
            #plt.clim([1e-6,1e-3])
            plt.title(f'$f = {f[ff]}$')
            #plt.ylim([0,np.max(f)])
            #plt.ylim([0,30])
        plt.show()


    def _post_proc_field_alpha_f(self, msh, p, shape):
        # FFT in span
        pp = np.fft.fft(p, axis = 1) #/ (0.5* self._constants['rhoInf']*self._constants['uInf'])
        # FFT in streamwise
        pp = np.fft.fft(pp, axis = 0)
        # PSD in time
        f, pxx = self.psd(pp.conj())

        alpha = np.fft.fftfreq(shape[0], msh[1,0] - msh[0,0])
        alpha = np.fft.fftshift(alpha)
        f = np.fft.fftshift(f)
        pxx = np.fft.fftshift(pxx).real

        for beta in range(10):
            pp = pxx[:,beta]

            m0, m1 = np.meshgrid(alpha, f)
            plt.figure()
            levels = np.linspace(np.min(pp), np.max(pp), 10)
            plt.contourf(m0, m1, pp.T, levels, locator = ticker.LogLocator(), cmap = 'binary')

            #plt.plot(np.array([-50*0.3,0]),np.array([50,0]),'--')

            plt.xlabel('$k_x$')
            plt.ylabel('$f$')
            #plt.colorbar()
            #plt.clim([1e-6,1e-3])
            plt.title(f'$beta = {beta}$')
            #plt.ylim([0,np.max(f)])
            #plt.ylim([0,30])
        plt.show()



    def psd(self, soln):
        fs = 1/self.dt*self.chord/self._constants['uInf']
        nperseg = 256
        noverlap = int(nperseg*0.75)
        window = get_window('hann', nperseg)
        f, PSD = welch(soln, fs, window, nperseg, noverlap, scaling='density', axis = -1)
        return f, PSD
