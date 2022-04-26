import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import time

class MPFAH_flux:

    def __init__(self,mesh,fluids,wells,bc_val,nlfv):

        nbe = mesh.edges.boundary.shape[0]
        nel = mesh.faces.center[:].shape[0]

        edges = mesh.edges.connectivities[:]
        coord = mesh.nodes.coords[:]
        esize = np.linalg.norm(coord[edges[:,1]]- coord[edges[:,0]], axis = 1)

        interface = nlfv.misc.interface_elem

        mobility = fluids.edge_mobility

        IsNeu = bc_val.pressure.Neu_edges 
        value = bc_val.pressure.val_array

        W_DMP = nlfv.pressure.W_DMP
        trans = nlfv.pressure.trans

        self.row = []
        self.col = []
        self.data = []

        self.I = fluids.source*mesh.faces.area[:]

        self.I[interface[IsNeu,0]] -= esize[IsNeu]*value[IsNeu]

        self.boundary_pressure(mesh,mobility,bc_val,nlfv)

        flux = (esize*mobility)[:,np.newaxis,np.newaxis]*trans

        flux[nbe:,0] *= W_DMP[:,1,np.newaxis]
        flux[nbe:,1] *= W_DMP[:,0,np.newaxis]

        self.matrix_assebmly(mesh,bc_val,nlfv,flux)
                
        self.hpoints_interpolation(mesh,mobility,bc_val,nlfv,flux)

        self.M = coo_matrix((self.data,(self.row,self.col)),shape=(nel,nel)).tocsr()

        del self.row, self.col, self.data

    def boundary_pressure(self,mesh,mobility,bc_val,nlfv):

        nbe = mesh.edges.boundary.shape[0]

        trans = nlfv.pressure.trans
        hpoints = nlfv.pressure.aux_point

        IsNeu = bc_val.pressure.Neu_edges
        IsDir = bc_val.pressure.Dir_edges
        value = bc_val.pressure.val_array

        self.p_ij = np.zeros([nbe,4])

        self.p_ij[IsDir,0] = -1
        self.p_ij[IsDir,1] = value[IsDir]
        
        hp = hpoints[IsNeu,0]
        ksi = trans[IsNeu,0]
        
        hp_neu = np.where(hp[:,0] == IsNeu, hp[:,0], hp[:,1])
        hp_aux = np.where(hp[:,0] == IsNeu, hp[:,1], hp[:,0])
        hp_aux = np.where(np.isin(hp_aux,IsNeu), -1 , hp_aux)

        self.p_ij[IsNeu,0] = hp_aux

        ksi_neu = np.where(hp[:,0] == IsNeu, ksi[:,0], ksi[:,1])
        ksi_aux = np.where(hp[:,0] == IsNeu, ksi[:,1], ksi[:,0])

        aux_InNeu = np.isin(hp_aux,IsNeu)
        if aux_InNeu.sum() > 0:
            pass

        aux_not_InNeu = np.invert(aux_InNeu)
        if aux_not_InNeu.sum() > 0:

            ksi1 = ksi_neu[aux_not_InNeu]            
            ksi2 = ksi_aux[aux_not_InNeu]

            M = (ksi1 + ksi2)/ksi1

            self.p_ij[IsNeu,1] = M

            F = value[IsNeu][aux_not_InNeu]/(mobility[IsNeu][aux_not_InNeu]*ksi1)

            self.p_ij[IsNeu,2] = F

            P = ksi2/ksi1

            self.p_ij[IsNeu,3] = P           

    def matrix_assebmly(self,mesh,bc_val,nlfv,flux):

        nbe = mesh.edges.boundary.shape[0]
        
        IsDir = bc_val.pressure.Dir_edges

        interface = nlfv.misc.interface_elem
        W_DMP = nlfv.pressure.W_DMP

        lef = interface[IsDir,0]

        self.row.extend(lef)
        self.col.extend(lef)
        self.data.extend(flux[IsDir,0].sum(axis = 1))

        lef = interface[nbe:,0]
        rel = interface[nbe:,1]

        lFlux = flux[nbe:,0].sum(axis = 1)
        rFlux = flux[nbe:,1].sum(axis = 1)

        self.row.extend(lef)
        self.col.extend(lef)
        self.data.extend(lFlux)
        
        self.row.extend(lef)
        self.col.extend(rel)
        self.data.extend(-rFlux)
        
        self.row.extend(rel)
        self.col.extend(rel)
        self.data.extend(rFlux)

        self.row.extend(rel)
        self.col.extend(lef)
        self.data.extend(-lFlux)

    def hpoints_interpolation(self,mesh,mobility,bc_val,nlfv,flux):
        
        nbe = mesh.edges.boundary.shape[0]

        IsDir = bc_val.pressure.Dir_edges 
    
        interface = nlfv.misc.interface_elem
        hpoints = nlfv.pressure.aux_point
        
        lef = interface[IsDir,0]

        hp = hpoints[IsDir,0,0]
        termo0 = flux[IsDir,0,0]

        self.boundary_interpolation(nbe,mobility,bc_val,nlfv,lef,termo0,hp)

        hp = hpoints[IsDir,0,1]
        termo0 = flux[IsDir,0,1]

        self.boundary_interpolation(nbe,mobility,bc_val,nlfv,lef,termo0,hp)

        lef = interface[nbe:,0]
        rel = interface[nbe:,1]

        col = lef.copy()

        hp = hpoints[nbe:,0,0]
        termo0 = flux[nbe:,0,0]
        
        self.domain_interpolation(nbe,bc_val,nlfv,lef,rel,col,termo0,hp)

        hp = hpoints[nbe:,0,1]
        termo0 = flux[nbe:,0,1]

        self.domain_interpolation(nbe,bc_val,nlfv,lef,rel,col,termo0,hp)

        col = rel.copy()

        hp = hpoints[nbe:,1,0]
        termo0 = - flux[nbe:,1,0]

        self.domain_interpolation(nbe,bc_val,nlfv,lef,rel,col,termo0,hp)

        hp = hpoints[nbe:,1,1]
        termo0 = - flux[nbe:,1,1]

        self.domain_interpolation(nbe,bc_val,nlfv,lef,rel,col,termo0,hp)

    def boundary_interpolation(self,nbe,mobility,bc_val,nlfv,lef,termo0,hp):

        IsNeu = bc_val.pressure.Neu_edges 
        IsDir = bc_val.pressure.Dir_edges
        value = bc_val.pressure.val_array

        interface = nlfv.misc.interface_elem
        W_DMP = nlfv.pressure.W_DMP

        InDir = np.isin(hp,IsDir)     
        if InDir.sum() > 0:

            val_array = termo0[InDir]*self.p_ij[hp[InDir],1]
            
            if self.I.shape[0] == 2:

                for edge in range(val_array.shape[0]):

                    self.I[lef[InDir][edge]] += val_array[edge]

            else:

                self.I[lef[InDir]] += val_array
                
        InDom = hp >= nbe
        if InDom.sum() > 0:

            elems = interface[hp[InDom]]
            weights = W_DMP[hp[InDom] - nbe]

            val_array_1 = -termo0[InDom]*weights[:,0]
            val_array_2 = -termo0[InDom]*weights[:,1]

            self.row.extend(lef[InDom])
            self.col.extend(elems[:,0])
            self.data.extend(val_array_1)

            self.row.extend(lef[InDom])
            self.col.extend(elems[:,1])
            self.data.extend(val_array_2)

        InNeu = np.isin(hp,IsNeu)
        if InNeu.sum() > 0:          

            M = self.p_ij[hp[InNeu],1]
            F = self.p_ij[hp[InNeu],2]

            val_array = -termo0[InNeu]*M

            self.row.extend(lef[InNeu])
            self.col.extend(lef[InNeu])
            self.data.extend(val_array)

            val_array = termo0[InNeu]*F

            self.I[lef[InNeu]] -= val_array

            aux_InDir = np.isin(self.p_ij[hp[InNeu],0],IsDir)
            if aux_InDir.sum() > 0:

                p = value[self.p_ij[hp[InNeu],0][aux_InDir].astype(int)]
                coeff = self.p_ij[hp[InNeu],3][aux_InDir]

                val_array = termo0[InNeu][aux_InDir]*coeff*p

                self.I[lef[InNeu][aux_InDir]] -= val_array

                for edge in range(val_array.shape[0]):

                        self.I[lef[InNeu][aux_InDir][edge]] -= val_array[edge]



                """if self.I.shape[0] == 2:

                    for edge in range(val_array.shape[0]):

                        self.I[lef[InNeu][aux_InDir][edge]] -= val_array[edge]

                else:

                    self.I[lef[InNeu][aux_InDir]] -= val_array"""

            aux_InDom = self.p_ij[hp[InNeu],0] >= nbe
            if aux_InDom.sum() > 0:
                
                hp_aux = self.p_ij[hp[InNeu],0][aux_InDom].astype(int)
                elems = interface[hp_aux]
                weights = W_DMP[hp_aux - nbe]
                coeff = self.p_ij[hp[InNeu],3][aux_InDom]

                val_array_1 = termo0[InNeu][aux_InDom]*coeff*weights[:,0]
                val_array_2 = termo0[InNeu][aux_InDom]*coeff*weights[:,1]
        
                self.row.extend(lef[InNeu][aux_InDom])
                self.col.extend(elems[:,0])
                self.data.extend(val_array_1)

                self.row.extend(lef[InNeu][aux_InDom])
                self.col.extend(elems[:,1])
                self.data.extend(val_array_2)

        pass
           
    def domain_interpolation(self,nbe,bc_val,nlfv,lef,rel,col,termo0,hp):
        
        IsNeu = bc_val.pressure.Neu_edges 
        IsDir = bc_val.pressure.Dir_edges 

        interface = nlfv.misc.interface_elem
        W_DMP = nlfv.pressure.W_DMP

        InDir = np.isin(hp,IsDir)
        if InDir.sum() > 0:

            val_array = termo0[InDir]*self.p_ij[hp[InDir],1]

            if self.I.shape[0] == 2:

                for edge in range(val_array.shape[0]):

                    self.I[lef[InDir][edge]] += val_array[edge]
                    self.I[rel[InDir][edge]] -= val_array[edge]

            else:

                self.I[lef[InDir]] += val_array
                self.I[rel[InDir]] -= val_array

        InDom = hp >= nbe
        if InDom.sum() > 0:

            elems = interface[hp[InDom]]
            weights = W_DMP[hp[InDom] - nbe]

            val_array_1 = -termo0[InDom]*weights[:,0]
            val_array_2 = -termo0[InDom]*weights[:,1]

            self.row.extend(lef[InDom])
            self.col.extend(elems[:,0])
            self.data.extend(val_array_1)

            self.row.extend(lef[InDom])
            self.col.extend(elems[:,1])
            self.data.extend(val_array_2)

            self.row.extend(rel[InDom])
            self.col.extend(elems[:,0])
            self.data.extend(- val_array_1)

            self.row.extend(rel[InDom])
            self.col.extend(elems[:,1])
            self.data.extend(- val_array_2)

        InNeu = np.isin(hp,IsNeu)
        if InNeu.sum() > 0:          

            M = self.p_ij[hp[InNeu],1]
    
            val_array = -termo0[InNeu]*M

            self.row.extend(lef[InNeu])
            self.col.extend(col[InNeu])
            self.data.extend(val_array)

            self.row.extend(rel[InNeu])
            self.col.extend(col[InNeu])
            self.data.extend(-val_array)

            F = self.p_ij[hp[InNeu],2]

            val_array = termo0[InNeu]*F

            self.I[lef[InNeu]] -= val_array

            aux_InDir = np.isin(self.p_ij[hp[InNeu],0],IsDir)
            if aux_InDir.sum() > 0:

                p = self.p_ij[hp[InNeu],1][aux_InDir]
                coeff = self.p_ij[hp[InNeu],3][aux_InDir]

                val_array = termo0[InNeu][aux_InDir]*coeff*p

                if self.I.shape[0] == 2:

                    for edge in range(val_array.shape[0]):

                        self.I[lef[InNeu][aux_InDir][edge]] -= val_array[edge]
                        self.I[rel[InNeu][aux_InDir][edge]] += val_array[edge]

                else:

                    self.I[lef[InNeu][aux_InDir]] -= val_array
                    self.I[rel[InNeu][aux_InDir]] += val_array

            aux_InDom = self.p_ij[hp[InNeu],0] >= nbe
            if aux_InDom.sum() > 0:

                hp_aux = self.p_ij[hp[InNeu],0][aux_InDom].astype(int)
                elems = interface[hp_aux]
                weights = W_DMP[hp_aux - nbe]
                coeff = self.p_ij[hp[InNeu],3][aux_InDom]

                val_array_1 = termo0[InNeu][aux_InDom]*coeff*weights[:,0]
                val_array_2 = termo0[InNeu][aux_InDom]*coeff*weights[:,1]
        
                self.row.extend(lef[InNeu][aux_InDom])
                self.col.extend(elems[:,0])
                self.data.extend(val_array_1)

                self.row.extend(lef[InNeu][aux_InDom])
                self.col.extend(elems[:,1])
                self.data.extend(val_array_2)

                self.row.extend(rel[InNeu][aux_InDom])
                self.col.extend(elems[:,0])
                self.data.extend(-val_array_1)

                self.row.extend(rel[InNeu][aux_InDom])
                self.col.extend(elems[:,1])
                self.data.extend(-val_array_2)

    def set_wells():
        pass

    def steady_solver(self,sol):

        sol.pressure.field_num = spsolve(self.M,self.I)

        pass

    def pressure_interp(self,mesh,bc_val,sol,nlfv):

        nbe = mesh.edges.boundary.shape[0]

        IsDir = bc_val.pressure.Dir_edges
        IsNeu = bc_val.pressure.Neu_edges
        bc_value = bc_val.pressure.val_array

        p = sol.pressure.field_num

        interface = nlfv.misc.interface_elem
        W_DMP = nlfv.pressure.W_DMP
        hpoints = nlfv.pressure.aux_point
        trans = nlfv.pressure.trans

        ipressure = (p[interface[nbe:]]*W_DMP).sum(axis = 1)
        bpressure = np.zeros(nbe)

        bpressure[IsDir] = self.p_ij[IsDir,1]

        M = self.p_ij[IsNeu,1]
        F = self.p_ij[IsNeu,2]
        coeff = self.p_ij[IsNeu,3]

        hp_aux = self.p_ij[IsNeu,0].astype(int)
        hp_value = np.zeros(hp_aux.shape[0])

        aux_InDir = np.isin(hp_aux,IsDir)
        if aux_InDir.sum() > 0:

            hp_value[aux_InDir] = self.p_ij[hp_aux[aux_InDir],1]

        aux_InDom = hp_aux >= nbe
        if aux_InDom.sum() > 0:
            elems = interface[hp_aux[aux_InDom]]
            weights = W_DMP[hp_aux[aux_InDom] - nbe]

            hp_value[aux_InDom] = (p[elems]*weights).sum(axis = 1)

        bpressure[IsNeu] = M*p[interface[IsNeu,0]] - coeff*hp_value - F

        self.edge_pressure = np.zeros(nbe + ipressure.shape[0])
        self.edge_pressure[:nbe] = bpressure
        self.edge_pressure[nbe:] = ipressure

        pass

    def flowrate(self,mesh,fluids,bc_val,sol,nlfv):

        nbe = mesh.edges.boundary.shape[0]
        nel = mesh.faces.center[:].shape[0]

        edges = mesh.edges.connectivities[:]
        coord = mesh.nodes.coords[:]
        esize = np.linalg.norm(coord[edges[:,1]]- coord[edges[:,0]], axis = 1)

        interface = nlfv.misc.interface_elem

        mobility = fluids.edge_mobility

        IsNeu = bc_val.pressure.Neu_edges
        IsDir = bc_val.pressure.Dir_edges
        value = bc_val.pressure.val_array

        econnec = nlfv.misc.edge_order

        W_DMP = nlfv.pressure.W_DMP
        trans = nlfv.pressure.trans
        hp = nlfv.pressure.aux_point

        p = sol.pressure.field_num

        flux = (esize*mobility)[:,np.newaxis,np.newaxis]*trans

        flux[nbe:,0] *= W_DMP[:,1,np.newaxis]
        flux[nbe:,1] *= W_DMP[:,0,np.newaxis]

        self.pressure_interp(mesh,bc_val,sol,nlfv)

        lflow1 = flux[:,0,0]*(p[interface[:,0]] - self.edge_pressure[hp][:,0,0])
        lflow2 = flux[:,0,1]*(p[interface[:,0]] - self.edge_pressure[hp][:,0,1])

        lflow = lflow1 + lflow2

        rflow1 = flux[:,1,0]*(p[interface[:,1]] - self.edge_pressure[hp][:,1,0])
        rflow2 = flux[:,1,1]*(p[interface[:,1]] - self.edge_pressure[hp][:,1,1])

        rflow = rflow1 + rflow2

        self.rate_edge = lflow - rflow

        self.rate_cell = self.rate_edge[econnec].sum(axis = 1)
        
        pass

class MPFAH_force:

    def __init__(self,mesh,rock,bc_val,nlfv):

        nbe = mesh.edges.boundary.shape[0]
        nel = mesh.faces.center[:].shape[0]

        edges = mesh.edges.connectivities[:]
        coord = mesh.nodes.coords[:]
        esize = np.linalg.norm(coord[edges[:,1]]- coord[edges[:,0]], axis = 1)

        interface = nlfv.misc.interface_elem

        W_DMP = nlfv.displ.W_DMP
        trans = nlfv.displ.trans

        self.row = []
        self.col = []
        self.data = []

        self.I = np.zeros(2*nel)

        self.I[0::2] = rock.source[0::2]*mesh.faces.area[:]
        self.I[1::2] = rock.source[1::2]*mesh.faces.area[:]

        self.boundary_displacemenet(mesh,bc_val,nlfv,esize)  

        IsNeu = bc_val.hdispl.Neu_edges 
        value = bc_val.hdispl.val_array

        val_array = esize[IsNeu]*value[IsNeu]

        for id in range(val_array.shape[0]):

            self.I[2*interface[IsNeu,0][id]] -= val_array[id]

        IsNeu = bc_val.vdispl.Neu_edges 
        value = bc_val.vdispl.val_array

        val_array = esize[IsNeu]*value[IsNeu]

        for id in range(val_array.shape[0]):
            
            self.I[2*interface[IsNeu,0][id] + 1] -= val_array[id]

        Fxx = esize[:,np.newaxis,np.newaxis]*trans.xx
        Fxy = esize[:,np.newaxis,np.newaxis]*trans.xy
        Fyx = esize[:,np.newaxis,np.newaxis]*trans.yx
        Fyy = esize[:,np.newaxis,np.newaxis]*trans.yy

        Fxx[nbe:,0] *= W_DMP[:,1,0,0,np.newaxis]
        Fxx[nbe:,1] *= W_DMP[:,0,0,0,np.newaxis]

        Fxy[nbe:,0] *= W_DMP[:,1,0,0,np.newaxis]
        Fxy[nbe:,1] *= W_DMP[:,0,0,0,np.newaxis]

        Fyx[nbe:,0] *= W_DMP[:,1,1,1,np.newaxis]
        Fyx[nbe:,1] *= W_DMP[:,0,1,1,np.newaxis]

        Fyy[nbe:,0] *= W_DMP[:,1,1,1,np.newaxis]
        Fyy[nbe:,1] *= W_DMP[:,0,1,1,np.newaxis]

        IsDir = bc_val.hdispl.Dir_edges

        xy = np.array([0,0])
        self.matrix_assebmly(nbe,IsDir,interface,Fxx,xy)

        xy = np.array([0,1])
        self.matrix_assebmly(nbe,IsDir,interface,Fxy,xy)
       
        IsDir = bc_val.vdispl.Dir_edges
        
        xy = np.array([1,0])
        self.matrix_assebmly(nbe,IsDir,interface,Fyx,xy)

        xy = np.array([1,1])
        self.matrix_assebmly(nbe,IsDir,interface,Fyy,xy)

        xy = np.array([0,0])
        IsDir = bc_val.hdispl.Dir_edges
        Var = bc_val.hdispl
        Var_aux = bc_val.vdispl
        U_ij = self.u_ij
        hpoints = nlfv.displ.aux_point.xx

        self.hpoints_interpolation(nbe,IsDir,Var,Var_aux,U_ij,hpoints,nlfv,Fxx,xy)

        xy = np.array([0,1])
        Var = bc_val.vdispl
        Var_aux = bc_val.hdispl
        U_ij = self.v_ij
        hpoints = nlfv.displ.aux_point.xy

        self.hpoints_interpolation(nbe,IsDir,Var,Var_aux,U_ij,hpoints,nlfv,Fxy,xy)

        xy = np.array([1,0])
        IsDir = bc_val.vdispl.Dir_edges
        Var = bc_val.hdispl
        Var_aux = bc_val.vdispl
        U_ij = self.u_ij
        hpoints = nlfv.displ.aux_point.yx

        self.hpoints_interpolation(nbe,IsDir,Var,Var_aux,U_ij,hpoints,nlfv,Fyx,xy)

        xy = np.array([1,1])
        Var = bc_val.vdispl
        Var_aux = bc_val.hdispl
        U_ij = self.v_ij
        hpoints = nlfv.displ.aux_point.yy

        self.hpoints_interpolation(nbe,IsDir,Var,Var_aux,U_ij,hpoints,nlfv,Fyy,xy)

        self.M = coo_matrix((self.data,(self.row,self.col)),shape=(2*nel,2*nel)).tocsr()
        
        del self.row, self.col, self.data

    def boundary_displacemenet(self,mesh,bc_val,nlfv,esize):

        nbe = mesh.edges.boundary.shape[0]
        shape = mesh.faces.classify_element[:].max() + 1

        trans = nlfv.displ.trans
        hpoints = nlfv.displ.aux_point

        IsNeux = bc_val.hdispl.Neu_edges
        valuex = bc_val.hdispl.val_array

        IsNeuy = bc_val.vdispl.Neu_edges
        valuey = bc_val.vdispl.val_array

        self.u_ij = np.zeros([nbe,3,3])
        self.v_ij = np.zeros([nbe,3,3])

        neu_edges = np.unique(np.hstack([IsNeux,IsNeuy]))
        for edge in neu_edges:

            neu_displ = np.array([np.isin(edge,IsNeux),
                                  np.isin(edge,IsNeuy)])

            ksi1 = trans.xx[edge,0]
            hp1 = hpoints.xx[edge,0]

            if hpoints.xx[edge,0,1] == edge:

                ksi1 = np.flip(ksi1)
                hp1 = np.flip(hp1)
            
            ksi2 = trans.xy[edge,0]
            hp2 = hpoints.xy[edge,0]

            if hpoints.xy[edge,0,1] == edge:

                ksi2 = np.flip(ksi2)
                hp2 = np.flip(hp2)
            
            ksi3 = trans.yx[edge,0]
            hp3 = hpoints.yx[edge,0]

            if hpoints.yx[edge,0,1] == edge:

                ksi3 = np.flip(ksi3)
                hp3 = np.flip(hp3)

            ksi4 = trans.yy[edge,0]
            hp4 = hpoints.yy[edge,0]

            if hpoints.yy[edge,0,1] == edge:

                ksi4 = np.flip(ksi4)
                hp4 = np.flip(hp4)

            if shape == 4:

                aux_points = np.array([hpoints.xx[edge,0],
                                       hpoints.xy[edge,0],
                                       hpoints.yx[edge,0],
                                       hpoints.yy[edge,0]])

                aux_ksi = np.array([trans.xx[edge,0],
                                    trans.xy[edge,0],
                                    trans.yx[edge,0],
                                    trans.yy[edge,0]])

                check = np.abs(aux_ksi) > aux_ksi.max()/1e6

                aux_points = np.unique(aux_points[check])

                neu_check = np.isin(aux_points,IsNeux) + np.isin(aux_points,IsNeuy)
                neu_hp = aux_points[neu_check]
                
                if neu_hp.shape[0] == 1:

                    if neu_displ[0]:

                        self.u_ij[edge,0] = np.array([ksi1.sum(),
                                                      ksi2.sum(),
                                                      valuex[edge]])/ksi1[0]

                        self.u_ij[edge,1] = np.array([ksi1[1],
                                                      ksi2[0],
                                                      ksi2[1]])/ksi1[0]

                        self.u_ij[edge,2] = np.array([hp1[1],
                                                      hp2[0],
                                                      hp2[1]])

                    if neu_displ[1]:

                        self.v_ij[edge,0] = np.array([ksi3.sum(),
                                                      ksi4.sum(),
                                                      valuey[edge]])/ksi4[0]

                        self.v_ij[edge,1] = np.array([ksi4[1],
                                                      ksi3[0],
                                                      ksi3[1]])/ksi4[0]

                        self.v_ij[edge,2] = np.array([hp4[1],
                                                      hp3[0],
                                                      hp3[1]])

                if neu_hp.shape[0] == 2:

                    aux_edge = neu_hp[neu_hp != edge][0]

                    neu_displ1 = neu_displ
                    neu_displ2 = np.array([np.isin(aux_edge,IsNeux),
                                           np.isin(aux_edge,IsNeuy)])

                    ksi1_aux = trans.xx[aux_edge,0]
                    hp_aux1 = hpoints.xx[aux_edge,0]

                    if hpoints.xx[aux_edge,0,1] == aux_edge:

                        ksi1_aux = np.flip(ksi1_aux)
                        hp_aux1 = np.flip(hp_aux1)
                    
                    ksi2_aux = trans.xy[aux_edge,0]
                    hp_aux2 = hpoints.xy[aux_edge,0]

                    if hpoints.xy[aux_edge,0,1] == aux_edge:

                        ksi2_aux = np.flip(ksi2_aux)
                        hp_aux2 = np.flip(hp_aux2)
                    
                    ksi3_aux = trans.yx[aux_edge,0]
                    hp_aux3 = hpoints.yx[aux_edge,0]

                    if hpoints.yx[aux_edge,0,1] == aux_edge:

                        ksi3_aux = np.flip(ksi3_aux)
                        hp_aux3 = np.flip(hp_aux3)

                    ksi4_aux = trans.yy[aux_edge,0]
                    hp_aux4 = hpoints.yy[aux_edge,0]

                    if hpoints.yy[aux_edge,0,1] == aux_edge:

                        ksi4_aux = np.flip(ksi4_aux)
                        hp_aux4 = np.flip(hp_aux4)

                    if neu_displ1[0] and neu_displ2[1]:
                        
                        A = np.array([[ksi1[0], ksi2[1]],
                                        [ksi3_aux[1],ksi4_aux[0]]])

                        B = np.zeros([2,3])
                        B[0,:] = np.array([ksi1[0],ksi2[1],valuex[edge]])
                        B[1,:] = np.array([ksi3_aux[1],ksi4_aux[0],valuey[aux_edge]])

                        X = np.linalg.solve(A,B)

                        self.u_ij[edge,0] = X[0,:]

                        self.u_ij[edge,1,:] = 0

                        self.u_ij[edge,2,:] = -1


                    elif neu_displ1[0]:
                        
                        self.u_ij[edge,0] = np.array([ksi1.sum(),
                                                        ksi2.sum(),
                                                        valuex[edge]])/ksi1[0]

                        self.u_ij[edge,1] = np.array([ksi1[1],
                                                        ksi2[0],
                                                        ksi2[1]])/ksi1[0]

                        self.u_ij[edge,2] = np.array([hp1[1],
                                                        hp2[0],
                                                        hp2[1]])

                    if neu_displ1[1] and neu_displ2[0]:

                        A = np.array([[ksi3[1], ksi4[0]],
                                        [ksi1_aux[0],ksi4_aux[1]]])

                        B = np.zeros([2,3])
                        B[0,:] = np.array([ksi3[1],ksi4[0],valuey[edge]])
                        B[1,:] = np.array([ksi1_aux[0],ksi4_aux[1],valuex[aux_edge]])

                        X = np.linalg.solve(A,B)

                        self.v_ij[edge,0] = X[1,:]

                        self.v_ij[edge,1,:] = 0

                        self.v_ij[edge,2,:] = -1

                    elif neu_displ1[1]:

                        self.v_ij[edge,0] = np.array([ksi3.sum(),
                                                        ksi4.sum(),
                                                        valuey[edge]])/ksi4[0]

                        self.v_ij[edge,1] = np.array([ksi4[1],
                                                        ksi3[0],
                                                        ksi3[1]])/ksi4[0]

                        self.v_ij[edge,2] = np.array([hp4[1],
                                                        hp3[0],
                                                        hp3[1]])




                        pass

                    

                    





                    pass

            if shape == 3:

                aux_points = np.unique(np.array([hpoints.xx[edge,0],
                                             hpoints.xy[edge,0],
                                             hpoints.yx[edge,0],
                                             hpoints.yy[edge,0]]))

                neu_check = np.isin(aux_points,IsNeux) + np.isin(aux_points,IsNeuy)
                neu_hp = aux_points[neu_check]

                if neu_displ.sum() == 1:

                    if neu_displ[0]:

                        self.u_ij[edge,0] = np.array([ksi1.sum(),
                                                    ksi2.sum(),
                                                    valuex[edge]])/ksi1[0]

                        self.u_ij[edge,1] = np.array([ksi1[1],
                                                    ksi2[0],
                                                    ksi2[1]])/ksi1[0]

                        self.u_ij[edge,2] = np.array([hp1[1],
                                                    hp2[0],
                                                    hp2[1]])

                    if neu_displ[1]:

                        self.v_ij[edge,0] = np.array([ksi3.sum(),
                                                    ksi4.sum(),
                                                    valuey[edge]])/ksi4[0]

                        self.v_ij[edge,1] = np.array([ksi4[1],
                                                    ksi3[0],
                                                    ksi3[1]])/ksi4[0]

                        self.v_ij[edge,2] = np.array([hp4[1],
                                                    hp3[0],
                                                    hp3[1]])

                if neu_displ.sum() == 2:

                    if neu_hp.shape[0] == 1:

                        A = np.array([[ksi1[0], ksi2[0]], [ksi3[0], ksi4[0]]])

                        B = np.zeros([2,5])

                        B[0,:] = np.array([ksi1.sum(),
                                        ksi2.sum(),
                                        valuex[edge],
                                        ksi1[1],
                                        ksi2[1]])

                        B[1,:] = np.array([ksi3.sum(),
                                        ksi4.sum(),
                                        valuey[edge],
                                        ksi3[1],
                                        ksi4[1]])

                        X = np.linalg.solve(A,B)

                        self.u_ij[edge,0] = X[0,:3]

                        self.u_ij[edge,1] = np.array([X[0,3], 0, X[0,4]])

                        self.u_ij[edge,2] = np.array([hp1[1],
                                                    -1,
                                                    hp2[1]])

                        self.v_ij[edge,0] = X[1,:3]

                        self.v_ij[edge,1] = np.array([X[1,3], 0, X[1,4]])

                        self.v_ij[edge,2] = np.array([hp3[1],
                                                    -1,
                                                    hp4[1]])

    def matrix_assebmly(self,nbe,IsDir,interface,Force,xy):

        lef = interface[IsDir,0]

        self.row.extend(2*lef + xy[0])
        self.col.extend(2*lef + xy[1])
        self.data.extend(- Force[IsDir,0].sum(axis = 1))

        lef = interface[nbe:,0]
        rel = interface[nbe:,1]

        lForce = - Force[nbe:,0].sum(axis = 1)
        rForce = - Force[nbe:,1].sum(axis = 1)

        self.row.extend(2*lef + xy[0])
        self.col.extend(2*lef + xy[1])
        self.data.extend(lForce)
        
        self.row.extend(2*lef + xy[0])
        self.col.extend(2*rel + xy[1])
        self.data.extend(-rForce)
        
        self.row.extend(2*rel + xy[0])
        self.col.extend(2*rel + xy[1])
        self.data.extend(rForce)

        self.row.extend(2*rel + xy[0])
        self.col.extend(2*lef + xy[1])
        self.data.extend(-lForce)

        pass

    def hpoints_interpolation(self,nbe,IsDir,Var,Var_aux,U_ij,hpoints,nlfv,Force,xy):
        
        interface = nlfv.misc.interface_elem
        
        lef = interface[IsDir,0]

        hp = hpoints[IsDir,0,0]
        termo0 = Force[IsDir,0,0]

        self.boundary_interpolation(nbe,Var,Var_aux,U_ij,nlfv,lef,termo0,hp,xy)

        hp = hpoints[IsDir,0,1]
        termo0 = Force[IsDir,0,1]

        self.boundary_interpolation(nbe,Var,Var_aux,U_ij,nlfv,lef,termo0,hp,xy)

        lef = interface[nbe:,0]
        rel = interface[nbe:,1]

        col = lef.copy()

        hp = hpoints[nbe:,0,0]
        termo0 = Force[nbe:,0,0]

        self.domain_interpolation(nbe,Var,Var_aux,U_ij,nlfv,lef,rel,col,termo0,hp,xy)

        hp = hpoints[nbe:,0,1]
        termo0 = Force[nbe:,0,1]

        self.domain_interpolation(nbe,Var,Var_aux,U_ij,nlfv,lef,rel,col,termo0,hp,xy)

        col = rel.copy()
        
        hp = hpoints[nbe:,1,0]
        termo0 = - Force[nbe:,1,0]

        self.domain_interpolation(nbe,Var,Var_aux,U_ij,nlfv,lef,rel,col,termo0,hp,xy)

        hp = hpoints[nbe:,1,1]
        termo0 = - Force[nbe:,1,1]

        self.domain_interpolation(nbe,Var,Var_aux,U_ij,nlfv,lef,rel,col,termo0,hp,xy)

    def boundary_interpolation(self,nbe,Var,Var_aux,U_ij,nlfv,lef,termo0,hp,xy):

        interface = nlfv.misc.interface_elem
        W_DMP = nlfv.displ.W_DMP

        IsDir = Var.Dir_edges
        IsNeu = Var.Neu_edges
        value = Var.val_array

        IsDir_aux = Var_aux.Dir_edges
        value_aux = Var_aux.val_array

        InDir = np.isin(hp,IsDir)     
        if InDir.sum() > 0:

            val_array = termo0[InDir]*value[hp[InDir]]
            
            for edge in range(val_array.shape[0]):
                
                self.I[2*lef[InDir][edge] + xy[0]] -= val_array[edge]

        InDom = hp >= nbe
        if InDom.sum() > 0:

            elems = interface[hp[InDom]]
            weights = W_DMP[hp[InDom] - nbe]

            val_array_1 = termo0[InDom]*((1 - xy[1])*weights[:,0,0,0] + xy[1]*weights[:,0,1,0])
            val_array_2 = termo0[InDom]*((1 - xy[1])*weights[:,1,0,0] + xy[1]*weights[:,1,1,0])

            self.row.extend(2*lef[InDom] + xy[0])
            self.col.extend(2*elems[:,0])
            self.data.extend(val_array_1)

            self.row.extend(2*lef[InDom] + xy[0])
            self.col.extend(2*elems[:,1])
            self.data.extend(val_array_2)

            val_array_1 = termo0[InDom]*((1 - xy[1])*weights[:,0,1,0] + xy[1]*weights[:,0,1,1])
            val_array_2 = termo0[InDom]*((1 - xy[1])*weights[:,0,1,0] + xy[1]*weights[:,1,1,1])

            self.row.extend(2*lef[InDom] + xy[0])
            self.col.extend(2*elems[:,0] + 1)
            self.data.extend(val_array_1)

            self.row.extend(2*lef[InDom] + xy[0])
            self.col.extend(2*elems[:,1] + 1)
            self.data.extend(val_array_2)

        InNeu = np.isin(hp,IsNeu)
        if InNeu.sum() > 0:

            Mu = U_ij[hp[InNeu],0,0]
            Mv = U_ij[hp[InNeu],0,1]
            F = U_ij[hp[InNeu],0,2]
            
            val_array = termo0[InNeu]*Mu

            self.row.extend(2*lef[InNeu] + xy[0])
            self.col.extend(2*lef[InNeu])
            self.data.extend(val_array)

            val_array = termo0[InNeu]*Mv

            self.row.extend(2*lef[InNeu] + xy[0])
            self.col.extend(2*lef[InNeu] + 1)
            self.data.extend(val_array)

            val_array = termo0[InNeu]*F

            for edge in range(val_array.shape[0]):
                
                self.I[2*lef[InNeu][edge] + xy[0]] -= val_array[edge]

            aux = U_ij[hp[InNeu],2,0].astype(int)
            coeff = U_ij[hp[InNeu],1,0]

            aux_InDir = np.isin(aux,IsDir)
            if aux_InDir.sum() > 0:

                val_array = termo0[InNeu][aux_InDir]*coeff[aux_InDir]*value[aux[aux_InDir]]
                
                for edge in range(val_array.shape[0]):

                    self.I[2*lef[InNeu][aux_InDir][edge] + xy[0]] += val_array[edge]

            aux_InDom = aux >= nbe
            if aux_InDom.sum() > 0:

                hp_aux = aux[aux_InDom]
                elems = interface[hp_aux]
                weights = W_DMP[hp_aux - nbe]

                val_array_1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*((1 - xy[1])*weights[:,0,0,0] + xy[1]*weights[:,0,1,0])
                val_array_2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*((1 - xy[1])*weights[:,1,0,0] + xy[1]*weights[:,1,1,0])

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,0])
                self.data.extend(-val_array_1)

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,1])
                self.data.extend(-val_array_2)

                val_array_1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*((1 - xy[1])*weights[:,0,1,0] + xy[1]*weights[:,0,1,1])
                val_array_2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*((1 - xy[1])*weights[:,0,1,0] + xy[1]*weights[:,0,1,1])

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,0] + 1)
                self.data.extend(-val_array_1)

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,1] + 1)
                self.data.extend(-val_array_2)
                
            aux = U_ij[hp[InNeu],2,1].astype(int)
            coeff = U_ij[hp[InNeu],1,1]

            aux_InDir = np.isin(aux,IsDir_aux)
            if aux_InDir.sum() > 0:

                val_array = termo0[InNeu][aux_InDir]*coeff[aux_InDir]*value_aux[aux[aux_InDir]]
                
                for edge in range(val_array.shape[0]):

                    self.I[2*lef[InNeu][aux_InDir][edge] + xy[0]] += val_array[edge]

            aux_InDom = aux >= nbe
            if aux_InDom.sum() > 0:

                hp_aux = aux[aux_InDom]
                elems = interface[hp_aux]
                weights = W_DMP[hp_aux - nbe]

                val_array_1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,0,0,0] + (1 - xy[1])*weights[:,0,1,0])
                val_array_2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,1,0,0] + (1 - xy[1])*weights[:,1,1,0])

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,0])
                self.data.extend(-val_array_1)

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,1])
                self.data.extend(-val_array_2)

                val_array_1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,0,1,0] + (1 - xy[1])*weights[:,0,1,1])
                val_array_2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,1,1,0] + (1 - xy[1])*weights[:,1,1,1])

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,0] + 1)
                self.data.extend(-val_array_1)

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,1] + 1)
                self.data.extend(-val_array_2)    

            aux = U_ij[hp[InNeu],2,2].astype(int)
            coeff = U_ij[hp[InNeu],1,2]

            aux_InDir = np.isin(aux,IsDir_aux)
            if aux_InDir.sum() > 0:

                val_array = termo0[InNeu][aux_InDir]*coeff[aux_InDir]*value_aux[aux[aux_InDir]]
                
                for edge in range(val_array.shape[0]):

                    self.I[2*lef[InNeu][aux_InDir][edge] + xy[0]] += val_array[edge]

            aux_InDom = aux >= nbe
            if aux_InDom.sum() > 0:

                hp_aux = aux[aux_InDom]
                elems = interface[hp_aux]
                weights = W_DMP[hp_aux - nbe]

                val_array_1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,0,0,0] + (1 - xy[1])*weights[:,0,1,0])
                val_array_2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,1,0,0] + (1 - xy[1])*weights[:,1,1,0])

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,0])
                self.data.extend(-val_array_1)

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,1])
                self.data.extend(-val_array_2)

                val_array_1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,0,1,0] + (1 - xy[1])*weights[:,0,1,1])
                val_array_2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,0,1,0] + (1 - xy[1])*weights[:,0,1,1])

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,0] + 1)
                self.data.extend(-val_array_1)

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,1] + 1)
                self.data.extend(-val_array_2)    
  
    def domain_interpolation(self,nbe,Var,Var_aux,U_ij,nlfv,lef,rel,col,termo0,hp,xy):

        interface = nlfv.misc.interface_elem
        W_DMP = nlfv.displ.W_DMP

        IsDir = Var.Dir_edges
        IsNeu = Var.Neu_edges
        value = Var.val_array

        IsDir_aux = Var_aux.Dir_edges
        value_aux = Var_aux.val_array
    
        InDir = np.isin(hp,IsDir)
        if InDir.sum() > 0:

            val_array = termo0[InDir]*value[hp[InDir]]
            for edge in range(val_array.shape[0]):

                self.I[2*lef[InDir][edge] + xy[0]] -= val_array[edge]
                self.I[2*rel[InDir][edge] + xy[0]] += val_array[edge]

        InDom = hp >= nbe
        if InDom.sum() > 0:

            elems = interface[hp[InDom]]
            weights = W_DMP[hp[InDom] - nbe]

            val_array_1 = termo0[InDom]*((1 - xy[1])*weights[:,0,0,0] + xy[1]*weights[:,0,1,0])
            val_array_2 = termo0[InDom]*((1 - xy[1])*weights[:,1,0,0] + xy[1]*weights[:,1,1,0])

            self.row.extend(2*lef[InDom] + xy[0])
            self.col.extend(2*elems[:,0])
            self.data.extend(val_array_1)

            self.row.extend(2*lef[InDom] + xy[0])
            self.col.extend(2*elems[:,1])
            self.data.extend(val_array_2)

            self.row.extend(2*rel[InDom] + xy[0])
            self.col.extend(2*elems[:,0])
            self.data.extend(-val_array_1)

            self.row.extend(2*rel[InDom] + xy[0])
            self.col.extend(2*elems[:,1])
            self.data.extend(-val_array_2)

            val_array_1 = termo0[InDom]*((1 - xy[1])*weights[:,0,1,0] + xy[1]*weights[:,0,1,1])
            val_array_2 = termo0[InDom]*((1 - xy[1])*weights[:,1,1,0] + xy[1]*weights[:,1,1,1])

            self.row.extend(2*lef[InDom] + xy[0])
            self.col.extend(2*elems[:,0] + 1)
            self.data.extend(val_array_1)

            self.row.extend(2*lef[InDom] + xy[0])
            self.col.extend(2*elems[:,1] + 1)
            self.data.extend(val_array_2)

            self.row.extend(2*rel[InDom] + xy[0])
            self.col.extend(2*elems[:,0] + 1)
            self.data.extend(-val_array_1)

            self.row.extend(2*rel[InDom] + xy[0])
            self.col.extend(2*elems[:,1] + 1)
            self.data.extend(-val_array_2)

        InNeu = np.isin(hp,IsNeu)
        if InNeu.sum() > 0:

            Mu = U_ij[hp[InNeu],0,0]
            Mv = U_ij[hp[InNeu],0,1]
            F = U_ij[hp[InNeu],0,2]
            
            val_array = termo0[InNeu]*Mu

            self.row.extend(2*lef[InNeu] + xy[0])
            self.col.extend(2*col[InNeu])
            self.data.extend(val_array)

            self.row.extend(2*rel[InNeu] + xy[0])
            self.col.extend(2*col[InNeu])
            self.data.extend(-val_array)

            val_array = termo0[InNeu]*Mv

            self.row.extend(2*lef[InNeu] + xy[0])
            self.col.extend(2*col[InNeu] + 1)
            self.data.extend(val_array)

            self.row.extend(2*rel[InNeu] + xy[0])
            self.col.extend(2*col[InNeu] + 1)
            self.data.extend(-val_array)

            val_array = termo0[InNeu]*F

            for edge in range(val_array.shape[0]):
                
                self.I[2*lef[InNeu][edge] + xy[0]] -= val_array[edge]
                self.I[2*rel[InNeu][edge] + xy[0]] += val_array[edge]

            aux = U_ij[hp[InNeu],2,0].astype(int)
            coeff = U_ij[hp[InNeu],1,0]

            aux_InDir = np.isin(aux,IsDir)
            if aux_InDir.sum() > 0:

                val_array = termo0[InNeu][aux_InDir]*coeff[aux_InDir]*value[aux[aux_InDir]]
                
                for edge in range(val_array.shape[0]):

                    self.I[2*lef[InNeu][aux_InDir][edge] + xy[0]] += val_array[edge]
                    self.I[2*rel[InNeu][aux_InDir][edge] + xy[0]] -= val_array[edge]

            aux_InDom = aux >= nbe
            if aux_InDom.sum() > 0:

                hp_aux = aux[aux_InDom]
                elems = interface[hp_aux]
                weights = W_DMP[hp_aux - nbe]

                val_array_1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*((1 - xy[1])*weights[:,0,0,0] + xy[1]*weights[:,0,1,0])
                val_array_2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*((1 - xy[1])*weights[:,1,0,0] + xy[1]*weights[:,1,1,0])

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,0])
                self.data.extend(-val_array_1)

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,1])
                self.data.extend(-val_array_2)

                self.row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,0])
                self.data.extend(val_array_1)

                self.row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,1])
                self.data.extend(val_array_2)

                val_array_1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*((1 - xy[1])*weights[:,0,1,0] + xy[1]*weights[:,0,1,1])
                val_array_2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*((1 - xy[1])*weights[:,1,1,0] + xy[1]*weights[:,1,1,1])

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,0] + 1)
                self.data.extend(-val_array_1)

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,1] + 1)
                self.data.extend(-val_array_2)

                self.row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,0] + 1)
                self.data.extend(val_array_1)

                self.row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,1] + 1)
                self.data.extend(val_array_2)
                
            aux = U_ij[hp[InNeu],2,1].astype(int)
            coeff = U_ij[hp[InNeu],1,1]

            aux_InDir = np.isin(aux,IsDir_aux)
            if aux_InDir.sum() > 0:

                val_array = termo0[InNeu][aux_InDir]*coeff[aux_InDir]*value_aux[aux[aux_InDir]]
                
                for edge in range(val_array.shape[0]):

                    self.I[2*lef[InNeu][aux_InDir][edge] + xy[0]] += val_array[edge]
                    self.I[2*rel[InNeu][aux_InDir][edge] + xy[0]] -= val_array[edge]

            aux_InDom = aux >= nbe
            if aux_InDom.sum() > 0:

                hp_aux = aux[aux_InDom]
                elems = interface[hp_aux]
                weights = W_DMP[hp_aux - nbe]

                val_array_1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,0,0,0] + (1 - xy[1])*weights[:,0,1,0])
                val_array_2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,1,0,0] + (1 - xy[1])*weights[:,1,1,0])

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,0])
                self.data.extend(-val_array_1)

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,1])
                self.data.extend(-val_array_2)

                self.row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,0])
                self.data.extend(val_array_1)

                self.row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,1])
                self.data.extend(val_array_2)

                val_array_1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,0,1,0] + (1 - xy[1])*weights[:,0,1,1])
                val_array_2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,1,1,0] + (1 - xy[1])*weights[:,1,1,1])

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,0] + 1)
                self.data.extend(-val_array_1)

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,1] + 1)
                self.data.extend(-val_array_2)  

                self.row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,0] + 1)
                self.data.extend(val_array_1)

                self.row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,1] + 1)
                self.data.extend(val_array_2)      

            aux = U_ij[hp[InNeu],2,2].astype(int)
            coeff = U_ij[hp[InNeu],1,2]

            aux_InDir = np.isin(aux,IsDir_aux)
            if aux_InDir.sum() > 0:

                val_array = termo0[InNeu][aux_InDir]*coeff[aux_InDir]*value_aux[aux[aux_InDir]]
                
                for edge in range(val_array.shape[0]):

                    self.I[2*lef[InNeu][aux_InDir][edge] + xy[0]] += val_array[edge]
                    self.I[2*rel[InNeu][aux_InDir][edge] + xy[0]] -= val_array[edge]

            aux_InDom = aux >= nbe
            if aux_InDom.sum() > 0:

                hp_aux = aux[aux_InDom]
                elems = interface[hp_aux]
                weights = W_DMP[hp_aux - nbe]

                val_array_1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,0,0,0] + (1 - xy[1])*weights[:,0,1,0])
                val_array_2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,1,0,0] + (1 - xy[1])*weights[:,1,1,0])

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,0])
                self.data.extend(-val_array_1)

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,1])
                self.data.extend(-val_array_2)

                self.row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,0])
                self.data.extend(val_array_1)

                self.row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,1])
                self.data.extend(val_array_2)

                val_array_1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,0,1,0] + (1 - xy[1])*weights[:,0,1,1])
                val_array_2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,1,1,0] + (1 - xy[1])*weights[:,1,1,1])

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,0] + 1)
                self.data.extend(-val_array_1)

                self.row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,1] + 1)
                self.data.extend(-val_array_2)

                self.row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,0] + 1)
                self.data.extend(val_array_1)

                self.row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.col.extend(2*elems[:,1] + 1)
                self.data.extend(val_array_2)    
      
    def steady_solver(self,sol):

        solution = spsolve(self.M,self.I)

        solution = np.where(np.abs(solution) < 1e-5, 0 , solution)

        sol.displacement.field_num[:,0] = solution[::2]
        sol.displacement.field_num[:,1] = solution[1::2]

        pass