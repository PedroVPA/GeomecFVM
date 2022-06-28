"""
UNIVERSIDADE FEDERAL DE PERNAMBUCO
CENTRO DE TECNOLOGIA E GEOCIENCIAS
PROGRAMA DE POS GRADUACAO EM ENGENHARIA MECÂNICA

Discentes: Pedro Albuquerque
           Danilo Maglhães
           Ricardo Emanuel
           Marcos Irandy
           Letônio

Docentes: Darlan Carvalho, Paulo Lyra.

File Author: Main -> Pedro Albuquerque
             Co 1->
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import time

class MPFAH:

    def __init__(self,mesh,fluids,wells,bc_val,misc_par,flux_par) -> None:

        nbe = mesh.edges.boundary.shape[0]
        nel = mesh.faces.center[:].shape[0]

        esize = misc_par.edge_sizes

        interface = misc_par.interface_elems

        mobility = fluids.edge_mobility

        IsNeu = bc_val.pressure.neu_edges 
        value = bc_val.pressure.bc_value

        W_DMP = flux_par.W_DMP
        trans = flux_par.trans

        self.M_row = []
        self.M_col = []
        self.M_data = []

        self.I_row = []
        self.I_data = []

        self.I_row.extend(interface[IsNeu,0])
        self.I_data.extend(esize[IsNeu]*value[IsNeu])

        self.boundary_pressure(mesh,mobility,bc_val,misc_par,flux_par)

        flux = (esize*mobility)[:,np.newaxis,np.newaxis]*trans

        flux[nbe:,0] *= W_DMP[:,1,np.newaxis]
        flux[nbe:,1] *= W_DMP[:,0,np.newaxis]

        self.centroid_coeff(mesh,bc_val,misc_par,flux_par,flux)
                
        self.hpoints_interpolation(mesh,mobility,bc_val,misc_par,flux_par,flux)

        self.set_wells(mesh,wells)

        self.matrix_assembly(nel)

        del self.M_row, self.M_col, self.M_data, self.I_row, self.I_data

    def boundary_pressure(self,mesh,mobility,bc_val,misc_par,flux_par):

        nbe = mesh.edges.boundary.shape[0]

        trans = flux_par.trans
        hpoints = flux_par.aux_point

        IsNeu = bc_val.pressure.neu_edges
        IsDir = bc_val.pressure.dir_edges
        value = bc_val.pressure.bc_value

        self.p_ij = np.zeros([nbe,4])

        self.p_ij[IsDir,0] = -1
        self.p_ij[IsDir,1] = value[IsDir]
        
        hp = hpoints[IsNeu,0]
        ksi = trans[IsNeu,0]
        
        hp_aux = np.where(hp[:,0] == IsNeu, hp[:,1], hp[:,0])
    
        ksi_neu = np.where(hp[:,0] == IsNeu, ksi[:,0], ksi[:,1])
        ksi_aux = np.where(hp[:,0] == IsNeu, ksi[:,1], ksi[:,0])

        aux_InNeu = np.isin(hp_aux,IsNeu)
        if aux_InNeu.sum() > 0:

            ksi1 = ksi_neu[aux_InNeu]            
            ksi2 = ksi_aux[aux_InNeu]

            opst_hp = hpoints[hp_aux[aux_InNeu],0]
            opst_ksi = trans[hp_aux[aux_InNeu],0]

            ksi3 = np.where(opst_hp[:,0] == IsNeu[aux_InNeu], opst_ksi[:,0], opst_ksi[:,1])
            ksi4 = np.where(opst_hp[:,0] == IsNeu[aux_InNeu], opst_ksi[:,1], opst_ksi[:,0])
            
            den = ksi1*ksi4 - ksi2*ksi3

            ksiA = ksi3*(ksi1 + ksi2)/den
            ksiB = ksi2*(ksi3 + ksi4)/den

            M = ksiA - ksiB

            self.p_ij[IsNeu[aux_InNeu],1] = M

            F = ksiA*value[IsNeu[aux_InNeu]] - ksiB*value[hp_aux[aux_InNeu]]

            self.p_ij[IsNeu[aux_InNeu],2] = F

        aux_not_InNeu = np.invert(aux_InNeu)
        if aux_not_InNeu.sum() > 0:

            ksi1 = ksi_neu[aux_not_InNeu]            
            ksi2 = ksi_aux[aux_not_InNeu]

            M = (ksi1 + ksi2)/ksi1

            self.p_ij[IsNeu[aux_not_InNeu],1] = M

            F = value[IsNeu[aux_not_InNeu]]/(mobility[IsNeu[aux_not_InNeu]]*ksi1)

            self.p_ij[IsNeu[aux_not_InNeu],2] = F

            P = ksi2/ksi1

            self.p_ij[IsNeu[aux_not_InNeu],3] = P 

        hp_aux = np.where(np.isin(hp_aux,IsNeu), -1 , hp_aux)

        self.p_ij[IsNeu,0] = hp_aux      

    def centroid_coeff(self,mesh,bc_val,misc_par,flux_par,flux):

        nbe = mesh.edges.boundary.shape[0]
        
        IsDir = bc_val.pressure.dir_edges

        interface = misc_par.interface_elems
        W_DMP = flux_par.W_DMP

        lef = interface[IsDir,0]

        self.M_row.extend(lef)
        self.M_col.extend(lef)
        self.M_data.extend(flux[IsDir,0].sum(axis = 1))

        lef = interface[nbe:,0]
        rel = interface[nbe:,1]

        lFlux = flux[nbe:,0].sum(axis = 1)
        rFlux = flux[nbe:,1].sum(axis = 1)

        self.M_row.extend(lef)
        self.M_col.extend(lef)
        self.M_data.extend(lFlux)
        
        self.M_row.extend(lef)
        self.M_col.extend(rel)
        self.M_data.extend(-rFlux)
        
        self.M_row.extend(rel)
        self.M_col.extend(rel)
        self.M_data.extend(rFlux)

        self.M_row.extend(rel)
        self.M_col.extend(lef)
        self.M_data.extend(-lFlux)

    def hpoints_interpolation(self,mesh,mobility,bc_val,misc_par,flux_par,flux):
        
        nbe = mesh.edges.boundary.shape[0]

        IsDir = bc_val.pressure.dir_edges 
    
        interface = misc_par.interface_elems
        hpoints = flux_par.aux_point
        
        lef = interface[IsDir,0]

        hp = hpoints[IsDir,0,0]
        termo0 = flux[IsDir,0,0]

        self.boundary_interpolation(nbe,mobility,bc_val,misc_par,flux_par,lef,termo0,hp)

        hp = hpoints[IsDir,0,1]
        termo0 = flux[IsDir,0,1]

        self.boundary_interpolation(nbe,mobility,bc_val,misc_par,flux_par,lef,termo0,hp)

        lef = interface[nbe:,0]
        rel = interface[nbe:,1]

        col = lef.copy()

        hp = hpoints[nbe:,0,0]
        termo0 = flux[nbe:,0,0]
        
        self.domain_interpolation(nbe,bc_val,misc_par,flux_par,lef,rel,col,termo0,hp)

        hp = hpoints[nbe:,0,1]
        termo0 = flux[nbe:,0,1]

        self.domain_interpolation(nbe,bc_val,misc_par,flux_par,lef,rel,col,termo0,hp)

        col = rel.copy()

        hp = hpoints[nbe:,1,0]
        termo0 = - flux[nbe:,1,0]

        self.domain_interpolation(nbe,bc_val,misc_par,flux_par,lef,rel,col,termo0,hp)

        hp = hpoints[nbe:,1,1]
        termo0 = - flux[nbe:,1,1]

        self.domain_interpolation(nbe,bc_val,misc_par,flux_par,lef,rel,col,termo0,hp)

    def boundary_interpolation(self,nbe,mobility,bc_val,misc_par,flux_par,lef,termo0,hp):

        IsNeu = bc_val.pressure.neu_edges 
        IsDir = bc_val.pressure.dir_edges
        value = bc_val.pressure.bc_value

        interface = misc_par.interface_elems
        W_DMP = flux_par.W_DMP

        InDir = np.isin(hp,IsDir)     
        if InDir.sum() > 0:

            val = termo0[InDir]*self.p_ij[hp[InDir],1]

            self.I_row.extend(lef[InDir])
            self.I_data.extend(val)
                
        InDom = hp >= nbe
        if InDom.sum() > 0:

            elems = interface[hp[InDom]]
            weights = W_DMP[hp[InDom] - nbe]

            val1 = termo0[InDom]*weights[:,0]
            val2 = termo0[InDom]*weights[:,1]

            self.M_row.extend(lef[InDom])
            self.M_col.extend(elems[:,0])
            self.M_data.extend(- val1)

            self.M_row.extend(lef[InDom])
            self.M_col.extend(elems[:,1])
            self.M_data.extend(- val2)

        InNeu = np.isin(hp,IsNeu)
        if InNeu.sum() > 0:          

            M = self.p_ij[hp[InNeu],1]

            val = termo0[InNeu]*M

            self.M_row.extend(lef[InNeu])
            self.M_col.extend(lef[InNeu])
            self.M_data.extend(- val)

            F = self.p_ij[hp[InNeu],2]

            val = termo0[InNeu]*F

            self.I_row.extend(lef[InNeu])
            self.I_data.extend(val)

            aux_InDir = np.isin(self.p_ij[hp[InNeu],0],IsDir)
            if aux_InDir.sum() > 0:

                p = value[self.p_ij[hp[InNeu],0][aux_InDir].astype(int)]
                coeff = self.p_ij[hp[InNeu],3][aux_InDir]

                val = termo0[InNeu][aux_InDir]*coeff*p

                self.I_row.extend(lef[InNeu][aux_InDir])
                self.I_data.extend(val)

            aux_InDom = self.p_ij[hp[InNeu],0] >= nbe
            if aux_InDom.sum() > 0:
                
                hp_aux = self.p_ij[hp[InNeu],0][aux_InDom].astype(int)
                elems = interface[hp_aux]
                weights = W_DMP[hp_aux - nbe]
                coeff = self.p_ij[hp[InNeu],3][aux_InDom]

                val1 = termo0[InNeu][aux_InDom]*coeff*weights[:,0]
                val2 = termo0[InNeu][aux_InDom]*coeff*weights[:,1]
        
                self.M_row.extend(lef[InNeu][aux_InDom])
                self.M_col.extend(elems[:,0])
                self.M_data.extend(val1)

                self.M_row.extend(lef[InNeu][aux_InDom])
                self.M_col.extend(elems[:,1])
                self.M_data.extend(val2)
           
    def domain_interpolation(self,nbe,bc_val,misc_par,flux_par,lef,rel,col,termo0,hp):
        
        IsNeu = bc_val.pressure.neu_edges 
        IsDir = bc_val.pressure.dir_edges 

        interface = misc_par.interface_elems
        W_DMP = flux_par.W_DMP

        InDir = np.isin(hp,IsDir)
        if InDir.sum() > 0:

            val = termo0[InDir]*self.p_ij[hp[InDir],1]

            self.I_row.extend(lef[InDir])
            self.I_data.extend(val)

            self.I_row.extend(rel[InDir])
            self.I_data.extend(- val)

        InDom = hp >= nbe
        if InDom.sum() > 0:

            elems = interface[hp[InDom]]
            weights = W_DMP[hp[InDom] - nbe]

            val1 = termo0[InDom]*weights[:,0]
            val2 = termo0[InDom]*weights[:,1]

            self.M_row.extend(lef[InDom])
            self.M_col.extend(elems[:,0])
            self.M_data.extend(- val1)

            self.M_row.extend(lef[InDom])
            self.M_col.extend(elems[:,1])
            self.M_data.extend(- val2)

            self.M_row.extend(rel[InDom])
            self.M_col.extend(elems[:,0])
            self.M_data.extend(val1)

            self.M_row.extend(rel[InDom])
            self.M_col.extend(elems[:,1])
            self.M_data.extend(val2)

        InNeu = np.isin(hp,IsNeu)
        if InNeu.sum() > 0:          

            M = self.p_ij[hp[InNeu],1]
    
            val = termo0[InNeu]*M

            self.M_row.extend(lef[InNeu])
            self.M_col.extend(col[InNeu])
            self.M_data.extend(- val)

            self.M_row.extend(rel[InNeu])
            self.M_col.extend(col[InNeu])
            self.M_data.extend(val)

            F = self.p_ij[hp[InNeu],2]

            val = termo0[InNeu]*F

            self.I_row.extend(lef[InNeu])
            self.I_data.extend(val)

            self.I_row.extend(rel[InNeu])
            self.I_data.extend(- val)

            aux_InDir = np.isin(self.p_ij[hp[InNeu],0],IsDir)
            if aux_InDir.sum() > 0:

                p = self.p_ij[hp[InNeu],1][aux_InDir]
                coeff = self.p_ij[hp[InNeu],3][aux_InDir]

                val = termo0[InNeu][aux_InDir]*coeff*p

                self.I_row.extend(lef[InNeu][aux_InDir])
                self.I_data.extend(- val)

                self.I_row.extend(rel[InNeu][aux_InDir])
                self.I_data.extend(val)

            aux_InDom = self.p_ij[hp[InNeu],0] >= nbe
            if aux_InDom.sum() > 0:

                hp_aux = self.p_ij[hp[InNeu],0][aux_InDom].astype(int)
                elems = interface[hp_aux]
                weights = W_DMP[hp_aux - nbe]
                coeff = self.p_ij[hp[InNeu],3][aux_InDom]

                val1 = termo0[InNeu][aux_InDom]*coeff*weights[:,0]
                val2 = termo0[InNeu][aux_InDom]*coeff*weights[:,1]
        
                self.M_row.extend(lef[InNeu][aux_InDom])
                self.M_col.extend(elems[:,0])
                self.M_data.extend(val1)

                self.M_row.extend(lef[InNeu][aux_InDom])
                self.M_col.extend(elems[:,1])
                self.M_data.extend(val2)

                self.M_row.extend(rel[InNeu][aux_InDom])
                self.M_col.extend(elems[:,0])
                self.M_data.extend(-val1)

                self.M_row.extend(rel[InNeu][aux_InDom])
                self.M_col.extend(elems[:,1])
                self.M_data.extend(-val2)

    def set_wells(self,mesh,wells):

        area = mesh.faces.area[:]

        if not wells.empty:

            if wells.rate is not None:

                rate = wells.rate.value
                elems = wells.rate.elems

                R = rate*area[elems]/area[elems].sum()
                
                self.I[elems] -= R

            if wells.pressure is not None:

                pwf = wells.pressure.value
                elems = wells.pressure.elems
                
                self.M_data = np.array(self.M_data)
                rowId = np.where(np.isin(self.M_row,elems))[0]
                self.M_data[rowId] = 0
                self.M_data = list(self.M_data)

                self.M_row.extend(elems)
                self.M_col.extend(elems)
                self.M_data.extend(np.ones(elems.shape[0]))

                self.I[elems] = pwf

    def matrix_assembly(self,nel):

        I_col = np.zeros(len(self.I_row))

        self.M = coo_matrix((self.M_data,(self.M_row,self.M_col)),shape=(nel,nel)).tocsr()
       
        self.I = coo_matrix((self.I_data,(self.I_row,I_col)),shape=(nel,1)).tocsr()

    def steady_solver(self,sol):

        nel = sol.pressure.field_num.shape[0]

        solution = spsolve(self.M,self.I)

        solution = np.where(np.abs(solution) < 1e-5, 0 , solution)

        sol.pressure.field_num = solution

    def pressure_interp(self,mesh,bc_val,p,misc_par,flux_par):

        nbe = mesh.edges.boundary.shape[0]

        IsDir = bc_val.pressure.dir_edges
        IsNeu = bc_val.pressure.neu_edges

        interface = misc_par.interface_elems
        W_DMP = flux_par.W_DMP

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

            hp_value[aux_InDom] = ipressure[hp_aux[aux_InDom] - nbe]

        bpressure[IsNeu] = M*p[interface[IsNeu,0]] - coeff*hp_value + F

        self.edge_pressure = np.zeros(nbe + ipressure.shape[0])
        self.edge_pressure[:nbe] = bpressure
        self.edge_pressure[nbe:] = ipressure

    def flowrate(self,mesh,fluids,bc_val,sol,misc_par,flux_par):

        nbe = mesh.edges.boundary.shape[0]
        nel = mesh.faces.center[:].shape[0]

        edges = mesh.edges.connectivities[:]
        coord = mesh.nodes.coords[:]
        esize = np.linalg.norm(coord[edges[:,1]]- coord[edges[:,0]], axis = 1)

        interface = misc_par.interface_elem

        mobility = fluids.edge_mobility

        IsNeu = bc_val.pressure.neu_edges
        IsDir = bc_val.pressure.dir_edges
        value = bc_val.pressure.bc_value

        econnec = misc_par.edge_order

        W_DMP = flux_par.W_DMP
        trans = flux_par.trans
        hp = flux_par.aux_point

        p = sol.pressure.field_num

        flux = (esize*mobility)[:,np.newaxis,np.newaxis]*trans

        flux[nbe:,0] *= W_DMP[:,1,np.newaxis]
        flux[nbe:,1] *= W_DMP[:,0,np.newaxis]

        self.pressure_interp(mesh,bc_val,sol,misc_par,flux_par)

        lflow1 = flux[:,0,0]*(p[interface[:,0]] - self.edge_pressure[hp][:,0,0])
        lflow2 = flux[:,0,1]*(p[interface[:,0]] - self.edge_pressure[hp][:,0,1])

        lflow = lflow1 + lflow2

        rflow1 = flux[:,1,0]*(p[interface[:,1]] - self.edge_pressure[hp][:,1,0])
        rflow2 = flux[:,1,1]*(p[interface[:,1]] - self.edge_pressure[hp][:,1,1])

        rflow = rflow1 + rflow2

        self.rate_edge = lflow - rflow

        self.rate_cell = self.rate_edge[econnec].sum(axis = 1)

class MPSAH:

    def __init__(self,mesh,rock,bc_val,misc_par,stress_par) -> None:

        nbe = mesh.edges.boundary.shape[0]
        nel = mesh.faces.center[:].shape[0]

        esize = misc_par.edge_sizes

        interface = misc_par.interface_elems

        W_DMP = stress_par.W_DMP
        trans = stress_par.trans

        self.M_row = []
        self.M_col = []
        self.M_data = []

        self.I_row = []
        self.I_data = []

        self.boundary_displacemenet(mesh,bc_val,stress_par)

        IsNeu = bc_val.hdispl.neu_edges 
        value = bc_val.hdispl.bc_value

        val = esize[IsNeu]*value[IsNeu]

        self.I_row.extend(2*interface[IsNeu,0])
        self.I_data.extend(- val)

        IsNeu = bc_val.vdispl.neu_edges 
        value = bc_val.vdispl.bc_value

        val = esize[IsNeu]*value[IsNeu]

        self.I_row.extend(2*interface[IsNeu,0] + 1)
        self.I_data.extend(- val)

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

        IsDir = bc_val.hdispl.dir_edges

        xy = np.array([0,0])
        self.centroid_coeff(nbe,IsDir,interface,Fxx,xy)

        xy = np.array([0,1])
        self.centroid_coeff(nbe,IsDir,interface,Fxy,xy)
       
        IsDir = bc_val.vdispl.dir_edges
        
        xy = np.array([1,0])
        self.centroid_coeff(nbe,IsDir,interface,Fyx,xy)

        xy = np.array([1,1])
        self.centroid_coeff(nbe,IsDir,interface,Fyy,xy)

        xy = np.array([0,0])
        IsDir = bc_val.hdispl.dir_edges
        Var = bc_val.hdispl
        Var_aux = bc_val.vdispl
        U_ij = self.u_ij
        hpoints = stress_par.aux_point.xx

        self.hpoints_interpolation(nbe,IsDir,Var,Var_aux,U_ij,hpoints,misc_par,stress_par,Fxx,xy)

        xy = np.array([0,1])
        Var = bc_val.vdispl
        Var_aux = bc_val.hdispl
        U_ij = self.v_ij
        hpoints = stress_par.aux_point.xy

        self.hpoints_interpolation(nbe,IsDir,Var,Var_aux,U_ij,hpoints,misc_par,stress_par,Fxy,xy)

        xy = np.array([1,0])
        IsDir = bc_val.vdispl.dir_edges
        Var = bc_val.hdispl
        Var_aux = bc_val.vdispl
        U_ij = self.u_ij
        hpoints = stress_par.aux_point.yx

        self.hpoints_interpolation(nbe,IsDir,Var,Var_aux,U_ij,hpoints,misc_par,stress_par,Fyx,xy)

        xy = np.array([1,1])
        Var = bc_val.vdispl
        Var_aux = bc_val.hdispl
        U_ij = self.v_ij
        hpoints = stress_par.aux_point.yy

        self.hpoints_interpolation(nbe,IsDir,Var,Var_aux,U_ij,hpoints,misc_par,stress_par,Fyy,xy)

        self.matrix_assembly(nel)

        del self.M_row, self.M_col, self.M_data,self.I_row, self.I_data

    def boundary_displacemenet(self,mesh,bc_val,stress_par):

        nbe = mesh.edges.boundary.shape[0]
        shape = mesh.faces.classify_element[:].max() + 1

        trans = stress_par.trans
        hpoints = stress_par.aux_point

        IsNeux = bc_val.hdispl.neu_edges
        valuex = bc_val.hdispl.bc_value

        IsNeuy = bc_val.vdispl.neu_edges
        valuey = bc_val.vdispl.bc_value

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

                    if hpoints.xx[aux_edge,0,1] == edge:

                        ksi1_aux = np.flip(ksi1_aux)
                        hp_aux1 = np.flip(hp_aux1)
                    
                    ksi2_aux = trans.xy[aux_edge,0]
                    hp_aux2 = hpoints.xy[aux_edge,0]

                    if hpoints.xy[aux_edge,0,1] == edge:

                        ksi2_aux = np.flip(ksi2_aux)
                        hp_aux2 = np.flip(hp_aux2)
                    
                    ksi3_aux = trans.yx[aux_edge,0]
                    hp_aux3 = hpoints.yx[aux_edge,0]

                    if hpoints.yx[aux_edge,0,1] == edge:

                        ksi3_aux = np.flip(ksi3_aux)
                        hp_aux3 = np.flip(hp_aux3)

                    ksi4_aux = trans.yy[aux_edge,0]
                    hp_aux4 = hpoints.yy[aux_edge,0]

                    if hpoints.yy[aux_edge,0,1] == edge:

                        ksi4_aux = np.flip(ksi4_aux)
                        hp_aux4 = np.flip(hp_aux4)

                    if aux_points.shape[0] < 3 and check.sum() > 7 :

                        H = np.zeros([4,4])
                        H[:2] = np.array([hp1,hp2,hp3,hp4]).reshape(2,4)
                        H[2:] = np.array([hp_aux1,hp_aux2,hp_aux3,hp_aux4]).reshape(2,4)
                        H[:,[1,2]] = H[:,[2,1]]

                        A = np.zeros([4,4])
                        A[:2] = np.array([ksi1,ksi2,ksi3,ksi4]).reshape(2,4)
                        A[2:] = np.array([ksi1_aux,ksi2_aux,ksi3_aux,ksi4_aux]).reshape(2,4)
                        A[:,[1,2]] = A[:,[2,1]]

                        B = np.zeros([4,3])
                        B[0] = np.array([ksi1.sum(),ksi2.sum(),valuex[edge]])
                        B[1] = np.array([ksi3.sum(),ksi4.sum(),valuey[edge]])
                        B[2] = np.array([ksi1_aux.sum(),ksi2_aux.sum(),valuex[aux_edge]])
                        B[3] = np.array([ksi3_aux.sum(),ksi4_aux.sum(),valuey[aux_edge]])

                        displ_unknown = np.hstack([neu_displ1,neu_displ2])
                        displ_known = np.invert(displ_unknown)

                        U = np.zeros([4,3 + displ_known.sum()])

                        C = A[:,displ_known]
                        B = np.hstack([B,C])
                        U[displ_unknown] = np.linalg.solve(A[:,displ_unknown][displ_unknown,:],B[displ_unknown])

                        H[displ_known] = -1
                        displ_known[[1,2]] = displ_known[[2,1]]

                        if displ_unknown[0]:

                            self.u_ij[edge,0] = U[0,:3]

                            self.u_ij[edge,1][displ_known[1:]] = U[0,3:]

                            self.u_ij[edge,2][displ_known[1:]] = H[0,displ_known]

                        if displ_unknown[1]:
                            
                            self.v_ij[edge,0] = U[1,:3]

                            self.v_ij[edge,1][displ_known[::-1][1:]] = U[1,3:][::-1]

                            self.v_ij[edge,2][displ_known[::-1][1:]] = H[1,displ_known][::-1]

                    else:

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

            if shape == 3:

                aux_points = np.unique(np.array([hpoints.xx[edge,0],
                                             hpoints.xy[edge,0],
                                             hpoints.yx[edge,0],
                                             hpoints.yy[edge,0]]))

                neu_check = np.isin(aux_points,IsNeux) + np.isin(aux_points,IsNeuy)
                neu_hp = aux_points[neu_check]

                if neu_hp.shape[0] == 1:

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

                        self.v_ij[edge,1] = np.array([X[1,4], 0, X[1,3]])

                        self.v_ij[edge,2] = np.array([hp4[1],
                                                    -1,
                                                    hp3[1]])

                if neu_hp.shape[0] == 2:

                    aux_edge = neu_hp[neu_hp != edge][0]

                    neu_displ1 = neu_displ
                    neu_displ2 = np.array([np.isin(aux_edge,IsNeux),
                                           np.isin(aux_edge,IsNeuy)])
                    
                    ksi1_aux = trans.xx[aux_edge,0]
                    hp_aux1 = hpoints.xx[aux_edge,0]

                    if hpoints.xx[aux_edge,0,1] == edge:

                        ksi1_aux = np.flip(ksi1_aux)
                        hp_aux1 = np.flip(hp_aux1)
                    
                    ksi2_aux = trans.xy[aux_edge,0]
                    hp_aux2 = hpoints.xy[aux_edge,0]

                    if hpoints.xy[aux_edge,0,1] == edge:

                        ksi2_aux = np.flip(ksi2_aux)
                        hp_aux2 = np.flip(hp_aux2)
                    
                    ksi3_aux = trans.yx[aux_edge,0]
                    hp_aux3 = hpoints.yx[aux_edge,0]

                    if hpoints.yx[aux_edge,0,1] == edge:

                        ksi3_aux = np.flip(ksi3_aux)
                        hp_aux3 = np.flip(hp_aux3)

                    ksi4_aux = trans.yy[aux_edge,0]
                    hp_aux4 = hpoints.yy[aux_edge,0]

                    if hpoints.yy[aux_edge,0,1] == edge:

                        ksi4_aux = np.flip(ksi4_aux)
                        hp_aux4 = np.flip(hp_aux4)

                    H = np.zeros([4,4])
                    H[:2] = np.array([hp1,hp2,hp3,hp4]).reshape(2,4)
                    H[2:] = np.array([hp_aux1,hp_aux2,hp_aux3,hp_aux4]).reshape(2,4)
                    H[:,[1,2]] = H[:,[2,1]]

                    A = np.zeros([4,4])
                    A[:2] = np.array([ksi1,ksi2,ksi3,ksi4]).reshape(2,4)
                    A[2:] = np.array([ksi1_aux,ksi2_aux,ksi3_aux,ksi4_aux]).reshape(2,4)
                    A[:,[1,2]] = A[:,[2,1]]

                    B = np.zeros([4,3])
                    B[0] = np.array([ksi1.sum(),ksi2.sum(),valuex[edge]])
                    B[1] = np.array([ksi3.sum(),ksi4.sum(),valuey[edge]])
                    B[2] = np.array([ksi1_aux.sum(),ksi2_aux.sum(),valuex[aux_edge]])
                    B[3] = np.array([ksi3_aux.sum(),ksi4_aux.sum(),valuey[aux_edge]])

                    displ_unknown = np.hstack([neu_displ1,neu_displ2])
                    displ_known = np.invert(displ_unknown)

                    U = np.zeros([4,3 + displ_known.sum()])

                    C = A[:,displ_known]
                    B = np.hstack([B,C])
                    U[displ_unknown] = np.linalg.solve(A[:,displ_unknown][displ_unknown,:],B[displ_unknown])

                    H[displ_known] = -1
                    displ_known[[1,2]] = displ_known[[2,1]]

                    if displ_unknown[0]:

                        self.u_ij[edge,0] = U[0,:3]

                        self.u_ij[edge,1][displ_known[1:]] = U[0,3:]

                        self.u_ij[edge,2][displ_known[1:]] = H[0,displ_known]

                    if displ_unknown[1]:
                        
                        self.v_ij[edge,0] = U[1,:3]

                        self.v_ij[edge,1][displ_known[::-1][1:]] = U[1,3:][::-1]

                        self.v_ij[edge,2][displ_known[::-1][1:]] = H[1,displ_known][::-1]
       

        print('lmao')

    def boundary_displacemenet1(self,mesh,bc_val,stress_par):

        nbe = mesh.edges.boundary.shape[0]
        shape = mesh.faces.classify_element[:].max() + 1

        trans = stress_par.trans
        hpoints = stress_par.aux_point

        IsNeux = bc_val.hdispl.neu_edges
        valuex = bc_val.hdispl.bc_value

        IsNeuy = bc_val.vdispl.neu_edges
        valuey = bc_val.vdispl.bc_value

        self.u_ij = np.zeros([nbe,3,3])
        self.v_ij = np.zeros([nbe,3,3])

        neu_edges = np.unique(np.hstack([IsNeux,IsNeuy]))
        for edge in neu_edges:

            unknown_displ = np.array([np.isin(edge,IsNeux),
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

            aux_points = np.array([hp1, hp2, hp3, hp4])

            aux_edges = np.unique(aux_points)

            aux_edges = aux_edges[aux_edges != edge]

            aux_InNeu = np.isin(aux_edges,neu_edges)

            if aux_InNeu.sum() == 0:
                
                if unknown_displ.sum() == 1:

                    if unknown_displ[0]:

                        self.u_ij[edge,0] = np.array([ksi1.sum(),
                                                      ksi2.sum(),
                                                      valuex[edge]])/ksi1[0]

                        self.u_ij[edge,1] = np.array([ksi1[1],
                                                      ksi2[0],
                                                      ksi2[1]])/ksi1[0]

                        self.u_ij[edge,2] = np.array([hp1[1],
                                                      hp2[0],
                                                      hp2[1]])

                    if unknown_displ[1]:

                        self.v_ij[edge,0] = np.array([ksi3.sum(),
                                                      ksi4.sum(),
                                                      valuey[edge]])/ksi4[0]

                        self.v_ij[edge,1] = np.array([ksi4[1],
                                                      ksi3[0],
                                                      ksi3[1]])/ksi4[0]

                        self.v_ij[edge,2] = np.array([hp4[1],
                                                      hp3[0],
                                                      hp3[1]])

                if unknown_displ.sum() == 2:
                    
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

                    self.v_ij[edge,1] = np.array([X[1,4], 0, X[1,3]])

                    self.v_ij[edge,2] = np.array([hp4[1],
                                                -1,
                                                hp3[1]])

            else:
                pass

            print('lmao')

    def centroid_coeff(self,nbe,IsDir,interface,Force,xy):

        lef = interface[IsDir,0]

        self.M_row.extend(2*lef + xy[0])
        self.M_col.extend(2*lef + xy[1])
        self.M_data.extend(- Force[IsDir,0].sum(axis = 1))

        lef = interface[nbe:,0]
        rel = interface[nbe:,1]

        lForce = - Force[nbe:,0].sum(axis = 1)
        rForce = - Force[nbe:,1].sum(axis = 1)

        self.M_row.extend(2*lef + xy[0])
        self.M_col.extend(2*lef + xy[1])
        self.M_data.extend(lForce)
        
        self.M_row.extend(2*lef + xy[0])
        self.M_col.extend(2*rel + xy[1])
        self.M_data.extend(-rForce)
        
        self.M_row.extend(2*rel + xy[0])
        self.M_col.extend(2*rel + xy[1])
        self.M_data.extend(rForce)

        self.M_row.extend(2*rel + xy[0])
        self.M_col.extend(2*lef + xy[1])
        self.M_data.extend(-lForce)

        pass

    def hpoints_interpolation(self,nbe,IsDir,Var,Var_aux,U_ij,hpoints,misc_par,stress_par,Force,xy):
        
        interface = misc_par.interface_elems
        
        lef = interface[IsDir,0]

        hp = hpoints[IsDir,0,0]
        termo0 = Force[IsDir,0,0]

        self.boundary_interpolation(nbe,Var,Var_aux,U_ij,interface,stress_par,lef,termo0,hp,xy)

        hp = hpoints[IsDir,0,1]
        termo0 = Force[IsDir,0,1]

        self.boundary_interpolation(nbe,Var,Var_aux,U_ij,interface,stress_par,lef,termo0,hp,xy)

        lef = interface[nbe:,0]
        rel = interface[nbe:,1]

        col = lef.copy()

        hp = hpoints[nbe:,0,0]
        termo0 = Force[nbe:,0,0]

        self.domain_interpolation(nbe,Var,Var_aux,U_ij,interface,stress_par,lef,rel,col,termo0,hp,xy)

        hp = hpoints[nbe:,0,1]
        termo0 = Force[nbe:,0,1]

        self.domain_interpolation(nbe,Var,Var_aux,U_ij,interface,stress_par,lef,rel,col,termo0,hp,xy)

        col = rel.copy()
        
        hp = hpoints[nbe:,1,0]
        termo0 = - Force[nbe:,1,0]

        self.domain_interpolation(nbe,Var,Var_aux,U_ij,interface,stress_par,lef,rel,col,termo0,hp,xy)

        hp = hpoints[nbe:,1,1]
        termo0 = - Force[nbe:,1,1]

        self.domain_interpolation(nbe,Var,Var_aux,U_ij,interface,stress_par,lef,rel,col,termo0,hp,xy)

    def boundary_interpolation(self,nbe,Var,Var_aux,U_ij,interface,stress_par,lef,termo0,hp,xy):

        W_DMP = stress_par.W_DMP

        IsDir = Var.dir_edges
        IsNeu = Var.neu_edges
        value = Var.bc_value

        IsDir_aux = Var_aux.dir_edges
        value_aux = Var_aux.bc_value

        InDir = np.isin(hp,IsDir)     
        if InDir.sum() > 0:

            val = termo0[InDir]*value[hp[InDir]]

            self.I_row.extend(2*lef[InDir] + xy[0])
            self.I_data.extend(- val)

        InDom = hp >= nbe
        if InDom.sum() > 0:

            elems = interface[hp[InDom]]
            weights = W_DMP[hp[InDom] - nbe]

            val1 = termo0[InDom]*((1 - xy[1])*weights[:,0,0,0] + xy[1]*weights[:,0,1,0])
            val2 = termo0[InDom]*((1 - xy[1])*weights[:,1,0,0] + xy[1]*weights[:,1,1,0])

            self.M_row.extend(2*lef[InDom] + xy[0])
            self.M_col.extend(2*elems[:,0])
            self.M_data.extend(val1)

            self.M_row.extend(2*lef[InDom] + xy[0])
            self.M_col.extend(2*elems[:,1])
            self.M_data.extend(val2)

            val1 = termo0[InDom]*((1 - xy[1])*weights[:,0,1,0] + xy[1]*weights[:,0,1,1])
            val2 = termo0[InDom]*((1 - xy[1])*weights[:,0,1,0] + xy[1]*weights[:,1,1,1])

            self.M_row.extend(2*lef[InDom] + xy[0])
            self.M_col.extend(2*elems[:,0] + 1)
            self.M_data.extend(val1)

            self.M_row.extend(2*lef[InDom] + xy[0])
            self.M_col.extend(2*elems[:,1] + 1)
            self.M_data.extend(val2)

        InNeu = np.isin(hp,IsNeu)
        if InNeu.sum() > 0:

            Mu = U_ij[hp[InNeu],0,0]
            Mv = U_ij[hp[InNeu],0,1]
            F = U_ij[hp[InNeu],0,2]
            
            val = termo0[InNeu]*Mu

            self.M_row.extend(2*lef[InNeu] + xy[0])
            self.M_col.extend(2*lef[InNeu])
            self.M_data.extend(val)

            val = termo0[InNeu]*Mv

            self.M_row.extend(2*lef[InNeu] + xy[0])
            self.M_col.extend(2*lef[InNeu] + 1)
            self.M_data.extend(val)

            val = termo0[InNeu]*F

            self.I_row.extend(2*lef[InNeu] + xy[0])
            self.I_data.extend(- val)

            aux = U_ij[hp[InNeu],2,0].astype(int)
            coeff = U_ij[hp[InNeu],1,0]

            aux_InDir = np.isin(aux,IsDir)
            if aux_InDir.sum() > 0:

                val = termo0[InNeu][aux_InDir]*coeff[aux_InDir]*value[aux[aux_InDir]]
                
                self.I_row.extend(2*lef[InNeu][aux_InDir] + xy[0])
                self.I_data.extend(val)

            aux_InDom = aux >= nbe
            if aux_InDom.sum() > 0:

                hp_aux = aux[aux_InDom]
                elems = interface[hp_aux]
                weights = W_DMP[hp_aux - nbe]

                val1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*((1 - xy[1])*weights[:,0,0,0] + xy[1]*weights[:,0,1,0])
                val2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*((1 - xy[1])*weights[:,1,0,0] + xy[1]*weights[:,1,1,0])

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,0])
                self.M_data.extend(-val1)

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,1])
                self.M_data.extend(-val2)

                val1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*((1 - xy[1])*weights[:,0,1,0] + xy[1]*weights[:,0,1,1])
                val2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*((1 - xy[1])*weights[:,0,1,0] + xy[1]*weights[:,0,1,1])

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,0] + 1)
                self.M_data.extend(-val1)

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,1] + 1)
                self.M_data.extend(-val2)
                
            aux = U_ij[hp[InNeu],2,1].astype(int)
            coeff = U_ij[hp[InNeu],1,1]

            aux_InDir = np.isin(aux,IsDir_aux)
            if aux_InDir.sum() > 0:

                val = termo0[InNeu][aux_InDir]*coeff[aux_InDir]*value_aux[aux[aux_InDir]]
                
                self.I_row.extend(2*lef[InNeu][aux_InDir] + xy[0])
                self.I_data.extend(val)

            aux_InDom = aux >= nbe
            if aux_InDom.sum() > 0:

                hp_aux = aux[aux_InDom]
                elems = interface[hp_aux]
                weights = W_DMP[hp_aux - nbe]

                val1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,0,0,0] + (1 - xy[1])*weights[:,0,1,0])
                val2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,1,0,0] + (1 - xy[1])*weights[:,1,1,0])

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,0])
                self.M_data.extend(-val1)

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,1])
                self.M_data.extend(-val2)

                val1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,0,1,0] + (1 - xy[1])*weights[:,0,1,1])
                val2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,1,1,0] + (1 - xy[1])*weights[:,1,1,1])

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,0] + 1)
                self.M_data.extend(-val1)

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,1] + 1)
                self.M_data.extend(-val2)    

            aux = U_ij[hp[InNeu],2,2].astype(int)
            coeff = U_ij[hp[InNeu],1,2]

            aux_InDir = np.isin(aux,IsDir_aux)
            if aux_InDir.sum() > 0:

                val = termo0[InNeu][aux_InDir]*coeff[aux_InDir]*value_aux[aux[aux_InDir]]
                
                self.I_row.extend(2*lef[InNeu][aux_InDir] + xy[0])
                self.I_data.extend(val)

            aux_InDom = aux >= nbe
            if aux_InDom.sum() > 0:

                hp_aux = aux[aux_InDom]
                elems = interface[hp_aux]
                weights = W_DMP[hp_aux - nbe]

                val1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,0,0,0] + (1 - xy[1])*weights[:,0,1,0])
                val2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,1,0,0] + (1 - xy[1])*weights[:,1,1,0])

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,0])
                self.M_data.extend(-val1)

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,1])
                self.M_data.extend(-val2)

                val1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,0,1,0] + (1 - xy[1])*weights[:,0,1,1])
                val2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,0,1,0] + (1 - xy[1])*weights[:,0,1,1])

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,0] + 1)
                self.M_data.extend(-val1)

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,1] + 1)
                self.M_data.extend(-val2)    
  
    def domain_interpolation(self,nbe,Var,Var_aux,U_ij,interface,stress_par,lef,rel,col,termo0,hp,xy):

        W_DMP = stress_par.W_DMP

        IsDir = Var.dir_edges
        IsNeu = Var.neu_edges
        value = Var.bc_value

        IsDir_aux = Var_aux.dir_edges
        value_aux = Var_aux.bc_value
    
        InDir = np.isin(hp,IsDir)
        if InDir.sum() > 0:

            val = termo0[InDir]*value[hp[InDir]]

            self.I_row.extend(2*lef[InDir] + xy[0])
            self.I_data.extend(- val)

            self.I_row.extend(2*rel[InDir] + xy[0])
            self.I_data.extend(val)

        InDom = hp >= nbe
        if InDom.sum() > 0:

            elems = interface[hp[InDom]]
            weights = W_DMP[hp[InDom] - nbe]

            val1 = termo0[InDom]*((1 - xy[1])*weights[:,0,0,0] + xy[1]*weights[:,0,1,0])
            val2 = termo0[InDom]*((1 - xy[1])*weights[:,1,0,0] + xy[1]*weights[:,1,1,0])

            self.M_row.extend(2*lef[InDom] + xy[0])
            self.M_col.extend(2*elems[:,0])
            self.M_data.extend(val1)

            self.M_row.extend(2*lef[InDom] + xy[0])
            self.M_col.extend(2*elems[:,1])
            self.M_data.extend(val2)

            self.M_row.extend(2*rel[InDom] + xy[0])
            self.M_col.extend(2*elems[:,0])
            self.M_data.extend(-val1)

            self.M_row.extend(2*rel[InDom] + xy[0])
            self.M_col.extend(2*elems[:,1])
            self.M_data.extend(-val2)

            val1 = termo0[InDom]*((1 - xy[1])*weights[:,0,1,0] + xy[1]*weights[:,0,1,1])
            val2 = termo0[InDom]*((1 - xy[1])*weights[:,1,1,0] + xy[1]*weights[:,1,1,1])

            self.M_row.extend(2*lef[InDom] + xy[0])
            self.M_col.extend(2*elems[:,0] + 1)
            self.M_data.extend(val1)

            self.M_row.extend(2*lef[InDom] + xy[0])
            self.M_col.extend(2*elems[:,1] + 1)
            self.M_data.extend(val2)

            self.M_row.extend(2*rel[InDom] + xy[0])
            self.M_col.extend(2*elems[:,0] + 1)
            self.M_data.extend(-val1)

            self.M_row.extend(2*rel[InDom] + xy[0])
            self.M_col.extend(2*elems[:,1] + 1)
            self.M_data.extend(-val2)

        InNeu = np.isin(hp,IsNeu)
        if InNeu.sum() > 0:

            Mu = U_ij[hp[InNeu],0,0]
            Mv = U_ij[hp[InNeu],0,1]
            F = U_ij[hp[InNeu],0,2]
            
            val = termo0[InNeu]*Mu

            self.M_row.extend(2*lef[InNeu] + xy[0])
            self.M_col.extend(2*col[InNeu])
            self.M_data.extend(val)

            self.M_row.extend(2*rel[InNeu] + xy[0])
            self.M_col.extend(2*col[InNeu])
            self.M_data.extend(- val)

            val = termo0[InNeu]*Mv

            self.M_row.extend(2*lef[InNeu] + xy[0])
            self.M_col.extend(2*col[InNeu] + 1)
            self.M_data.extend(val)

            self.M_row.extend(2*rel[InNeu] + xy[0])
            self.M_col.extend(2*col[InNeu] + 1)
            self.M_data.extend(- val)

            val = termo0[InNeu]*F

            self.I_row.extend(2*lef[InNeu] + xy[0])
            self.I_data.extend(- val)

            self.I_row.extend(2*rel[InNeu] + xy[0])
            self.I_data.extend(val)

            aux = U_ij[hp[InNeu],2,0].astype(int)
            coeff = U_ij[hp[InNeu],1,0]

            aux_InDir = np.isin(aux,IsDir)
            if aux_InDir.sum() > 0:

                val = termo0[InNeu][aux_InDir]*coeff[aux_InDir]*value[aux[aux_InDir]]
                
                self.I_row.extend(2*lef[InNeu][aux_InDir] + xy[0])
                self.I_data.extend(val)

                self.I_row.extend(2*rel[InNeu][aux_InDir] + xy[0])
                self.I_data.extend(- val)

            aux_InDom = aux >= nbe
            if aux_InDom.sum() > 0:

                hp_aux = aux[aux_InDom]
                elems = interface[hp_aux]
                weights = W_DMP[hp_aux - nbe]

                val1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*((1 - xy[1])*weights[:,0,0,0] + xy[1]*weights[:,0,1,0])
                val2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*((1 - xy[1])*weights[:,1,0,0] + xy[1]*weights[:,1,1,0])

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,0])
                self.M_data.extend(-val1)

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,1])
                self.M_data.extend(-val2)

                self.M_row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,0])
                self.M_data.extend(val1)

                self.M_row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,1])
                self.M_data.extend(val2)

                val1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*((1 - xy[1])*weights[:,0,1,0] + xy[1]*weights[:,0,1,1])
                val2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*((1 - xy[1])*weights[:,1,1,0] + xy[1]*weights[:,1,1,1])

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,0] + 1)
                self.M_data.extend(-val1)

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,1] + 1)
                self.M_data.extend(-val2)

                self.M_row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,0] + 1)
                self.M_data.extend(val1)

                self.M_row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,1] + 1)
                self.M_data.extend(val2)
                
            aux = U_ij[hp[InNeu],2,1].astype(int)
            coeff = U_ij[hp[InNeu],1,1]

            aux_InDir = np.isin(aux,IsDir_aux)
            if aux_InDir.sum() > 0:

                val = termo0[InNeu][aux_InDir]*coeff[aux_InDir]*value_aux[aux[aux_InDir]]
                
                self.I_row.extend(2*lef[InNeu][aux_InDir] + xy[0])
                self.I_data.extend(val)

                self.I_row.extend(2*rel[InNeu][aux_InDir] + xy[0])
                self.I_data.extend(- val)

            aux_InDom = aux >= nbe
            if aux_InDom.sum() > 0:

                hp_aux = aux[aux_InDom]
                elems = interface[hp_aux]
                weights = W_DMP[hp_aux - nbe]

                val1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,0,0,0] + (1 - xy[1])*weights[:,0,1,0])
                val2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,1,0,0] + (1 - xy[1])*weights[:,1,1,0])

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,0])
                self.M_data.extend(-val1)

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,1])
                self.M_data.extend(-val2)

                self.M_row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,0])
                self.M_data.extend(val1)

                self.M_row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,1])
                self.M_data.extend(val2)

                val1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,0,1,0] + (1 - xy[1])*weights[:,0,1,1])
                val2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,1,1,0] + (1 - xy[1])*weights[:,1,1,1])

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,0] + 1)
                self.M_data.extend(-val1)

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,1] + 1)
                self.M_data.extend(-val2)  

                self.M_row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,0] + 1)
                self.M_data.extend(val1)

                self.M_row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,1] + 1)
                self.M_data.extend(val2)      

            aux = U_ij[hp[InNeu],2,2].astype(int)
            coeff = U_ij[hp[InNeu],1,2]

            aux_InDir = np.isin(aux,IsDir_aux)
            if aux_InDir.sum() > 0:

                val = termo0[InNeu][aux_InDir]*coeff[aux_InDir]*value_aux[aux[aux_InDir]]
                
                self.I_row.extend(2*lef[InNeu][aux_InDir] + xy[0])
                self.I_data.extend(val)

                self.I_row.extend(2*rel[InNeu][aux_InDir] + xy[0])
                self.I_data.extend(- val)

            aux_InDom = aux >= nbe
            if aux_InDom.sum() > 0:

                hp_aux = aux[aux_InDom]
                elems = interface[hp_aux]
                weights = W_DMP[hp_aux - nbe]

                val1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,0,0,0] + (1 - xy[1])*weights[:,0,1,0])
                val2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,1,0,0] + (1 - xy[1])*weights[:,1,1,0])

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,0])
                self.M_data.extend(-val1)

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,1])
                self.M_data.extend(-val2)

                self.M_row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,0])
                self.M_data.extend(val1)

                self.M_row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,1])
                self.M_data.extend(val2)

                val1 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,0,1,0] + (1 - xy[1])*weights[:,0,1,1])
                val2 = termo0[InNeu][aux_InDom]*coeff[aux_InDom]*(xy[1]*weights[:,1,1,0] + (1 - xy[1])*weights[:,1,1,1])

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,0] + 1)
                self.M_data.extend(-val1)

                self.M_row.extend(2*lef[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,1] + 1)
                self.M_data.extend(-val2)

                self.M_row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,0] + 1)
                self.M_data.extend(val1)

                self.M_row.extend(2*rel[InNeu][aux_InDom] + xy[0])
                self.M_col.extend(2*elems[:,1] + 1)
                self.M_data.extend(val2)    
    
    def matrix_assembly(self,nel):

        I_col = np.zeros(len(self.I_row))

        self.M = coo_matrix((self.M_data,(self.M_row,self.M_col)),shape=(2*nel,2*nel)).tocsr()
       
        self.I = coo_matrix((self.I_data,(self.I_row,I_col)),shape=(2*nel,1)).tocsr()

    def steady_solver(self,sol):

        nel = sol.displacement.field_num.shape[0]

        solution = spsolve(self.M,self.I)

        solution = np.where(np.abs(solution) < 1e-5, 0 , solution)

        sol.displacement.field_num = solution.reshape(nel,2)

    def displ_interp(self,mesh,bc_val,U,misc_par,stress_par):

        nbe = mesh.edges.boundary.shape[0]
        nie = mesh.edges.internal.shape[0]

        interface = misc_par.interface_elems

        W_DMP = stress_par.W_DMP

        u = U[:,0]
        v = U[:,1]    

        IsDirx = bc_val.hdispl.dir_edges
        IsNeux = bc_val.hdispl.neu_edges
        valuex = bc_val.hdispl.bc_value

        IsDiry = bc_val.vdispl.dir_edges
        IsNeuy = bc_val.vdispl.neu_edges
        valuey = bc_val.vdispl.bc_value

        idispl = np.zeros([nie,2])

        idispl[:,0] = W_DMP[:,0,0,0]*u[interface[nbe:,0]] +  W_DMP[:,1,0,0]*u[interface[nbe:,1]] + W_DMP[:,0,0,1]*v[interface[nbe:,0]] +  W_DMP[:,1,0,1]*v[interface[nbe:,1]]
        
        idispl[:,1] = (W_DMP[:,0,1,0]*u[interface[nbe:,0]] +  W_DMP[:,1,1,0]*u[interface[nbe:,1]] 
        + W_DMP[:,0,1,1]*v[interface[nbe:,0]] +  W_DMP[:,1,1,1]*v[interface[nbe:,1]])

        bdispl = np.zeros([nbe,2])

        bdispl[IsDirx,0] = valuex[IsDirx]
        bdispl[IsDiry,1] = valuey[IsDiry]

        if IsNeux.shape[0] > 0:

            M = self.u_ij[IsNeux,0]
            coeff = self.u_ij[IsNeux,1]
            hp_aux = self.u_ij[IsNeux,2].astype(int)
            hp_value = np.zeros(hp_aux.shape)

            aux_InDir = np.isin(hp_aux[:,0],IsDirx)
            if aux_InDir.sum() > 0:

                hp_value[aux_InDir,0] = valuex[hp_aux[aux_InDir,0]]

            aux_InDom = hp_aux[:,0] >= nbe
            if aux_InDom.sum() > 0:

                hp_value[aux_InDom,0] = idispl[hp_aux[aux_InDom,0] - nbe,0]

            aux_InDir = np.isin(hp_aux[:,1],IsDiry)
            if aux_InDir.sum() > 0:

                hp_value[aux_InDir,1] = valuey[hp_aux[aux_InDir,1]]

            aux_InDom = hp_aux[:,1] >= nbe
            if aux_InDom.sum() > 0:

                hp_value[aux_InDom,1] = idispl[hp_aux[aux_InDom,1] - nbe,1]

            aux_InDir = np.isin(hp_aux[:,2],IsDiry)
            if aux_InDir.sum() > 0:

                hp_value[aux_InDir,2] = valuey[hp_aux[aux_InDir,2]]

            aux_InDom = hp_aux[:,2] >= nbe
            if aux_InDom.sum() > 0:

                hp_value[aux_InDom,2] = idispl[hp_aux[aux_InDom,2] - nbe,1]

            bdispl[IsNeux,0] = (M[:,0]*u[interface[IsNeux,0]] + M[:,1]*v[interface[IsNeux,0]]
            - (coeff*hp_value).sum(axis = 1) + M[:,2])

        if IsNeuy.shape[0] > 0:

            M = self.v_ij[IsNeuy,0]
            coeff = self.v_ij[IsNeuy,1]
            hp_aux = self.v_ij[IsNeuy,2].astype(int)
            hp_value = np.zeros(hp_aux.shape)

            aux_InDir = np.isin(hp_aux[:,0],IsDiry)
            if aux_InDir.sum() > 0:

                hp_value[aux_InDir,0] = valuey[hp_aux[aux_InDir,0]]

            aux_InDom = hp_aux[:,0] >= nbe
            if aux_InDom.sum() > 0:

                hp_value[aux_InDom,0] = idispl[hp_aux[aux_InDom,0] - nbe,1]

            aux_InDir = np.isin(hp_aux[:,1],IsDirx)
            if aux_InDir.sum() > 0:

                hp_value[aux_InDir,1] = valuex[hp_aux[aux_InDir,1]]

            aux_InDom = hp_aux[:,1] >= nbe
            if aux_InDom.sum() > 0:

                hp_value[aux_InDom,1] = idispl[hp_aux[aux_InDom,1] - nbe,0]

            aux_InDir = np.isin(hp_aux[:,2],IsDirx)
            if aux_InDir.sum() > 0:

                hp_value[aux_InDir,2] = valuex[hp_aux[aux_InDir,2]]

            aux_InDom = hp_aux[:,2] >= nbe
            if aux_InDom.sum() > 0:

                hp_value[aux_InDom,2] = idispl[hp_aux[aux_InDom,2] - nbe,0]

            bdispl[IsNeuy,1] = (M[:,0]*u[interface[IsNeuy,0]] + M[:,1]*v[interface[IsNeuy,0]]
            - (coeff*hp_value).sum(axis = 1) + M[:,2])

        self.edge_displ = np.zeros([nbe + nie,2])
        self.edge_displ[:nbe] = bdispl
        self.edge_displ[nbe:] = idispl




        
