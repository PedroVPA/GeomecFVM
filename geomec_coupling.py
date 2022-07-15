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

from functools import partial
import numpy as np
from scipy.sparse import identity
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

class PGHP:

    def __init__(self, mesh, misc_par) -> None:

        normals = misc_par.normal
        esize = misc_par.edge_sizes

        self.partial_x = esize*normals[:,0]
        self.partial_y = esize*normals[:,1]

    def unknow_pressure(self, nbe, interface, W_DMP, partial_x, partial_y):

        # Function for usage in fully implicit algorithms
        # WIP

        lef = interface[nbe:,0]
        rel = interface[nbe:,1]

        self.M_row.extend(2*lef)
        self.M_col.extend(lef)
        self.M_data.extend(W_DMP[:,0]*partial_x[nbe:])

        self.M_row.extend(2*lef)
        self.M_col.extend(rel)
        self.M_data.extend(W_DMP[:,1]*partial_x[nbe:])

        self.M_row.extend(2*rel)
        self.M_col.extend(lef)
        self.M_data.extend(- W_DMP[:,0]*partial_x[nbe:])

        self.M_row.extend(2*rel)
        self.M_col.extend(rel)
        self.M_data.extend(- W_DMP[:,1]*partial_x[nbe:])

        self.M_row.extend(2*lef + 1)
        self.M_col.extend(lef)
        self.M_data.extend(W_DMP[:,0]*partial_y[nbe:])

        self.M_row.extend(2*lef + 1)
        self.M_col.extend(rel)
        self.M_data.extend(W_DMP[:,1]*partial_y[nbe:])

        self.M_row.extend(2*rel + 1)
        self.M_col.extend(lef)
        self.M_data.extend(- W_DMP[:,0]*partial_y[nbe:])

        self.M_row.extend(2*rel + 1)
        self.M_col.extend(rel)
        self.M_data.extend(- W_DMP[:,1]*partial_y[nbe:])

    def matrix_assembly(self,nel):

        self.M = coo_matrix((self.M_data,(self.M_row,self.M_col)),shape=(2*nel,nel)).tocsr()

    def known_pressure(self, nbe, nel, interface, p_ij):

        self.row = []
        self.col = []
        self.data = []

        grad_x = self.partial_x*p_ij
        grad_y = self.partial_y*p_ij

        lef = interface[:nbe,0]

        self.row.extend(2*lef)
        self.data.extend(grad_x[:nbe])

        self.row.extend(2*lef + 1)
        self.data.extend(grad_y[:nbe])

        lef = interface[nbe:,0]
        rel = interface[nbe:,1]

        self.row.extend(2*lef)
        self.data.extend(grad_x[nbe:])

        self.row.extend(2*lef + 1)
        self.data.extend(grad_y[nbe:])

        self.row.extend(2*rel)
        self.data.extend(- grad_x[nbe:])

        self.row.extend(2*rel + 1)
        self.data.extend(- grad_y[nbe:])

        self.vector_assembly(nel)

    def vector_assembly(self,nel):
        
        col = np.zeros(len(self.row))

        self.I = coo_matrix((self.data,(self.row,col)),shape=(2*nel,1)).tocsr()

class rhie_chow:

    def __init__(self) -> None:
        pass

    def interp(self,mesh,misc_par,rc_discr, grad_discr, u_new, u_ij, p_new):

        self.row = []
        self.col = []
        self.data = []

        nbe = mesh.edges.boundary.shape[0]
        nel = mesh.faces.all.shape[0]

        interface = misc_par.interface_elems
        normals = misc_par.normal
        esize = misc_par.edge_sizes

        self.edge_displ(mesh,misc_par,rc_discr, grad_discr, p_new, u_new)

        u_ij_rc = u_ij

        u_ij_rc[nbe:,0] = self.u_ij.squeeze()
        u_ij_rc[nbe:,1] = self.v_ij.squeeze()

        U_ij = esize*(u_ij_rc[:,0]*normals[:,0] + u_ij_rc[:,1]*normals[:,1])

        lef = interface[:nbe,0]

        self.row.extend(lef)
        self.data.extend(U_ij[:nbe])

        lef = interface[nbe:,0]
        rel = interface[nbe:,1]

        self.row.extend(lef)
        self.data.extend(U_ij[nbe:])

        self.row.extend(rel)
        self.data.extend(- U_ij[nbe:])

        self.vector_assembly(nel)

    def edge_displ(self,mesh,misc_par,rc_discr, grad_discr, p_new, u_new):
        
        area = mesh.faces.area[:]
        nbe = mesh.edges.boundary.shape[0]

        interface = misc_par.interface_elems[nbe:]

        grad_p = grad_discr.I.toarray()

        grad_ij_x, grad_ij_y = self.pressure_grad_edge(mesh,p_new,misc_par)

        grad_p_x = grad_p[0::2]
        grad_p_y = grad_p[1::2]

        a = rc_discr.M.diagonal()

        a_L = a[interface[:,0]][:,np.newaxis]
        a_R = a[interface[:,1]][:,np.newaxis]

        u = u_new[:,0,np.newaxis]
        v = u_new[:,1,np.newaxis]

        u_ij_bar = u[interface[:,0]] + u[interface[:,1]]
        v_ij_bar = v[interface[:,0]] + v[interface[:,1]]

        grad_L_x  = grad_p_x[interface[:,0]]/a_L
        grad_L_y  = grad_p_y[interface[:,0]]/a_L

        grad_R_x  = grad_p_x[interface[:,1]]/a_R
        grad_R_y  = grad_p_y[interface[:,1]]/a_R

        coeff_ij = area[interface[:,0]][:,np.newaxis]/a_L + area[interface[:,1][:,np.newaxis]]/a_R

        grad_ij_x *= coeff_ij
        grad_ij_y *= coeff_ij

        self.u_ij = (u_ij_bar + grad_L_x + grad_R_x - grad_ij_x)/2
        self.v_ij = (v_ij_bar + grad_L_y + grad_R_y - grad_ij_y)/2
        
    def pressure_grad_edge(self,mesh,p_new,misc_par) -> np.ndarray:

        nbe = mesh.edges.boundary.shape[0]
        center = mesh.faces.center[:]
        ecenter = mesh.edges.center[:]

        interface = misc_par.interface_elems[nbe:]
        normals = misc_par.normal[nbe:]

        x = center[interface]

        p = p_new[interface]

        r = x[:,1] - x[:,0]
        
        r_proj = r[:,0]*normals[:,0] + r[:,1]*normals[:,1]

        grad_ij = (p[:,1] - p[:,0])/r_proj

        grad_ij_x = grad_ij*normals[:,0]
        grad_ij_y = grad_ij*normals[:,1]

        return grad_ij_x[:,np.newaxis], grad_ij_y[:,np.newaxis]

    def vector_assembly(self,nel):
         
        col = np.zeros(len(self.row))

        self.I = coo_matrix((self.data,(self.row,col)),shape=(nel,1)).tocsr()

class FSSPC:

    def __init__(self,
                 mesh,
                 benchmark,
                 rocks,
                 fluids,
                 sol,
                 bc_val,
                 misc_par,
                 flux_par,
                 stress_par,
                 flux_discr,
                 rc_discr,
                 stress_discr,
                 grad_discr,
                 coupling):

        dt = benchmark.time.step
        total_time = benchmark.time.total
        time_level = 0
        sum_step = 0

        iter_list = []

        area = mesh.faces.area[:]
        nel = area.shape[0]
        nbe = mesh.edges.boundary.shape[0]

        phi = rocks.porosity
        alpha = rocks.biot
        cs = rocks.compressibility

        cf = fluids.compressibility

        interface = misc_par.interface_elems

        Id = identity(nel,format = 'csr')

        Mf = flux_discr.M
        If = flux_discr.I

        Ie = coo_matrix((nel,1)).tocsr()

        Ms = stress_discr.M
        Is = stress_discr.I

        p_old = sol.pressure.field_num
        u_old = sol.displacement.field_num

        error = 1e30
        tol = 1e-16

        while sum_step < total_time:

            simu_percent = sum_step*100/total_time
            
            print(f'Simulation at {simu_percent}%')

            time_level += 1
            sum_step += dt

            s = phi*cf + (phi - alpha)*cs

            coeff = area*s/dt

            Mt = Id.multiply(coeff)
            
            It = (coeff*Id)[:,np.newaxis]*p_old

            Mp = Mt + Mf
            Ip = It + If

            p_old = spsolve(Mp,Ip)

            flux_discr.pressure_interp(mesh,bc_val,p_old,misc_par,flux_par)

            p_ij = flux_discr.edge_pressure

            grad_discr.known_pressure(nbe, nel, interface, p_ij)

            Ig = grad_discr.I

            Mu = Ms
            Iu = Is + Ig

            u_old = (spsolve(Mu,Iu)).reshape([nel,2])

            stress_discr.displ_interp(mesh,bc_val,u_old,misc_par,stress_par)

            u_ij = stress_discr.edge_displ

            iter = 0

            while error > tol:

                iter += 1

                coupling.interp(mesh,misc_par,rc_discr, grad_discr, u_old, u_ij, p_old)

                Ie = coupling.I

                Ip += Ie

                p_new = spsolve(Mp,Ip)

                flux_discr.pressure_interp(mesh,bc_val,p_new,misc_par,flux_par)

                p_ij = flux_discr.edge_pressure

                grad_discr.known_pressure(nbe, nel, interface, p_ij)

                Ig = grad_discr.I

                Mu = Ms
                Iu = Is + Ig

                u_new = (spsolve(Mu,Iu)).reshape([nel,2])

                stress_discr.displ_interp(mesh,bc_val,u_new,misc_par,stress_par)

                u_ij = stress_discr.edge_displ

                p_error = (p_new - p_old).max()
                u_error = (u_new - u_old).max()

                error = max([p_error,u_error])

                p_old = p_new
                u_old = u_new


 

            print('lmao')

class FSSPI:

    def __init__(self) -> None:
        pass