import numpy as np
from scipy.sparse import identity
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

class pressure_grad:

    def __init__(self, mesh, misc_par) -> None:

        normals = misc_par.normal
        esize = misc_par.edge_sizes

        self.row = []
        self.col = []
        self.data = []

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

    def __init__(self,mesh) -> None:
        pass

    def pressure_grad_edge(self,mesh,sol,misc_par):

        nbe = mesh.edges.boundary.shape[0]
        center = mesh.faces.center[:]
        ecenter = mesh.edges.center[:]

        interface = misc_par.interface_elems[nbe:]
        normals = misc_par.normal[nbe:]

        pressure = sol.pressure.field_num

        x = center[interface]

        p = pressure[interface]

        r = x[:,1] - x[:,0]
        
        r_proj = r[:,0]*normals[:,0] + r[:,1]*normals[:,1]

        grad = (p[:,1] - p[:,0])/r_proj

        grad_x = grad*normals[:,0]
        grad_y = grad*normals[:,1]

        pass

class fixed_strain_single_phase:

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
                 stress_discr,
                 grad_discr):

        dt = benchmark.time.step
        if dt < 0:

                cfl = True

        total_time = benchmark.time.total
        time_level = 0
        sum_step = 0

        area = mesh.faces.area[:]
        nel = area.shape[0]

        phi = rocks.porosity
        alpha = rocks.biot
        cs = rocks.compressibility

        cf = fluids.compressibility

        Mf = flux_discr.M
        If = flux_discr.I

        Ms = stress_discr.M
        Is = stress_discr.I

        Ig = grad_discr.I

        while sum_step < total_time:

            if cfl:

                dt = self.cfl_time()

            m = phi*cf + (phi - alpha)*cs

            coeff = area/(m*dt)

            Isub = identity(nel,format = 'csr')

            Mt = Isub.multiply(area)
            
            It = coeff*Isub

            Mp = Mt + Mf
            Ip = It + If

            while error < tol:

                sol.pressure.field_num = spsolve(Mp,Ip) 

                flux_discr.pressure_interp(mesh,bc_val,sol,misc_par,flux_par)

                grad_discr.known_pressure(nbe, nel, interface, p_ij)

                Mu = Ms
                Iu = Is + Ig

                sol.displ.field_num = spsolve(Mu,Iu) 








    def stability_check():
        pass

    def cfl_check():
        pass

    def cfl_time():
        pass
