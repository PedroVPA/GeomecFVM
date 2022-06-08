import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

class pressure_grad:

    def __init__(self,mesh,misc_par,flux_par):

        nbe = mesh.edges.boundary.shape[0]
        nie = mesh.edges.internal.shape[0]
        nel = mesh.faces.all.shape[0]

        interface = misc_par.interface_elems
        normals = misc_par.normal
        esize = misc_par.edge_sizes

        W_DMP = flux_par.W_DMP

        self.row = []
        self.col = []
        self.data = []

        partial_x = esize*normals[:,0]
        partial_y = esize*normals[:,1]

        lef = interface[:nbe,0]

        self.row.extend(2*lef)
        self.col.extend(lef)
        self.data.extend(partial_x[:nbe])

        self.row.extend(2*lef + 1)
        self.col.extend(lef)
        self.data.extend(partial_y[:nbe])

        lef = interface[nbe:,0]
        rel = interface[nbe:,1]

        self.row.extend(2*lef)
        self.col.extend(lef)
        self.data.extend(W_DMP[:,0]*partial_x[nbe:])

        self.row.extend(2*lef)
        self.col.extend(rel)
        self.data.extend(W_DMP[:,1]*partial_x[nbe:])

        self.row.extend(2*rel)
        self.col.extend(lef)
        self.data.extend(- W_DMP[:,0]*partial_x[nbe:])

        self.row.extend(2*rel)
        self.col.extend(rel)
        self.data.extend(- W_DMP[:,1]*partial_x[nbe:])

        self.row.extend(2*lef + 1)
        self.col.extend(lef)
        self.data.extend(W_DMP[:,0]*partial_y[nbe:])

        self.row.extend(2*lef + 1)
        self.col.extend(rel)
        self.data.extend(W_DMP[:,1]*partial_y[nbe:])

        self.row.extend(2*rel + 1)
        self.col.extend(lef)
        self.data.extend(- W_DMP[:,0]*partial_y[nbe:])

        self.row.extend(2*rel + 1)
        self.col.extend(rel)
        self.data.extend(- W_DMP[:,1]*partial_y[nbe:])


    def matrix_assembly(self,nel):

        self.M = coo_matrix((self.M_data,(self.M_row,self.M_col)),shape=(2*nel,2*nel)).tocsr()
       
class time_diff:

    def __init__(self, mesh, dt, rocks, fluids):

        nel = mesh.faces.all.shape[0]

        self.M_row = []
        self.M_data = []

        self.I_row = []
        self.I_data = []

        pass

class rhie_chow:

    def __init__(self):
        pass

class fixed_strain_single_phase:

    def __init__(self,
                 mesh,
                 benchmark,
                 sol,
                 misc_par,
                 flux_par,
                 stress_par,
                 flux_discr,
                 stress_discr,
                 grad_discr):

        dt = benchmark.time.step
        total_time = benchmark.time.total
        time_level = 0
        sum_step = 0

        nel = mesh.faces.all.shape[0]

        stress_discr.matrix_assembly(nel)

        grad_discr.matrix_assembly(nel)

        flux_discr.matrix_assembly(nel)

        while sum_step < total_time:

            if dt < 0:

                dt = self.cfl_time()

    def stability_check():
        pass

    def cfl_check():
        pass

    def cfl_time():
        pass
