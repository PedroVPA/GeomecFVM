import numpy as np
from scipy.sparse import identity
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from analytic_solutions import compressible_flow_1D
from matplotlib import pyplot as plt

class implicit_euler:

    def __init__(self, rocks, fluids, key, dim) -> None:

        nel = rocks.compressibility.shape[0]

        k = rocks.perm
        phi = rocks.porosity
        cs = rocks.compressibility

        mu = fluids.viscosity
        cf = fluids.compressibility

        if key == 0:

            ct = cf + cs
            
            s = phi*ct

        Id = identity(nel,format = 'csr')

        self.coeff = Id.multiply(s)

    def update_matrices(self,dt,p_old):

        self.M = self.coeff.multiply(1/dt)

        self.I = self.M*p_old

    def unsteady_solver(self,mesh,benchmark,flux_discr, rocks, fluids, sol):

        L = 609.6
        pi = 2000*6894.76
        pe = 1900*6894.76

        k = rocks.perm[:,0,0]
        phi = rocks.porosity
        cs = rocks.compressibility

        mu = fluids.viscosity
        cf = fluids.compressibility

        ct = cf + cs

        td_coeff = k/(ct*phi*mu*L**2)

        analytic = compressible_flow_1D()

        total_time = benchmark.time.total
        dt = benchmark.time.step

        x = mesh.faces.center[:][:,0]
        xd = x/L

        pd = np.zeros_like(xd)

        sum_step = 0

        p_old = sol.pressure.field_num

        print('Simulation Started')

        while sum_step < total_time:

            sum_step += dt

            print(f't = {sum_step}')
            
            self.update_matrices(dt,p_old)

            M = self.M + flux_discr.M

            I = self.I + flux_discr.I

            p_new = spsolve(M,I)

            p_old = p_new[:,np.newaxis]

            td = (td_coeff*sum_step).max()

            for i in range(xd.shape[0]):

                pd[i] = analytic.pressure(xd[i],td,1000)

            p_exact = pe + (pi - pe)*pd

            error = (p_new - p_exact)/p_exact

            print(f'erro máximo:{np.abs(error).max()}')

            if int(sum_step) in np.array([600,1*86400,5*86400,10*86400,15*86400]).astype(int):

                plt.clf()
                plt.title('Pressure distribution at td = ' + str(td)[0:5])
                plt.plot(xd, p_exact*1e-6, label = 'Analytic', color = 'g', marker = "s")
                plt.plot(xd, p_new*1e-6, label = 'Numeric', color = 'b', marker = "o")
                plt.xlabel('Dimensionless Distance')
                plt.ylabel('Pressure (MPa)')
                plt.ylim(1880*6894.76*1e-6,2000*6894.76*1e-6)
                plt.grid()
                plt.legend()
                plt.savefig('Pressure distribution at td = ' + str(td)[0:5] + '.png')







        


        
        


        

        
        
