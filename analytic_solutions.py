
import numpy as np
import matplotlib.pyplot as plt

class compressible_flow_1D:

    def __init__(self) -> None:
        pass

    def pressure(self,xd,td,inf) -> float:

        n = np.arange(1, inf + 1)

        yn = (2*n-1)*np.pi/2

        term1 = np.sin(yn*xd)
        term2 = np.exp(-1*yn*yn*td)

        pd = np.sum(2*term1*term2/yn)

        return pd

if __name__ == "__main__":

    L = 2000 #ft
    cs = 5e-4 #1/psi
    k = 500 #md
    phi = 0.2 
    cf = 1.04e-5 #1/psi
    mu = 0.249 #cp
    pi = 2000 #psi
    pe = 1900 #psi
    inf = 100

    t = 5 #days

    td = 0.157

    sol = compressible_flow_1D()

    xd = np.linspace(0,1)

    pd = np.zeros_like(xd)

    for i in range(xd.shape[0]):

        pd[i] = sol.pressure(xd[i],td,1000)

    p = pe + (pi - pe)*pd

    plt.clf()
    plt.title('Pressure distribution at td = 0.157')
    plt.plot(xd, p, label = 'Analytic', color = 'r', marker = "s")
    plt.xlabel('Dimensionless Distance')
    plt.ylabel('Dimensionless Pressure')
    plt.ylim(1880,2000)
    plt.grid()
    plt.legend()
    plt.savefig('Teste')

    

        





        

