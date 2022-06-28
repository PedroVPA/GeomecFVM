import numpy as np
import scipy.optimize as opt

class Terzaghi(object):
    def __init__(self, z, t, cv, ndata= 100) -> None:
        self.h = z
        self.times = t
        self.cv = cv
        self.p = np.zeros((len(t), ndata))
        self.x = np.linspace(0, z, ndata)
        self.t = np.linspace(0, np.max(t), ndata) * cv /((z)**2)
        self.u = np.zeros_like(self.t)
    

    def pore_pressure(self) -> np.ndarray:
        """Dimensionless pressure profile"""
        oo = 1000
        for i in range(len(self.times)):
            sum_ = np.zeros_like(self.x)
            for j in range(1,oo):
                sum_ += (((-1)**(j-1))/(2*j-1)) * \
                        np.cos((2*j-1)*(np.pi/2)*(self.x/(self.h))) * \
                        np.exp((-(2*j-1)**2)*(np.pi**2/4)*((self.cv*self.times[i]))/((self.h)**2))

            self.p[i] = ((4/np.pi)*sum_)
        return self.p
    
    
    def consolidation(self) -> np.ndarray:
        """Degree of consolidation"""
        oo = 1000
        sum_ = np.zeros_like(self.t)
        for i in range(1,oo + 1):
            sum_ += 1/((2*i-1)**2) * np.exp(-(2*i-1)**2 * (np.pi**2/4) * self.t)
        self.u = 1 - (8/(np.pi**2)) * sum_

        return self.u


class Mandel(object):
    def __init__(self, z, t, F, E, nu, perm, visco, poro, cf, ndata= 100) -> None:
        self.a = z
        self.times = t
        # Terms needed to compute the constants
        K = E / 3.0 / (1 - 2.0 * nu)    # Drained bulk modulus
        G = E / 2.0 / (1 + nu)  # shear modulus 
        M = 1.0 / (poro * cf)   # Biot modulus
        Ku = K + M  # Undrained bulk modulus
        B = M / Ku  # Skempton coefficient
        nu_u = (3.0 * Ku - 2.0 * G) / (6.0 * Ku + 2.0 * G)  # Undrained Poisson's ratio
        self.cv = (2*perm*(B**2)*G*(1-nu)*(1+nu_u)**2)/(9*visco*(1-nu_u)*(nu_u-nu)) # [m^2/s] Fluid diffusivity
        
        # Solutions to tan(x) - ((1-nu)/(nu_u-nu)) x = 0
        """
        This is somehow tricky, we have to solve the equation numerically in order to
        find all the positive solutions to the equation. Later we will use them to 
        compute the infinite sums. Experience has shown that 200 roots are more than enough to
        achieve accurate results. Note that we find the roots using the bisection method.
        """
        f      = lambda x: np.tan(x) - ((1-nu)/(nu_u-nu))*x # define the algebraic eq. as a lambda function
        n_series = 200           # number of roots
        self.a_n = np.zeros(n_series) # initializing roots array
        x0 = 0                   # initial point
        for i in range(0,len(self.a_n)):
            self.a_n[i] = opt.bisect( f,                           # function
                                x0+np.pi/4,                  # left point 
                                x0+np.pi/2-10000*2.2204e-16, # right point (a tiny bit less than pi/2)
                                xtol=1e-30,                  # absolute tolerance
                                rtol=1e-15                   # relative tolerance
                            )        
            x0 += np.pi # apply a phase change of pi to get the next root

        # Terms needed to compute the solutions (these are constants)
        self.p0 = F * B * (1.0 + nu_u) / 3.0 / self.a 
        self.ux0_1 = ((F*nu)/(2*G*self.a))
        self.ux0_2 = -((F*nu_u)/(G*self.a))
        self.ux0_3 = F/G
        self.uy0_1 = (-F*(1-nu))/(2*G*self.a)
        self.uy0_2 = (F*(1-nu_u)/(G*self.a))
        self.sigma0_1 = -F/self.a
        self.sigma0_2 = (-2*F*B*(nu_u-nu)) / (self.a*(1-nu))
        self.sigma0_3 = (2*F)/self.a

        # solution
        self.x = np.linspace(0, z, ndata)
        self.t = np.linspace(0, np.max(self.times), ndata) * self.cv /((z)**2)
        self.p = np.zeros((len(self.times), ndata))
        self.ux = np.zeros_like(self.p)
        self.uy = np.zeros_like(self.p)
        self.syy = np.zeros_like(self.p)
        
        
    def pore_pressure(self) -> np.ndarray:
        # Storing solutions for the subsequent time steps
        self.p[0] = self.p0 * np.ones((len(self.x),))

        for ii in range(1,len(self.times)):
        # Analytical Pressures
            p_sum = 0 
            for n in range(len(self.a_n)):   
                p_sum += ( 
                        ((np.sin(self.a_n[n]))/(self.a_n[n] - (np.sin(self.a_n[n]) * np.cos(self.a_n[n])))) * 
                        (np.cos((self.a_n[n]*self.x)/self.a) - np.cos(self.a_n[n])) * 
                        np.exp((-(self.a_n[n]**2) * self.cv * self.times[ii])/(self.a**2))
                )
        
            self.p[ii] = p_sum *  2.0 * self.p0
        return self.p
        
        
    def displacement_x(self) -> np.ndarray:
        # Storing solutions for the subsequent time steps
        for ii in range(1,len(self.times)):
        
            # Analytical horizontal displacements
            ux_sum1 = 0
            ux_sum2 = 0
            for n in range(len(self.a_n)):      
                ux_sum1 += ( 
                        (np.sin(self.a_n[n])*np.cos(self.a_n[n]))/(self.a_n[n] - np.sin(self.a_n[n])*np.cos(self.a_n[n])) * 
                        np.exp((-(self.a_n[n]**2) * self.cv * self.times[ii])/(self.a**2))
                        )    
                ux_sum2 += (
                    (np.cos(self.a_n[n])/(self.a_n[n] - (np.sin(self.a_n[n]) * np.cos(self.a_n[n])))) *
                    np.sin(self.a_n[n] * (self.x/self.a)) * 
                    np.exp((-(self.a_n[n]**2) * self.cv * self.times[ii])/(self.a**2))
                        ) 
            self.ux[ii] =  (self.ux0_1 + self.ux0_2*ux_sum1) * self.x + self.ux0_3 * ux_sum2
        return self.ux
        
    def displacement_y(self) -> np.ndarray:
        # Storing solutions for the subsequent time steps
        for ii in range(1,len(self.times)):        
            # Analytical vertical displacements
            uy_sum = 0
            for n in range(len(self.a_n)):
                uy_sum += (
                    ((np.sin(self.a_n[n]) * np.cos(self.a_n[n]))/(self.a_n[n] - np.sin(self.a_n[n]) * np.cos(self.a_n[n]))) * 
                    np.exp((-(self.a_n[n]**2) * self.cv* self.times[ii])/(self.a**2)) 
                    )
            self.uy[ii] =  (self.uy0_1 + self.uy0_2*uy_sum) * self.x
        return self.uy
        
        
    def stress_yy(self) -> np.ndarray:
        # Storing solutions for the subsequent time steps
        for ii in range(1,len(self.times)):    
            # Analitical vertical stress
            sigma_sum1 = 0
            sigma_sum2 = 0
            for n in range(len(self.a_n)):
                sigma_sum1 += (
                        ((np.sin(self.a_n[n]))/(self.a_n[n] - (np.sin(self.a_n[n]) * np.cos(self.a_n[n])))) * 
                        np.cos(self.a_n[n] * (self.x/self.a)) * 
                        np.exp((-(self.a_n[n]**2) * self.cv * self.times[ii])/(self.a**2)) 
                        )
                sigma_sum2 += (
                        ((np.sin(self.a_n[n])*np.cos(self.a_n[n]))/(self.a_n[n] - np.sin(self.a_n[n])*np.cos(self.a_n[n]))) * 
                        np.exp((-(self.a_n[n]**2) * self.cv * self.times[ii])/(self.a**2))                  
                        )
            self.syy[ii] =  (self.sigma0_1 + self.sigma0_2*sigma_sum1) + (self.sigma0_3 * sigma_sum2)
        return self.syy
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Rock parameters 
    E = 5e9 # Young modulus [Pa]
    nu = 0.30 # Poisson coeff.
    poro = 0.3 # porosity [-]
    k = 1e-14 # permeability [m2]
    # Fluid parameters
    mu = 1e-3 # viscosity [Pa.s]
    cf = 4.5e-10 # fluid compressibility [1/Pa]
    # Problem parameters
    F = 1.908e6 # load [Pa]
    h = 1.0 # sample heigth [m]
    # Porous medium parameters
    alpha = 1.0 # Biot's coefficient [-]
    K = E / (3 * (1 - 2 * nu)) # bulk modulos [Pa]
    G = 0.5 * E / (1 + nu) # shear modulus [Pa]
    S = poro * cf # storativity (sometimes called: storage coefficient) [1/Pa] 
    mv = 1.0 / (K + (4.0 / 3.0) * G) # Confined compressibility of the porous medium [1/Pa]
    cv = k / mu / (S + alpha * alpha * mv) # Consolidation coefficient [m/s]
    times = np.array([0.001, 0.41, 5, 14.])
    
    # compute terzaghi analytical solution
    # pore pressure
    terz = Terzaghi(1.0, times, cv)
    p = terz.pore_pressure()
    z = 1.0

    fig = plt.figure(figsize=(8,8))
    ax1 = plt.subplot(111)
    for i in range(len(times)):
        ax1.plot( p[i], z, '-k', linewidth = 2)
        
    # Put axis labels
    ax1.set_xlabel(r'$p/p_0$',size=16)
    ax1.set_ylabel(r'$y/h$',size=16)
    ax1.grid(True)
    plt.savefig('Terzaghi analítico - Pressão de Poro')

    # degree of consolidation
    t, u = terz.consolidation()
    fig = plt.figure(figsize=(8,8))
    ax2 = plt.subplot(111)
    ax2.semilogx( t, u, '-k', linewidth = 2, label = 'Analytical')
    plt.savefig('Terzaghi analítico - Consolidação')
    