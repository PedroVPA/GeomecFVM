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
from analytic_solutions import compressible_flow_1D

# Setting a Benchmark ================================================================
class set_benchmark:

    def __init__(self,case,malha) -> None:

        self.case = case
        self.malha = malha
        self.rock = benchmark_rock(case)
        self.fluid = benchmark_fluid(case)
        self.bc_val = benchmark_bc(case)
        self.wells = benchmark_wells(case)
        self.time = benchmark_time(case)
        self.solution = benchmark_solution(case)
        self.gravity = 9.81 # m/s²

# Rock properties
class benchmark_rock:

    def __init__(self,case) -> None:
        
        homogeneo = np.array([11,13,21,22,23,24,25,26]) # homogeneo, sem acoplamento
        tese_darlan = np.array([12]) # heterogeneo, sem acoplamento
        demizdzic88 = np.array([28]) # propriedades tiradas do exemplo 1 de Demirdzic,Martinovic & Ivankovic (1988)
        fillipini08 = np.array([29]) # propriedades tiradas do exemplo 3 de Fillipini et al 2008
        xueli2012 = np.array([14])
        ribeiro2016 = np.array([31])

        if np.isin(case,homogeneo):

            self.density = np.zeros(1)
            self.compressibility = np.zeros(1)

            self.porosity = np.zeros(1)

            self.young = np.ones(1)
            #self.poisson = np.array([0.3])
            self.poisson = np.zeros(1)

            self.perm = np.zeros([1,2,2])
            self.perm[:] = np.eye(2) 
        
        elif np.isin(case,tese_darlan):

            self.density = np.array([0,0])
            self.compressibility = np.array([0,0])

            self.porosity = np.array([0,0])

            self.young = np.array([0,0])
            self.poisson = np.array([0,0])

            self.perm = np.zeros([2,2,2])
            permA = 10*np.eye(2)
            permB = 50*np.eye(2)
            self.perm[0] = permA
            self.perm[1] = permB
            
        elif np.isin(case,demizdzic88):

            self.density = np.zeros(1)
            self.compressibility = np.zeros(1)

            self.porosity = np.zeros(1)

            self.young = np.array([2.1e11])
            self.poisson = np.array([0.33])

            self.perm = self.perm = np.zeros([1,2,2])
            self.perm[:] = np.eye(2) 

        elif np.isin(case,fillipini08):

            self.density = np.zeros(1)
            self.compressibility = np.zeros(1)

            self.porosity = np.zeros(1)

            self.young = np.array([3e7])
            self.poisson = np.array([0.3])

            self.perm = np.zeros([1,2,2])
            self.perm[:] = np.eye(2)

        elif np.isin(case,ribeiro2016):

            self.density = np.zeros(1)
            self.compressibility = np.array([(25/9)*1e-11])

            self.biot = np.array([7/9])

            self.porosity = np.array([0.19])

            self.young = np.array([14,4e9])
            self.poisson = np.array([0.2])

            self.perm = np.zeros([1,2,2])
            self.perm[:] = 1.9e-15*np.eye(2)

        elif np.isin(case,xueli2012):

            self.density = np.zeros(1)
            self.compressibility = np.array([5e-4/6894.76])

            self.porosity = np.array([0.2])

            self.young = np.zeros(1)
            self.poisson = np.zeros(1)

            self.perm = np.zeros([1,2,2])
            self.perm[:] = 500e-15*np.eye(2)

# Fluid properties
class benchmark_fluid:

    def __init__(self,case) -> None:
        
        
        monofasico = np.array([11,12,13,21,22,23,24,25,26,28,29]) # Monofásico com viscosidade 1
        ribeiro2016 = np.array([31])
        xueli2012 = np.array([14])
        bifasico = np.array([])

        if np.isin(case,monofasico):
            
            self.compressibility = 0
            self.density = 0
            self.viscosity = 1

        if np.isin(case,ribeiro2016):
           
            self.compressibility = (300/99)*1e-10
            self.density = 0
            self.viscosity = 0.001

        if np.isin(case,xueli2012):
            
            self.compressibility = 1.04e-5/6894.76
            self.density = 0
            self.viscosity = 0.249e-3

# Boundary conditions
class benchmark_bc:

    ''' 
    Atribui as condições de contorno bem como seus respectivos valores
    
    [801,101,1] -> [ referencia à face, tipo de condição de contorno, valor da condição]  
    
    801,802,803,804 = Face sul, Face lest, Face norte, Face oeste, respectivamente
    Entre 100 e 200 = Condição de Dirichlet
    Acima de 200    = Condição de Neumann    

    Obs: valores diferentes tem flags diferentes
  
    
    '''

    def __init__(self,case) -> None:
        
        self.pressure = None
        self.hdispl = None
        self.vdispl = None

        if case == 11:
            
            self.pressure = np.array([[201,0],[101,1],[102,0]])
                                    
        if case == 12:

            ## Pressão
        
             self.pressure = np.array([[801,101,1],
                                      [802,201,0],
                                      [803,102,0], 
                                      [804,201,0]]) 

            ## Deslocamento 
            #Horizontal 

             self.hdispl = np.array([[801,101,0],
                                     [802,101,0],
                                     [803,101,0],
                                     [804,101,0]])
            ## Deslocamento 
            #Vertical 

             self.vdispl = np.array([[801,101,0],
                                    [802,101,0],
                                    [803,101,0],
                                    [804,101,0]])  

        if case == 13:

            ## Pressão
        
             self.pressure = np.array([[801,201,0],
                                      [802,201,0],
                                      [803,201,0], 
                                      [804,201,0]]) 

            ## Deslocamento 
            #Horizontal 

             self.hdispl = np.array([[801,101,0],
                                     [802,101,0],
                                     [803,101,0],
                                     [804,101,0]])
            ## Deslocamento 
            #Vertical 

             self.vdispl = np.array([[801,101,0],
                                    [802,101,0],
                                    [803,101,0],
                                    [804,101,0]]) 

        if case == 14:

            ## Pressão
        
             self.pressure = np.array([[801,201,0],
                                      [802,101,2000*6894.76],
                                      [803,201,0], 
                                      [804,102,1900*6894.76]]) 

            ## Deslocamento 
            #Horizontal 

             self.hdispl = np.array([[801,101,0],
                                     [802,101,0],
                                     [803,101,0],
                                     [804,101,0]])
            ## Deslocamento 
            #Vertical 

             self.vdispl = np.array([[801,101,0],
                                    [802,101,0],
                                    [803,101,0],
                                    [804,101,0]])  

        if case == 21:

            ## Pressão

            self.pressure = np.array([[801,101,1],
                                      [802,201,0],
                                      [803,102,0],
                                      [804,201,0]])

            ## Deslocamento 
            #Horizontal

            self.hdispl = np.array([[801,101,-1],
                                    [802,102,1],
                                    [803,101,-1],
                                    [804,103,0]])
                    
            ## Deslocamento 
            #Vertical
             
            self.vdispl = np.array([[801,101,0],
                                    [802,101,0],
                                    [803,101,0],
                                    [804,101,0]])

        if case == 22:

            ## Pressão

            self.pressure = np.array([[801,101,1],
                                      [802,201,0],
                                      [803,102,0],
                                      [804,201,0]])

            ## Deslocamento 
            #Horizontal

            self.hdispl = np.array([[801,101,-1],
                                    [802,101,1],
                                    [803,101,-1],
                                    [804,102,0]])
                    
            ## Deslocamento 
            #Vertical
             
            self.vdispl = np.array([[801,101,0],
                                    [802,201,0],
                                    [803,101,0],
                                    [804,101,0]])

        if case == 23:

            ## Pressão

            self.pressure = np.array([[801,101,1],
                                      [802,201,0],
                                      [803,102,0],
                                      [804,201,0]])

            ## Deslocamento 
            #Horizontal

            self.hdispl = np.array([[801,101,-1],
                                    [802,201,1],
                                    [803,101,-1],
                                    [804,102,0]])
                    
            ## Deslocamento 
            #Vertical
             
            self.vdispl = np.array([[801,101,0],
                                    [802,201,0],
                                    [803,101,0],
                                    [804,101,0]])
        
        if case == 24:

            ## Pressão

            self.pressure = np.array([[801,101,1],
                                      [802,201,0],
                                      [803,102,0],
                                      [804,201,0]])

            ## Deslocamento 
            #Horizontal

            self.hdispl = np.array([[801,201,0],
                                    [802,202,1],
                                    [803,201,0],
                                    [804,101,0]])
                    
            ## Deslocamento 
            #Vertical
             
            self.vdispl = np.array([[801,101,0],
                                    [802,101,0],
                                    [803,101,0],
                                    [804,101,0]])

        if case == 25:

            ## Pressão

            self.pressure = np.array([[801,101,1],
                                      [802,201,0],
                                      [803,102,0],
                                      [804,201,0]])

            ## Deslocamento 
            #Horizontal

            self.hdispl = np.array([[801,201,0],
                                    [802,102,1],
                                    [803,201,0],
                                    [804,101,0]])
                    
            ## Deslocamento 
            #Vertical
             
            self.vdispl = np.array([[801,201,0],
                                    [802,201,0],
                                    [803,201,0],
                                    [804,101,0]])

        if case == 26:

            ## Pressão

            self.pressure = np.array([[801,101,1],
                                      [802,201,0],
                                      [803,102,0],
                                      [804,201,0]])

            ## Deslocamento 
            #Horizontal

            self.hdispl = np.array([[801,101,0],
                                    [802,201,0],
                                    [803,201,0],
                                    [804,201,0]])
                    
            ## Deslocamento 
            #Vertical
             
            self.vdispl = np.array([[801,101,0],
                                    [802,201,0],
                                    [803,202,0.5],
                                    [804,201,0]])

        if case == 28:

            ## Pressão
            
            self.pressure = np.array([[801,101,1],
                                      [802,201,0],
                                      [803,102,0],
                                      [804,201,0]])

            ## Deslocamento 
            #Horizontal 

            self.hdispl = np.array([[801,101,0],
                                    [802,101,0],
                                    [803,101,0],
                                    [804,101,0]])

            ## Deslocamento 
            #Vertical 

            self.vdispl = np.array([[801,201,0],
                                    [802,202,-5e10],
                                    [803,201,0],
                                    [804,101, 0]])

        if case == 29:

            ## Pressão
            
            self.pressure = np.array([[801,101,1],
                                      [802,201,0],
                                      [803,102,0],
                                      [804,201,0],
                                      [805,201,0]])

            ## Deslocamento 
            #Horizontal 

            self.hdispl = np.array([[801,201,0],
                                    [802,201,0],
                                    [803,201,0],
                                    [804,101,0],
                                    [805,201,0]])

            ## Deslocamento 
            #Vertical 

            self.vdispl = np.array([[801,101,0],
                                    [802,201,0],
                                    [803,202,5e3],
                                    [804,201,0],
                                    [805,201,0]])

        if case == 31:

            ## Pressão
            
            self.pressure = np.array([[801,201,0],
                                      [802,201,0],
                                      [803,101,0],
                                      [804,201,0]])

            ## Deslocamento 
            #Horizontal 

            self.hdispl = np.array([[801,201,0],
                                    [802,101,0],
                                    [803,201,0],
                                    [804,101,0]])

            ## Deslocamento 
            #Vertical 

            self.vdispl = np.array([[801,101,0],
                                    [802,201,0],
                                    [803,202,1e6],
                                    [804,201,0]])

class benchmark_time:

    def __init__(self,case) -> None:
        

        if case == 31:

            self.total = 400
            self.step = 0.1

        if case == 14:

            self.total = 20*86400
            self.step = 1

class benchmark_wells:

    def __init__(self,case) -> None:

        if case == 13:

            self.empty = False
            self.id = np.array([0,1])
            self.coords = np.array([[0,0],[1,1]])
            self.flag = np.array([1,0])
            self.pressure = np.array([1,0])
            self.rate = np.array([-1,0])

        else:

            self.empty = True

class benchmark_solution:

    def __init__(self,case) -> None:
        
        if case == 14:
        
            self.pressure_init = 2000*6894.76
            self.dipl_init = None

            self.analytic = compressible_flow_1D()
