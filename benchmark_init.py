import numpy as np

# Setting a Benchmark ================================================================
class set_benchmark:

    def __init__(self,case,malha):

        self.case = case
        self.malha = malha
        self.rock = benchmark_rock(case)
        self.fluid = benchmark_fluid(case)
        self.bc_val = benchmark_bc(case)
        self.wells = benchmark_wells(case)
        self.time = benchmark_time(case)
        self.gravity = 9.81 # m/s²

# Rock properties
class benchmark_rock:

    def __init__(self,case):
        
        homogeneo = np.array([11,13,21,22,23,24,25,26,27]) # homogeneo, sem acoplamento
        tese_darlan = np.array([12]) # heterogeneo, sem acoplamento
        demizdzic88 = np.array([28]) # propriedades tiradas do exemplo 1 de Demirdzic,Martinovic & Ivankovic (1988)
        fillipini08 = np.array([29]) # propriedades tiradas do exemplo 3 de Fillipini et al 2008


        if np.isin(case,homogeneo):

            self.density = np.zeros(1)
            self.compressibility = np.zeros(1)

            self.porosity = np.zeros(1)

            self.young = np.ones(1)
            self.poisson = np.array([0.33])

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

            self.perm = self.perm = np.zeros([1,2,2])
            self.perm[:] = np.eye(2)

# Fluid properties
class benchmark_fluid:

    def __init__(self,case):
        
        
        monofasico = np.array([11,12,13,21,22,23,24,25,26,27,28,29]) # Monofásico com viscosidade 1

        if np.isin(case,monofasico):
            
            self.compressibility = 0
            self.density = 0
            self.viscosity = 1
         
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

    def __init__(self,case):
        
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
                                    [804,201,0]])

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
                                    [803,202,1],
                                    [804,201,0]])

        if case == 27:

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

            self.vdispl = np.array([[801,101,-1],
                                    [802,101,-1],
                                    [803,101,-1],
                                    [804,101,-1]])

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
                                    [802,202,-50e9],
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

class benchmark_time:

    def __init__(self,case):
        

        if case == 31:

            self.total = 1
            self.step = 0.1

class benchmark_wells:

    def __init__(self,case):

        if case == 13:

            self.empty = False
            self.id = np.array([0,1])
            self.coords = np.array([[0,0],[1,1]])
            self.flag = np.array([1,0])
            self.pressure = np.array([1,0])
            self.rate = np.array([-1,0])

        else:

            self.empty = True
    