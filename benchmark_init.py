import numpy as np
import matplotlib.pyplot as plt

# Setting a Benchmark ================================================================
class set_benchmark:

    def __init__(self,case,malha):

        self.case = case
        self.malha = malha
        self.rock = bench_rock(case)
        self.fluid = bench_fluid(case)
        self.bc_val = bc_val(case)
        self.gravity = 9.81 # m/s²

# Rock properties
class bench_rock:

    def __init__(self,case):
        
        group_1 = np.array([11,21,22,23,24,25,26,27]) # homogeneo, sem acoplamento
        group_2 = np.array([12]) # heterogeneo, sem acoplamento
        group_3 = np.array([28]) # propriedades tiradas do exemplo 1 de Demirdzic,Martinovic & Ivankovic (1988)
        group_4 = np.array([29]) # propriedades tiradas do exemplo 3 de Fillipini et al 2008


        if np.isin(case,group_1):

            self.density = np.zeros(1)
            self.compressibility = np.zeros(1)

            self.porosity = np.zeros(1)

            self.young = np.ones(1)
            self.poisson = np.array([0.33])

            self.perm = np.zeros([1,2,2])
            self.perm[:] = np.eye(2) 
        
        elif np.isin(case,group_2):

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
            
        elif np.isin(case,group_3):

            self.density = np.zeros(1)
            self.compressibility = np.zeros(1)

            self.porosity = np.zeros(1)

            self.young = np.array([2.1e11])
            self.poisson = np.array([0.33])

            self.perm = self.perm = np.zeros([1,2,2])
            self.perm[:] = np.eye(2) 

        elif np.isin(case,group_4):

            self.density = np.zeros(1)
            self.compressibility = np.zeros(1)

            self.porosity = np.zeros(1)

            self.young = np.array([3e7])
            self.poisson = np.array([0.3])

            self.perm = self.perm = np.zeros([1,2,2])
            self.perm[:] = np.eye(2)

# Fluid properties
class bench_fluid:

    def __init__(self,case):
        
        
        group_1 = np.array([11,12,21,22,23,24,25,26,27,28,29]) # Monofásico com viscosidade 1

        if np.isin(case,group_1):
            
            self.compressibility = 0
            self.density = 0
            self.viscosity = 1
         
# Boundary conditions
class bc_val:

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
                                    [804,201,0],
                                    [805,201,0]])

            ## Deslocamento 
            #Vertical 

            self.vdispl = np.array([[801,101,0],
                                    [802,201,0],
                                    [803,202,5e3],
                                    [804,201,0],
                                    [805,201,0]])

# Creating a Rock object with property fields ========================================
class set_rock:

    def __init__(self,mesh,benchmark):

        center = mesh.faces.center[:]
        flag_dict = mesh.faces.flag.items()

        nel = center.shape[0]

        flag_array = np.zeros(nel,dtype = int)
        for flag, faces in flag_dict:
            flag_array[faces] = flag - 1
            
        self.density = benchmark.rock.density[flag_array]
        self.compressibility = benchmark.rock.compressibility[flag_array]

        self.porosity = benchmark.rock.porosity[flag_array]

        self.young = benchmark.rock.young[flag_array]
        self.poisson = benchmark.rock.poisson[flag_array]

        self.perm = benchmark.rock.perm[flag_array]

        self.elastic = elastic_tensors(mesh,benchmark,flag_array)

        self.source = np.zeros(2*nel)
        self.source[1::2] = benchmark.gravity*self.density           

class elastic_tensors:

    def __init__(self,mesh,benchmark,flag_array):
        
        center = mesh.faces.center[:]
        nel = center.shape[0]
        young = benchmark.rock.young
        poisson = benchmark.rock.poisson

        lame1 = (young*poisson)/((1 + poisson)*(1 - 2*poisson))
        lame2 = young/(2*(1+poisson))

        self.xx = np.zeros([nel,2,2])
        self.xy = np.zeros([nel,2,2])
        self.yx = np.zeros([nel,2,2])
        self.yy = np.zeros([nel,2,2])
  
        self.xx[:,0,0] = (lame1 + 2*lame2)[flag_array]
        self.xx[:,0,1] = 0
        self.xx[:,1,0] = 0
        self.xx[:,1,1] = lame2[flag_array]

        self.xy[:,0,0] = 0
        self.xy[:,0,1] = lame2[flag_array]
        self.xy[:,1,0] = lame1[flag_array]
        self.xy[:,1,1] = 0

        self.yx[:,0,0] = 0
        self.yx[:,0,1] = lame1[flag_array]
        self.yx[:,1,0] = lame2[flag_array]
        self.yx[:,0,0] = 0

        self.yy[:,0,0] = lame2[flag_array]
        self.yy[:,0,1] = 0
        self.yy[:,1,0] = 0
        self.yy[:,1,1] = (lame1 + 2*lame2)[flag_array]

# Creating a Fluid object with property fields =======================================
class set_fluid:

    def __init__(self,mesh,benchmark):

        ''' Apenas Monofasico por equanto '''

        center = mesh.faces.center[:]
        nel = center.shape[0]

        edges = mesh.edges.center[:]
        nte = edges.shape[0]

        self.density = benchmark.fluid.density*np.ones(nel)
        self.compressibility = benchmark.fluid.compressibility*np.ones(nel)
        self.viscosity = benchmark.fluid.viscosity*np.ones(nel)
        
        self.cell_mobility = 1/self.viscosity
        self.edge_mobility = np.ones(nte)/benchmark.fluid.viscosity

        self.source = np.zeros(nel)

# Creating Wells ======================================================================
class set_well:

    def __init__(self,mesh,strategy):

        center = mesh.faces.center[:]
        nel = center.shape[0]

        bottom_lef = center.min(axis=0)
        upper_right = center.max(axis=0)

        testx = (center[:,0] == bottom_lef[0]).astype(int)
        testy = (center[:,1] == bottom_lef[1]).astype(int)

        elem_1 = np.where((testx + testy) > 1)[0]

        if strategy == 0:
            pass

# Setting bc on edges =================================================================
class set_boundary:

    def __init__(self,mesh,benchmark):

        if benchmark.bc_val.hdispl is None:

            self.pressure = legacy_pressure_bc(mesh,benchmark)

        else:

            bc = bc_sort(mesh,benchmark)

            self.pressure = set_flag_val(bc.pflag,bc.pval)
            self.hdispl = set_flag_val(bc.uflag,bc.uval)
            self.vdispl = set_flag_val(bc.vflag,bc.vval)  

class legacy_pressure_bc:

    def __init__(self,mesh,benchmark):

        nbe = mesh.edges.boundary.shape[0]
        flag_dict = mesh.edges.flag.items()

        bc_val = benchmark.bc_val.pressure

        flag_array = np.zeros(nbe)
        bc_value = np.zeros(nbe)

        for flag, edge in flag_dict:

            flag_array[edge] = flag
            bc_value[edge] = bc_val[bc_val[:,0] == flag,1]

        neu_edges = np.where(flag_array > 200)[0]
        dir_edges = np.where(flag_array < 200)[0]

        self.neu_edges = neu_edges
        self.dir_edges = dir_edges
        self.bc_value = bc_value

class bc_sort:

    def __init__(self,mesh,benchmark):

        nbe = mesh.edges.boundary.shape[0]
        flag_dict = mesh.edges.flag.items()

        p = benchmark.bc_val.pressure
        u = benchmark.bc_val.hdispl
        v = benchmark.bc_val.vdispl

        self.pressure = None
        self.hdispl = None
        self.vdispl = None

        flag_array = np.zeros(nbe)
        
        pflag = np.zeros(nbe)
        pval = np.zeros(nbe)

        uflag = np.zeros(nbe)
        uval = np.zeros(nbe)

        vflag = np.zeros(nbe)
        vval = np.zeros(nbe)

        for flag, edge in flag_dict:

            flag_array[edge] = flag

            pflag[edge] = p[p[:,0] == flag,1]
            pval[edge] = p[p[:,0] == flag,2]

            uflag[edge] = u[u[:,0] == flag,1]
            uval[edge] = u[u[:,0] == flag,2]

            vflag[edge] = v[v[:,0] == flag,1]
            vval[edge] = v[v[:,0] == flag,2]

        # Setting boundary with a function for benchamrks case 21,22...
        edge_center = mesh.edges.center[:nbe]

        unit_square = np.array([21,22,23,24,25,26])
        if np.isin(benchmark.case,unit_square):
            uval = np.where(uval == -1, edge_center[:,0], uval)

        unit_square_vertical = np.array([27])
        if np.isin(benchmark.case,unit_square_vertical):
            vval = np.where(vval == -1, edge_center[:,1], vval)

        demirdzic88 = np.array([28])
        if np.isin(benchmark.case,demirdzic88):
            G = benchmark.rock.young/(2*(1 + benchmark.rock.poisson))
            T = - 50e9
            vval = np.where(vval == -1, (T/G)*edge_center[:,0], vval)

        demirdzic94 = np.array([29])
        if np.isin(benchmark.case,demirdzic94):

            a = 0.5
            b = 2
            fx = 1e5

            nte = mesh.edges.all.shape[0]

            r = edge_center[:,0]**2 + edge_center[:,1]**2
            theta = np.arctan(edge_center[:,1]/edge_center[:,0])

            c1 = a**2/r
            c2 = 1.5*(c1**2)

            sigmaxx = fx*(1-c1*(1.5*np.cos(2*theta)+np.cos(4*theta))+c2*np.cos(4*theta))
            sigmayy = fx*(-c1*(0.5*np.cos(2*theta)-np.cos(4*theta))-c2*np.cos(4*theta))
            sigmaxy = fx*(-c1*(0.5*np.sin(2*theta)+np.sin(4*theta))+c2*np.sin(4*theta))

            normals = mesh.edges.normal[:][mesh.edges.boundary]

            tx = sigmaxx*normals[:,0] + sigmaxy*normals[:,1]
            ty = sigmaxy*normals[:,0] + sigmayy*normals[:,1]

            uval = np.where(uval == -1, tx, uval)
            vval = np.where(vval == -1, ty, vval)

            print('lmao')







        self.pflag = pflag
        self.pval = pval
        
        self.uflag = uflag
        self.uval = uval

        self.vflag = vflag
        self.vval = vval

class set_flag_val:

    def __init__(self,flag_array,bc_value):

        neu_edges = np.where(flag_array > 200)[0]
        dir_edges = np.where(flag_array < 200)[0] 

        self.neu_edges = neu_edges
        self.dir_edges = dir_edges
        self.bc_value = bc_value

# Initializing Solution ===============================================================
class set_solution:

    def __init__(self,mesh,benchmark):

        pressure = pressure_sol(mesh)

        self.pressure = pressure

        displacement = displacement_sol(mesh)

        self.displacement = displacement

    def analytic(self,mesh,benchmark):

        center = mesh.faces.center[:]
        nel = center.shape[0]
        self.pressure.field_exact = np.zeros(nel)
        self.displacement.field_exact = np.zeros([nel,2])

        unit_square = np.array([21,22,23,24,25])
        if np.isin(benchmark.case, unit_square):

            self.displacement.field_exact[:,0] = center[:,0]

        unit_square_vertical = np.array([26,27])
        if np.isin(benchmark.case, unit_square_vertical):

            self.displacement.field_exact[:,1] = center[:,1]


        demirdzic88 = np.array([28])
        if np.isin(benchmark.case,demirdzic88):

            G = benchmark.rock.young/(2*(1 + benchmark.rock.poisson))
            T = -50e9

            self.displacement.field_exact[:,1] = (T/G)*center[:,0]

        demirdzic94 = np.array([29])
        if np.isin(benchmark.case,demirdzic94):

            a = 0.5
            b = 2
            fx = 1e5

            r = center[:,0]**2 + center[:,1]**2
            theta = np.arctan(center[:,1]/center[:,0])

            c1 = a**2/r
            c2 = 1.5*(c1**2)

            sigmaxx = fx*(1-c1*(1.5*np.cos(2*theta)+np.cos(4*theta))+c2*np.cos(4*theta))
            sigmayy = fx*(-c1*(0.5*np.cos(2*theta)-np.cos(4*theta))-c2*np.cos(4*theta))
            sigmaxy = fx*(-c1*(0.5*np.sin(2*theta)+np.sin(4*theta))+c2*np.sin(4*theta))

            self.displacement.sigmaxx = sigmaxx
            self.displacement.sigmaxy = sigmaxy
            self.displacement.sigmayy = sigmayy


    def export(self,mesh,benchmark):

        case = str(benchmark.case)
        malha = benchmark.malha.replace('mesh/','').replace('.msh',' case-') + case
       
        mesh.pressure_cell[:] = self.pressure.field_num[:]
        mesh.displacement_cell[:] = self.displacement.field_num
        """mesh.sigmaxx[:] = self.displacement.sigmaxx
        mesh.sigmaxy[:] = self.displacement.sigmaxy
        mesh.sigmayy[:] = self.displacement.sigmayy"""
        mesh.core.print(file = malha, extension = ".vtk")

    def error(self):

        self.pressure.error = np.abs((self.pressure.field_exact - self.pressure.field_num)/self.pressure.field_exact)
        self.displacement.error = np.abs((self.displacement.field_exact - self.displacement.field_num)/self.displacement.field_exact)

    def plot_line(self, mesh, unknown, axis, coord, exact = None):

        center = mesh.faces.center[:]
        a = np.where(center[:,(1 - axis)] < (coord + coord/10000))[0]
        b = np.where(center[:,(1 - axis)] > (coord - coord/10000))[0]
        elem_list = np.intersect1d(a,b)

        if unknown == 'v':

            sol_num = self.displacement.field_num[:,1]
            sol_exact = self.displacement.field_exact[:,1]
            sol_error = self.displacement.error[:,1]

            plotname = 'Vertical'

        if unknown == 'u':

            sol_num = self.displacement.field_num[:,0]
            sol_exact = self.displacement.field_exact[:,0]
            sol_error = self.displacement.error[:,0]

            plotname = 'Horizontal'

        plt.clf() 
            

        plot_y_num = np.flip(np.sort(sol_num[elem_list]))
        plot_x = np.sort(center[:,axis][elem_list])

        plot_y_exact = np.flip(np.sort(sol_exact[elem_list]))
        plot_y_error = np.flip(np.sort(sol_error[elem_list]))

        plt.plot(plot_x,
                 plot_y_num, 
                 label = plotname + ' Displacemenet(Numeric)',
                 marker = "s")

        plt.plot(plot_x,
                 plot_y_exact, 
                 label = plotname + ' Displacemenet(Analytic)', 
                 color = 'g',
                 marker = 'o')

        plt.title(plotname + ' displacemenet at y = 0.05m')

        plt.xlabel('Legnth(m)')
        plt.ylabel(plotname + ' Displacemenet(m)')

        plt.grid()
        plt.legend()
        plt.savefig(plotname + ' displacemenet at y = 0.05m.png')

        plt.clf() 

        plt.plot(plot_x,
                 plot_y_error,
                 label = plotname +' Displacement(Error)', 
                 color = 'r',
                 marker = 's')

        plt.xlabel('Legnth(m)')
        plt.ylabel(plotname + ' Displacemenet(m)')

        plt.grid()
        plt.legend()
        plt.savefig(plotname +' displacemenet error at y = 0.05m.png')

class pressure_sol:

    def __init__(self,mesh):
        
        center = mesh.faces.center[:]
        nel = center.shape[0]

        # Solução Numérica
        self.field_num = np.zeros([nel,1])

        # Solução analítica pra quando Danilo implementar o Sheng Yuan
        self.field_exact = np.zeros([nel,1])

class displacement_sol:
    
    def __init__(self,mesh):
        
        center = mesh.faces.center[:]
        nel = center.shape[0]

        # Solução Numérica
        self.field_num = np.zeros([nel,2])

        # Solução analítica 
        self.field_exact = np.zeros([nel,2])

        self.sigmaxx = np.zeros([nel,1])
        self.sigmaxy = np.zeros([nel,1])
        self.sigamyy = np.zeros([nel,1])

