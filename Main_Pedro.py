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
"""

import numpy as np
import time

from preprocessor.meshHandle.finescaleMesh import FineScaleMesh as impress
from benchmark_init import set_benchmark
from global_propreties import set_rock,set_fluid, set_well,set_boundary, set_solution
from nlfv_param import set_nlfv_misc,set_nlfv_flux, set_nlfv_stress
from lfv_hp import MPFAH, MPSAH
from geomec_coupling import RCIN, PGHP, FSSPC

run_start = time.time()
print('Run Time Started!')

#-----------------------------------------------------------------------------
print('\n************************   MESH IMPORT   ************************\n')
#-----------------------------------------------------------------------------
''' 
------------------------- Opções de Malha --------------------------
1. Malhas com apenas fluxo

1.1 Quadrado Unitáro homogêneo

'mesh/homogeneo.msh' -> 2x2 divisões, malha quadrada

1.1 Malhas do paper Contreras, Lyra & Carvalho(2019)

-Sheng & Yuan(2016)/ Malha A/
'mesh/MeshSkewed_02_10x10.msh' -> 10x10 Divisões
'mesh/MeshSkewed_02_20x20.msh' -> 20x20 Divisões
'mesh/MeshSkewed_02_40x40.msh' -> 40x40 Divisões
'mesh/MeshSkewed_02_80x80.msh' -> 80x80 Divisões
'mesh/MeshSkewed_02_160x160.msh' -> 160x160 Divisões

-Sheng & Yuan(2016)/ Malha B/
'mesh/ExampleRogerio_TriangMesh_10x10_Aligned.msh' -> 10x10 Divisões
'mesh/ExampleRogerio_TriangMesh_20x20_Aligned.msh' -> 20x20 Divisões
'mesh/ExampleRogerio_TriangMesh_40x40_Aligned.msh' -> 40x40 Divisões
'mesh/ExampleRogerio_TriangMesh_80x80_Aligned.msh' -> 80x80 Divisões
'mesh/ExampleRogerio_TriangMesh_160x160_Aligned.msh' -> 160x160 Divisões

1.2 Malhas da Tese de Darlan

- quad:
'mesh/TeseDarlan serie 1x2 quad.msh'
'mesh/TeseDarlan serie 2x2 quad.msh'
'mesh/TeseDarlan serie 4x4 quad.msh'
'mesh/TeseDarlan serie 8x8 quad.msh'

- tri:
'mesh/TeseDarlan serie 1x2 right-tri.msh'
'mesh/TeseDarlan serie 2x2 right-tri.msh'
'mesh/TeseDarlan serie 4x4 right-tri.msh'
'mesh/TeseDarlan serie 8x8 right-tri.msh'
'mesh/TeseDarlan serie 16x16 right-tri.msh'

2. Malhas para problemas mecânicos

2.1 'mec NxN elem' - Domínio quadrilateral unitário
N -> Número de divisões em x e y (1, 2, 4, 6, 8, 16, 32, 64, 128)
elem -> Tipo de elemento ('quad' -> quadrado, 'right-ri' -> triangulo retangulo)

- quad:
'mesh/mec 1x2 quad.msh'
'mesh/mec 2x1 quad.msh'
'mesh/mec 2x2 quad.msh'
'mesh/mec 4x4 quad.msh'
'mesh/mec 8x8 quad.msh'
'mesh/mec 16x16 quad.msh'
'mesh/mec 32x32 quad.msh'
'mesh/mec 64x64 quad.msh'
'mesh/mec 128x128 quad.msh'

- tri:
'mesh/mec 1x1 right-tri.msh'
'mesh/mec 2x1 right-tri.msh'
'mesh/mec 2x2 right-tri.msh'
'mesh/mec 128x128 right-tri.msh'

2.2 'demiredzic88 NxN elem' - Exemplo 1 de Demirdzic,Martinovic & Ivankovic (1988)

- quad:
'mesh/demiredzic88 1x2 quad.msh'
'mesh/demiredzic88 1x20 quad.msh' -> Original do paper
'mesh/demiredzic88 1x40 quad.msh'
'mesh/demiredzic88 3x20 quad.msh'
'mesh/demiredzic88 128x128 quad.msh'

- tri:
'mesh/demiredzic88 2x20 right-tri.msh'

3. Malhar para problemas poromecânicos

3.1 Terzaghi homogêneo - Exemplo 1 da tese Ribeiro (2016)

- quad:
'mesh/ribeiro2016 3x12 quad.msh'

------------------------------------------------------------------------
'''
start = time.time()

malha = 'mesh/ribeiro2016 3x12 quad.msh'
mesh  = impress(mesh_file = malha, dim = 2)

print(f"\nMesh generated successfuly! Only took {time.time()- start} seconds!")

#-----------------------------------------------------------------------------
print('\n********************       SETING CASE         *******************\n')
#-----------------------------------------------------------------------------
''' 
------------------------- Opções de Benchmark --------------------------
1. Benchmark com apenas fluxo

    1.1 benchmark homogeneo, permeabilidade = identidade
    1.2 Exemplo 1.1 do item 4.4.4 da Tese de Darlan
    1.3 homogêneo com poços
    1.4 Buckley-Leverett com poços

2. Benchmark com apenas deformação

    2.1 quadrado de lado 1, homogêneo, poisson = 0, totalmente dirichlet
    2.2 quadrado de lado 1, homogêneo, poisson = 0, com Fx = 1 e v = 0 em x = 1, resto dirichlet
    2.3 quadrado de lado 1, homogêneo, poisson = 0, com Fx = 1 e Fy = 0 em x = 1, resto dirichlet
    2.4 quadrado de lado 1, homogêneo, poisson = 0, com Fx = 1 em x = 1, Fx = 0 em y = 1 e y = 0, resto dirichlet
    2.5 quadrado de lado 1, homogêneo, poisson = 0, em x = 0 -> Fy = 0 , x = 1 -> Fy = 0, y = 1 e y = 0 -> Fx = 0 e Fy = 0, resto dirichlet
    2. demiredzic et al 1998, totalmente dirichlet
    2. demiredzic et al 1998, com Fy = 5e9 N e u = 0 em x = 1, resto dirichlet
    2.8 exemplo 1 de demiredzic et al 1998
    2.9 placa com furo sobre carregamento 
------------------------------------------------------------------------
'''
prep_time = time.time()
start = time.time()

# case = 1.1 sem o ponto
case = 31
benchmark = set_benchmark(case,malha)

print(f"Benchmark set successfuly! Only took {time.time()- start} seconds!")

# Propriedades da rocha
start = time.time()

rocks = set_rock(mesh,benchmark)

print(f"Rocks generated successfuly! Only took {time.time()- start} seconds!")

# Propriedades de Fluido
start = time.time()

fluids = set_fluid(mesh,benchmark)

print(f"Fluids generated successfuly! Only took {time.time()- start} seconds!")

# Poços
start = time.time()

wells = set_well(mesh,benchmark)

print(f"Wells generated successfuly! Only took {time.time()- start} seconds!")

# Condições de Contorno
start = time.time()

bc_val = set_boundary(mesh,benchmark)

print(f"Bonudary conditions set successfuly! Only took {time.time()- start} seconds!")

# Inicializando a solução
start = time.time()

sol = set_solution(mesh,benchmark)
sol.analytic(mesh,benchmark)

print(f"Solution initialized successfuly! Only took {time.time()- start} seconds!")

print(f'\nPreprocessor is done! Only took {time.time() - prep_time} seconds')

#-----------------------------------------------------------------------------
print('\n********************   NLFV PREPROCESSOR      *******************\n')
#-----------------------------------------------------------------------------
# Preprocesso dos VF do prof Fernando
start = time.time()

misc_par = set_nlfv_misc(mesh)
flux_par = set_nlfv_flux(mesh,rocks.perm,misc_par)
rc_par = set_nlfv_flux(mesh,rocks.elastic.G,misc_par)
stress_par = set_nlfv_stress(mesh,rocks.elastic,misc_par)


print(f'NLFV parameters generated successfuly! Only took {time.time() - start} seconds!')

#-----------------------------------------------------------------------------
print('\n*********************   CONTINUITY-DISCR    *********************\n')
#-----------------------------------------------------------------------------

# Discretização do termo do fluxo de darcy usando o MPFA-H
start = time.time()

flux_discr = MPFAH(mesh,fluids,wells,bc_val,misc_par,flux_par)

#flux_discr.steady_solver(sol)
#flux_discr.pressure_interp(mesh,bc_val,sol,misc_par,flux_par)

#print(f'Flux discretization done successfuly! Only took {time.time() - start} seconds!')

rc_discr = MPFAH(mesh,fluids,wells,bc_val,misc_par,rc_par)
coupling = RCIN() 

print(f'Continuity Equation discretization done successfuly! Only took {time.time() - start} seconds!')

#-----------------------------------------------------------------------------
print('\n*********************    MOMENTUM-DISCR     *********************\n')
#-----------------------------------------------------------------------------

# Discretização do termo de tensao efetiva usando o MPFA-H
start = time.time()

stress_discr = MPSAH(mesh,rocks,bc_val,misc_par,stress_par)

stress_discr.steady_solver(sol)

stress_discr.displ_interp(mesh,bc_val,sol.displacement.field_num,misc_par,stress_par)

#print(f'Stress discretization done successfuly! Only took {time.time() - start} seconds!')

grad_discr = PGHP(mesh,misc_par)

print(f'Momentum Equation discretization done successfuly! Only took {time.time() - start} seconds!')

#-----------------------------------------------------------------------------
print('\n*********************     FIXED-STRAIN      *********************\n')
#-----------------------------------------------------------------------------

start = time.time()

print('Simulation Start!')

run = FSSPC(mesh,
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
            coupling)

print("fim")
