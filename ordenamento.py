

elem = mesh.faces.connectivities[:]
center = mesh.faces.center[:]

# Array com vetores que sai do centroide até os vertices, referente a cada vertice
vec = np.zeros([elem.shape[0],4,3]) # shape = [numero de elementos, numero de vertices em cada elemento,numero de coordenadas de cada vetor]

vec[:,0,:] = center 
vec[:,1,:] = center
vec[:,2,:] = center
vec[:,3,:] = center

vec -= coord[elem]

# Acha o ângulo usando o produto escalar
cos = vec[:,:,0]/np.sqrt(vec[:,:,0]**2 + vec[:,:,1]**2)

angle = np.arccos(cos)

# Se o vetor pertencer ao 3ro e 4rto quadrante tem que corrigir o angulo
angle = np.where(vec[:,:,1] < 0, 2*np.pi - angle, angle)

# Ordena os vertices usando a ordenação dos angulos
sort_elem = np.take_along_axis(elem, np.argsort(angle), axis=1) # np.argsort(angle) ordenando os angulos de forma crescente