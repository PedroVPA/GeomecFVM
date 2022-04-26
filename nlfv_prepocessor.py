import numpy as np  
import time

class set_nlfv:

    def __init__(self,mesh,rock,benchmark):

        start = time.time()

        misc = nlfv_misc()
        misc.interface(mesh)        
        misc.heights(mesh,misc)
        misc.edge_ordering(mesh)

        print(f'Miscelanous computations done! Only took {time.time() - start} seconds!')

        self.misc = misc

        start = time.time()

        pressure = nlfv_pressure(mesh,rock,misc)
        pressure.weights(mesh,misc)
        pressure.harmonic_points(mesh,misc)
        pressure.transmissibility(mesh,misc)

        print(f'Flux parameters computation done! Only took {time.time() - start} seconds!')


        self.pressure = pressure

        if benchmark.bc_val.hdispl is not None:

            start = time.time()

            displ = nlfv_displ(mesh,rock,misc)
            displ.weights(mesh,rock,misc)
            displ.harmonic_points(mesh,misc)
            displ.transmissibility(mesh,misc)

            print(f'Force parameters computation done! Only took {time.time() - start} seconds!')

            self.displ = displ

class nlfv_misc:

    def __init__(self):
        pass

    def interface(self,mesh):

        # Determinig left and right elements
        nodes = mesh.edges.bridge_adjacencies(mesh.edges.internal, "edges", "nodes")
        interface = mesh.edges.bridge_adjacencies(mesh.edges.internal, "edges", "faces")

        centr = mesh.faces.center[:]
        coord = mesh.nodes.coords[:]
        nel = centr.shape[0]

        # 2CV purposes
        if nodes.shape[0] == 2:
            nodes = nodes[np.newaxis]
            interface = interface[np.newaxis]

        vecnode = coord[nodes[:,1],:] - coord[nodes[:,0],:]
        vecelem = centr[interface[:,0]] - coord[nodes[:,0],:]

        # by JCT: more efficient
        vecsign = np.cross(vecnode,vecelem)[:,-1] > 0
        interface = np.where(vecsign[:,np.newaxis] > 0, interface, np.fliplr(interface))

        blef = mesh.edges.bridge_adjacencies(mesh.edges.boundary, "edges", "faces").squeeze()
        brel = np.zeros(blef.shape[0], dtype = int)

        ilef = interface[:,0]
        irel = interface[:,1]

        nbe = blef.shape[0]
        nie = ilef.shape[0]
        nte = nbe + nie

        interface_elem = np.zeros([nte,2],dtype = int)

        interface_elem[:nbe,0] = blef
        interface_elem[:nbe,1] = brel

        interface_elem[nbe:,0] = ilef
        interface_elem[nbe:,1] = irel

        self.interface_elem = interface_elem

    def heights(self,mesh,misc):

        # Computing Heights
        nodes = mesh.edges.bridge_adjacencies(mesh.edges.internal, "edges", "nodes")
        coord = mesh.nodes.coords[:]
        centr = mesh.faces.center[:]

        # 2CV purposes
        if nodes.shape[0] == 2:
            nodes = nodes[np.newaxis]

        nbe = mesh.edges.boundary.shape[0] 
        nie = mesh.edges.internal.shape[0] 
        nel = centr.shape[0]

        interface = misc.interface_elem[nbe:]

        lef = interface[:,0]
        rel = interface[:,1]

        # Geometric propeties
        # vector crossing the edge nodes
        IJ = coord[nodes[:,1],:] - coord[nodes[:,0],:] 

        # vector crossing centroid and one edge node on the right side
        IR = centr[rel,:] - coord[nodes[:,0],:] 

        # vector crossing centroid and one edge node on the lef side
        IL = centr[lef,:] - coord[nodes[:,0],:]

        # Computing the hights (shortest distance from cell center to the edge)
        # IJ and IR, IJ and IL make a paralelogram, in which the area is the
        # cross product, its base is the size of IJ and its high its the high
        # we are trying to find

        # right side
        vecd = np.cross(IJ,IR, axis=1)
        hrel = np.linalg.norm(vecd,axis=1)/np.linalg.norm(IJ,axis=1)

        # left side
        vece = np.cross(IJ,IL, axis=1)
        hlef = np.linalg.norm(vecd,axis=1)/np.linalg.norm(IJ,axis=1)

        h_ij = np.zeros([hlef.shape[0],2])
        h_ij[:,0] = hlef
        h_ij[:,1] = hrel

        self.h_ij = h_ij    

    def conormal(self,mesh,tensor,misc):

        interface = misc.interface_elem

        inedge = mesh.edges.internal
        bedge = mesh.edges.boundary

        nbe = bedge.shape[0]
        nie = inedge.shape[0]
        nte = nbe + nie

        bconnec = mesh.edges.connectivities(bedge)
        iconnec = mesh.edges.bridge_adjacencies(inedge, "edges", "nodes")
        
        connec = np.zeros([nte,2], dtype = int)
        connec[:nbe] = bconnec
        connec[nbe:] = iconnec

        coord = mesh.nodes.coords[:]

        nie = inedge.shape[0]  
        nbe = bedge.shape[0]
        nte = nie + nbe

        Kn = np.zeros([nte,2])
        K_nij = np.zeros([nte,2,2])

        lef = interface[:,0]
        rel = interface[:,1]

        # Pressure harmonic points =====================
        Klef = tensor[lef]
        Krel = tensor[rel]

        IJ = coord[connec[:,1],:] - coord[connec[:,0],:]
        IJ = IJ/np.linalg.norm(IJ, axis = 1)[:,np.newaxis]

        R = np.array([[0, 1],[-1, 0]])
        normals = np.zeros([nte,2])
        normals[:,0] = R[0,0]*IJ[:,0] + R[0,1]*IJ[:,1]
        normals[:,1] = R[1,0]*IJ[:,0] + R[1,1]*IJ[:,1]

        K_nij[:,0,0] = Klef[:,0,0]*normals[:,0] + Klef[:,0,1]*normals[:,1]
        K_nij[:,0,1] = Klef[:,1,0]*normals[:,0] + Klef[:,1,1]*normals[:,1]

        K_nij[:,1,0] = -(Krel[:,0,0]*normals[:,0] + Krel[:,0,1]*normals[:,1])
        K_nij[:,1,1] = -(Krel[:,1,0]*normals[:,0] + Krel[:,1,1]*normals[:,1])

        Kn[:,0] = K_nij[:,0,0]*normals[:,0] + K_nij[:,0,1]*normals[:,1]
        Kn[:,1] = -(K_nij[:,1,0]*normals[:,0] + K_nij[:,1,1]*normals[:,1])
        
        self.co_normal = K_nij
        self.normal_proj = Kn

    def edge_ordering(self,mesh):
        
        elem = mesh.faces.connectivities[:]
        ecenter = mesh.edges.center[:]
        coord = mesh.nodes.coords[:]
        
        
        shape_array = mesh.faces.classify_element[:]
        center = mesh.faces.center[:]

        nel = center.shape[0]

        edge_order = np.zeros([nel,shape_array.max() + 1],dtype=int)

        if shape_array.max() == 3:

            auxcoord1 = (coord[elem[:,0],:] + coord[elem[:,1],:])/2
            auxcoord2 = (coord[elem[:,1],:] + coord[elem[:,2],:])/2 
            auxcoord3 = (coord[elem[:,2],:] + coord[elem[:,3],:])/2 
            auxcoord4 = (coord[elem[:,3],:] + coord[elem[:,0],:])/2 
            
            for ind in range(nel):

                edge_order[ind,0] = np.where((ecenter[:,0] == auxcoord1[ind,0]) & (ecenter[:,1] == auxcoord1[ind,1]))[0]
                edge_order[ind,1] = np.where((ecenter[:,0] == auxcoord2[ind,0]) & (ecenter[:,1] == auxcoord2[ind,1]))[0]
                edge_order[ind,2] = np.where((ecenter[:,0] == auxcoord3[ind,0]) & (ecenter[:,1] == auxcoord3[ind,1]))[0]
                edge_order[ind,3] = np.where((ecenter[:,0] == auxcoord4[ind,0]) & (ecenter[:,1] == auxcoord4[ind,1]))[0]
                
        if shape_array.max() == 2:

            auxcoord1 = (coord[elem[:,0],:] + coord[elem[:,1],:])/2
            auxcoord2 = (coord[elem[:,1],:] + coord[elem[:,2],:])/2 
            auxcoord3 = (coord[elem[:,2],:] + coord[elem[:,0],:])/2 
            
            for ind in range(nel):

                edge_order[ind,0] = np.where((ecenter[:,0] == auxcoord1[ind,0]) & (ecenter[:,1] == auxcoord1[ind,1]))[0]
                edge_order[ind,1] = np.where((ecenter[:,0] == auxcoord2[ind,0]) & (ecenter[:,1] == auxcoord2[ind,1]))[0]
                edge_order[ind,2] = np.where((ecenter[:,0] == auxcoord3[ind,0]) & (ecenter[:,1] == auxcoord3[ind,1]))[0]
        
        self.edge_order = edge_order

class nlfv_pressure:

    def __init__(self,mesh,rock,misc):

        misc.conormal(mesh, rock.perm, misc)
        self.co_normal = misc.co_normal
        self.normal_proj = misc.normal_proj    
    
    # Weights Computation
    def weights(self,mesh,misc):

        nbe = mesh.edges.boundary.shape[0]
        nie = mesh.edges.internal.shape[0]

        h_ij = misc.h_ij

        K_nij = self.co_normal[nbe:]
        Kn = self.normal_proj[nbe:]
        
        c1 = (h_ij[:,1]*Kn[:,0])[:,np.newaxis]
        c2 = (h_ij[:,0]*Kn[:,1])[:,np.newaxis]
        c3 = (h_ij[:,0]*h_ij[:,1])[:,np.newaxis]*(K_nij[:,0] + K_nij[:,1])
        c4 = c1 + c2

        wlef = c1/c4
        wrel = c2/c4

        hp_term = c3/c4

        W_DMP = np.zeros([nie,2])
        W_DMP[:,0] = wlef.squeeze()
        W_DMP[:,1] = wrel.squeeze()

        self.W_DMP = W_DMP
        self.hp_term = hp_term

    # Harmonic Points Computation
    def harmonic_points(self,mesh,misc):

        W_DMP = self.W_DMP
        hp_term = self.hp_term
        
        nbe = mesh.edges.boundary.shape[0]
        nie = mesh.edges.internal.shape[0]

        center = mesh.faces.center[:]
        ecenter = mesh.edges.center[:]

        coord = mesh.nodes.coords[:]
        enodes = mesh.edges.connectivities[:]

        interface = misc.interface_elem[nbe:]

        lef = interface[:,0]
        rel = interface[:,1]

        yp = np.zeros([nbe+nie,3],dtype=float)

        # Boundary harmonic points are the medium points
        yp[:nbe] = ecenter[:nbe]

        # Internal harmonic points
        yp[nbe:,:2] = W_DMP[:,0,np.newaxis]*center[lef,:2] + W_DMP[:,1,np.newaxis]*center[rel,:2] + hp_term

        # Correction for hp that are outside the edge
        vec1 = np.linalg.norm((ecenter[nbe:] - coord[enodes[nbe:,0]]), axis = 1)
        vec2 = np.linalg.norm((ecenter[nbe:] - yp[nbe:]), axis = 1)

        ph_out = vec1 < vec2

        yp[nbe:,0][ph_out] -= hp_term[:,0][ph_out]
        yp[nbe:,1][ph_out] -= hp_term[:,1][ph_out]

        self.hp = yp 
    
    # Transmissibilities Computation
    def transmissibility(self,mesh,misc):

        points = self.hp
        K_nij = self.co_normal
        
        ksi_ph = transmissibilities(mesh,misc,points,K_nij)

        self.trans = ksi_ph.trans
        self.aux_point = ksi_ph.aux_point

        #test = transmissibilitiesv2(mesh,misc,points,K_nij)

class nlfv_displ:

    def __init__(self,mesh,rock,misc):

        nbe = mesh.edges.boundary.shape[0]

        misc.conormal(mesh, rock.elastic.xx, misc)
        Cxx_nij = misc.co_normal
        Cxxn = misc.normal_proj

        misc.conormal(mesh, rock.elastic.xy, misc)
        Cxy_nij = misc.co_normal
        Cxyn = misc.normal_proj

        misc.conormal(mesh, rock.elastic.yx, misc)
        Cyx_nij = misc.co_normal
        Cyxn = misc.normal_proj

        misc.conormal(mesh, rock.elastic.xx, misc)
        Cyy_nij = misc.co_normal
        Cyyn = misc.normal_proj

        self.co_normal = mec_conormal(Cxx_nij,Cxy_nij,Cyx_nij,Cyy_nij)
        self.normal_proj = mec_proj(Cxxn,Cxyn,Cyxn,Cyyn)

    # Weights Computation
    def weights(self,mesh,rock,misc):

        nbe = mesh.edges.boundary.shape[0]
        nie = mesh.edges.internal.shape[0]

        h_ij = misc.h_ij

        P = np.zeros([nie,2,2])
        P[:,0,0] = h_ij[:,1]*self.normal_proj.xx[nbe:,0]
        P[:,0,1] = h_ij[:,1]*self.normal_proj.xy[nbe:,0]
        P[:,1,0] = h_ij[:,1]*self.normal_proj.yx[nbe:,0]
        P[:,1,1] = h_ij[:,1]*self.normal_proj.yy[nbe:,0]

        Q = np.zeros([nie,2,2])
        Q[:,0,0] = h_ij[:,0]*self.normal_proj.xx[nbe:,1]
        Q[:,0,1] = h_ij[:,0]*self.normal_proj.xy[nbe:,1]
        Q[:,1,0] = h_ij[:,0]*self.normal_proj.yx[nbe:,1]
        Q[:,1,1] = h_ij[:,0]*self.normal_proj.yy[nbe:,1]

        M = P + Q

        Minv = np.zeros([nie,2,2])

        den = M[:,0,0]*M[:,1,1] - M[:,0,1]*M[:,1,0]

        Minv[:,0,0] = M[:,1,1]/den
        Minv[:,0,1] = - M[:,0,1]/den
        Minv[:,1,0] = - M[:,1,0]/den
        Minv[:,1,1] = M[:,0,0]/den

        Wlef = np.zeros([nie,2,2])
        Wrel = np.zeros([nie,2,2])

        Wlef[:,0,0] = Minv[:,0,0]*P[:,0,0] + Minv[:,0,1]*P[:,1,0]
        Wlef[:,0,1] = Minv[:,0,0]*P[:,0,1] + Minv[:,0,1]*P[:,1,1]
        Wlef[:,1,0] = Minv[:,1,0]*P[:,0,0] + Minv[:,1,1]*P[:,1,0]
        Wlef[:,1,1] = Minv[:,1,0]*P[:,0,1] + Minv[:,1,1]*P[:,1,1]

        Wrel[:,0,0] = Minv[:,0,0]*Q[:,0,0] + Minv[:,0,1]*Q[:,1,0]
        Wrel[:,0,1] = Minv[:,0,0]*Q[:,0,1] + Minv[:,0,1]*Q[:,1,1]
        Wrel[:,1,0] = Minv[:,1,0]*Q[:,0,0] + Minv[:,1,1]*Q[:,1,0]
        Wrel[:,1,1] = Minv[:,1,0]*Q[:,0,1] + Minv[:,1,1]*Q[:,1,1]

        W_DMP = np.zeros([nie,2,2,2])
        W_DMP[:,0] = Wlef
        W_DMP[:,1] = Wrel

        hp_term = np.zeros([nie,2,4])
        hp_term[:,0,:2] = self.co_normal.xx[nbe:,0] + self.co_normal.xx[nbe:,1]
        hp_term[:,0,2:] = self.co_normal.xy[nbe:,0] + self.co_normal.xy[nbe:,1]
        hp_term[:,1,:2] = self.co_normal.yx[nbe:,0] + self.co_normal.yx[nbe:,1]
        hp_term[:,1,2:] = self.co_normal.yy[nbe:,0] + self.co_normal.yy[nbe:,1]

        hp_term[:,0] *= (h_ij[:,0]*h_ij[:,1]/den)[:,np.newaxis]
        hp_term[:,1] *= (h_ij[:,0]*h_ij[:,1]/den)[:,np.newaxis]

        self.W_DMP = W_DMP
        self.hp_term = hp_term     

    # Harmonic Points Computation
    def harmonic_points(self,mesh,misc):

        W_DMP = self.W_DMP
        hp_term = self.hp_term
        
        nbe = mesh.edges.boundary.shape[0]
        nie = mesh.edges.internal.shape[0]

        center = mesh.faces.center[:]
        ecenter = mesh.edges.center[:]

        coord = mesh.nodes.coords[:]
        enodes = mesh.edges.connectivities[:]

        interface = misc.interface_elem[nbe:]

        lef = interface[:,0]
        rel = interface[:,1]

        yu = np.zeros([nbe+nie,3],dtype=float)
        yv = np.zeros([nbe+nie,3],dtype=float)

        # Boundary harmonic points are the medium points
        yu[:nbe] = ecenter[:nbe]
        yv[:nbe] = ecenter[:nbe]

        # Internal harmonic points
        WL = W_DMP[:,0]
        WR = W_DMP[:,1]

        XL = np.zeros([nie,2,4])
        XR = np.zeros([nie,2,4])

        XL[:,0,:2] = WL[:,0,0,np.newaxis]*center[lef,:2]
        XL[:,0,2:] = WL[:,0,1,np.newaxis]*center[lef,:2]
        XL[:,1,:2] = WL[:,1,0,np.newaxis]*center[lef,:2]
        XL[:,1,2:] = WL[:,1,1,np.newaxis]*center[lef,:2]

        XR[:,0,:2] = WR[:,0,0,np.newaxis]*center[rel,:2]
        XR[:,0,2:] = WR[:,0,1,np.newaxis]*center[rel,:2]
        XR[:,1,:2] = WR[:,1,0,np.newaxis]*center[rel,:2]
        XR[:,1,2:] = WR[:,1,1,np.newaxis]*center[rel,:2]

        YIJ = XL + XR + hp_term

        yu[nbe:,:2] = YIJ[:,0,:2]
        yv[nbe:,:2] = YIJ[:,1,2:]

        self.uhp = yu
        self.vhp = yv
    
    # Transmissibilities Computation
    def transmissibility(self,mesh,misc):

        points = self.uhp
        K_nij = self.co_normal.xx
        
        ksi_ph_xx = transmissibilities(mesh,misc,points,K_nij)

        points = self.uhp
        K_nij = self.co_normal.xy
        
        ksi_ph_xy = transmissibilities(mesh,misc,points,K_nij)

        points = self.vhp
        K_nij = self.co_normal.yx
        
        ksi_ph_yx = transmissibilities(mesh,misc,points,K_nij)

        points = self.vhp
        K_nij = self.co_normal.yy
        
        ksi_ph_yy = transmissibilities(mesh,misc,points,K_nij)

        self.trans = mec_trans(ksi_ph_xx,ksi_ph_xy,ksi_ph_yx,ksi_ph_yy)
        self.aux_point = mec_aux_point(ksi_ph_xx,ksi_ph_xy,ksi_ph_yx,ksi_ph_yy)

class mec_conormal:

    def __init__(self,Cxx_nij,Cxy_nij,Cyx_nij,Cyy_nij):

        self.xx = Cxx_nij
        self.xy = Cxy_nij
        self.yx = Cyx_nij
        self.yy = Cyy_nij

        pass

class mec_proj:

     def __init__(self,Cxxn,Cxyn,Cyxn,Cyyn):

        self.xx = Cxxn
        self.xy = Cxyn
        self.yx = Cyxn
        self.yy = Cyyn

        pass

class mec_trans:

    def __init__(self,ksi_ph_xx,ksi_ph_xy,ksi_ph_yx,ksi_ph_yy):

        self.xx = ksi_ph_xx.trans
        self.xy = ksi_ph_xy.trans
        self.yx = ksi_ph_yx.trans
        self.yy = ksi_ph_yy.trans

class mec_aux_point:

    def __init__(self,ksi_ph_xx,ksi_ph_xy,ksi_ph_yx,ksi_ph_yy):

        self.xx = ksi_ph_xx.aux_point
        self.xy = ksi_ph_xy.aux_point
        self.yx = ksi_ph_yx.aux_point
        self.yy = ksi_ph_yy.aux_point

class transmissibilities:

    def __init__(self, mesh, misc, points, K_nij):

        nbe         = mesh.edges.boundary.shape[0]  
        nie         = mesh.edges.internal.shape[0]
        nte         = nbe + nie

        self.trans = np.zeros([nte,2,2],dtype=float)
        self.aux_point = np.zeros([nte,2,2],dtype=int)
    
        #-------------------------------------------------------------------------            
        # Boundary Edges
        #-------------------------------------------------------------------------
        
        bedge = 0
        row = 0
            
        self.computation(mesh, misc, points, K_nij[:,row],bedge,row)     
        
        #-------------------------------------------------------------------------            
        # Faces interiores: Elementos a esquerda
        #-------------------------------------------------------------------------
        
        bedge = 1
        row = 0       
            
        self.computation(mesh, misc, points, K_nij[:,row], bedge, row)  

        #-------------------------------------------------------------------------            
        # Faces interiores: Elementos a direita
        #-------------------------------------------------------------------------
        
        bedge = 1
        row = 1      
            
        self.computation(mesh, misc, points, K_nij[:,row], bedge, row)  

    def computation(self, mesh, misc, points, K_nij, bedge, row):

        center = mesh.faces.center[:]
        amountedges = mesh.faces.classify_element[:] + 1
        nbe = mesh.edges.boundary.shape[0]

        if bedge == 0:
            connec = mesh.edges.connectivities[:nbe]
            elemen = misc.interface_elem[:][:nbe,row]

        else:
            connec = mesh.edges.connectivities[nbe:]
            elemen = misc.interface_elem[:][nbe:,row]

        # 2CV purposes
        if connec.shape[0] == 2:
            connec = connec[np.newaxis]

        facelement  = misc.edge_order
        y = points

        rz          = np.array([[0], [0], [1]])
        auxindelem  = np.array([1, 2, 3, 4])
        
        for ifacont in range(connec.shape[0]):
            
            lef    = elemen[ifacont] 
            klef   = amountedges[lef]

            # ej(iface) do elemento a esquerda
            ve2 = np.zeros(3)
            ve2[:2] = K_nij[ifacont + bedge*nbe]

            if np.linalg.norm(ve2) < 1e-5:

                # atribuindo valores a os coeficientes
                self.trans[ifacont + bedge*nbe,row,0] = 0
                self.trans[ifacont + bedge*nbe,row,1] = 0
                
                # indexando as faces respetivamente
                self.aux_point[ifacont + bedge*nbe,row,0] = -1
                self.aux_point[ifacont + bedge*nbe,row,1] = -1
            
            else:

                # percorrendo todos as faces dos elemento "lef"
                auxvetor = facelement[lef,:klef]
                auxindex = auxindelem[:klef]
                auxindex[klef-1] = 0
            
                ksii = 1e30
                ksij = 1e30
                        
                #for ind, elem in enumerate(auxvetor):
                    
                ind  = np.arange(len(auxvetor))
                elem1 = auxvetor
                    
                vej = y[auxvetor[auxindex[ind]],:] - center[lef,:]
                vei = y[elem1,:] - center[lef,:]
                
                cosj = vej@ve2/(np.linalg.norm(vej)*np.linalg.norm(ve2))
                cosi = vei@ve2/(np.linalg.norm(vei)*np.linalg.norm(ve2))
                
                # Estes condições evitam que o acos seja numero complexo.
                cosj = np.where(cosj > 1, 1, cosj)
                cosi = np.where(cosi > 1, 1, cosi)
                
                thetalef2 = np.arccos(cosj)
                thetalef1 = np.arccos(cosi)
                
                # Analiza que o K.n pertece ao primeiro quadrante
                vecquadrant1 = np.cross(vei,ve2)
                vecquadrant2 = np.cross(ve2,vej)
                vecquadrant3 = np.cross(vei,vej)
                
                auxquadrant1 = vecquadrant1[:,2]
                auxquadrant2 = vecquadrant2[:,2]
                
                sgnquadrant1 = np.sign(auxquadrant1)
                sgnquadrant2 = np.sign(auxquadrant2)
                
                test11 = sgnquadrant1 == sgnquadrant2        
                test12 = np.abs(auxquadrant1) > 1e-10        
                test13 = np.abs(auxquadrant2) > 1e-10
                
                test21 = sgnquadrant1 == 0
                test22 = sgnquadrant2  > 0
                test23 = sgnquadrant1  > 0
                test24 = sgnquadrant2 == 0
                
                test1  = np.all([test11,np.any([test12,test13],axis=0)],axis=0)
                test2  = np.any([np.all([test21,test22],axis=0),np.all([test23,test24],axis=0)],axis=0)
                        
                test   = np.all([np.any([test1,test2],axis=0),thetalef2 + thetalef1 < np.pi],axis=0)
                
                if np.any(test):
                    
                    itest = np.where(test == True)[0]
                    itest = itest[-1]
                    
                    ksii  = vecquadrant2[itest]@rz/(vecquadrant3[itest]@rz)
                    ksij  = vecquadrant1[itest]@rz/(vecquadrant3[itest]@rz)
                    aux11 = np.int(elem1[itest])
                    aux12 = np.int(auxvetor[auxindex[ind[itest]]])
                        
                if (ksii==1e30 and ksij==1e30) or (ksii>1e20 and ksij>1e20):
                    
                    aux11 = -1
                    aux12 = -1
                    
                    #[ksii,ksij,aux11,aux12,auxy]=aroundfacelement(F,y,lef,ve2,klef,kmap);
                    # atribuindo valores a os coeficientes
                    self.trans[ifacont + bedge*nbe,row,0] = ksii
                    self.trans[ifacont + bedge*nbe,row,1] = ksij
                    
                    # indexando as faces respetivamente
                    self.aux_point[ifacont + bedge*nbe,row,0] = aux11
                    self.aux_point[ifacont + bedge*nbe,row,1] = aux12
                    
                    # verificando a identidade
                    #coefficient[row,4,ifacont+nbe] = np.linalg.norm((auxy[aux11,:] - center[lef,:])*coefficient[row,0,ifacont+nbe] + (auxy[aux12,:] - center[lef,:])*coefficient[row,1,ifacont+nbe]-ve2.T)
                    """
                    if abs(coefficient[row,4,ifacont]) < 1e-10:
                        pass
                    else:    
                        print('Não satisfaz a identidade (7) do artigo Gao and Wu (2013)');
                    """
                else:
                    # atribuindo valores a os coeficientes
                    self.trans[ifacont + bedge*nbe,row,0] = ksii
                    self.trans[ifacont + bedge*nbe,row,1] = ksij
                    
                    # indexando as faces respetivamente
                    self.aux_point[ifacont + bedge*nbe,row,0] = aux11
                    self.aux_point[ifacont + bedge*nbe,row,1] = aux12
                    
                    # verificando a identidade
                    #coefficient[row,4,ifacont+nbe] = np.linalg.norm((y[[aux11],:]-center[[lef],:])*coefficient[row,0,ifacont+nbe] + (y[[aux12],:] - center[[lef],:])*coefficient[row,1,ifacont+nbe]-ve2.T)
                    """
                    if abs(coefficient[row,4,ifacont+nbe]) < 1e-10:
                        pass
                    else:    
                        print('Não satisfaz a identidade (7) do artigo Gao and Wu (2013)')
                    """

class transmissibilitiesv2:

    def __init__(self, mesh, misc, points, K_nij):

        center = mesh.faces.center[:]
        shape_array = mesh.faces.classify_element[:] + 1
        elem_shape = shape_array.max()

        nel = center.shape[0]

        nbe         = mesh.edges.boundary.shape[0]  
        nie         = mesh.edges.internal.shape[0]
        nte         = nbe + nie

        order = misc.edge_order
        interface = misc.interface_elem

        self.trans = np.zeros([nte,2,2],dtype=float)
        self.aux_point = np.zeros([nte,2,2],dtype=int)
    
        y = points[order]

        aux = (np.arange(nel)[:,np.newaxis]*np.ones([nel,elem_shape])).astype(int)
        center_array = center[aux]

        vec_aux = (y - center_array)
        vec_aux_norm = np.linalg.norm(vec_aux, axis = 2)

        # Primeiro, o lado esquerdo
        

        pass



