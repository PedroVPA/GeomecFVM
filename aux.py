import numpy as np


            check_ex = np.zeros([sort_elem.shape[0],ecenter.shape[0]])
            check_ey = np.zeros([sort_elem.shape[0],ecenter.shape[0]])

            check_ex[::-1,:] = ecenter[:,0] 
            check_ey[::-1,:] = ecenter[:,1]

            check_c1x = np.zeros([sort_elem.shape[0],ecenter.shape[0]])
            check_c1y = np.zeros([sort_elem.shape[0],ecenter.shape[0]])

            check_c1x[:,::-1] = auxcoord1[:,0,np.newaxis]
            check_c1y[:,::-1] = auxcoord1[:,1,np.newaxis]

            check1 = (check_c1x == check_ex).astype(int)
            check2 = (check_c1y == check_ey).astype(int)

            check = (check1 + check2) > 1
            
            edge_order1[:,0] = np.where(check > 0)[1]










