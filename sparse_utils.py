from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

""" WIP """
class sparse_utils:

    def __init__(self) -> None:
        pass

    def list_add(discr, row1, col1, data1, flag, row2 = None, col2 = None, data2 = None):

        if flag == 0:

            discr.row.extend(row1)
            discr.col.extend(col1)
            discr.data.extend(data1)

        if flag == 1:

            discr.row.extend(row1)
            discr.col.extend(col1)
            discr.data.extend(data1)

            discr.row.extend(row1)
            discr.col.extend(col2)
            discr.data.extend(data2)

        if flag == 2:

            discr.row.extend(row1)
            discr.col.extend(row1)
            discr.data.extend(data1)

            discr.row.extend(col1)
            discr.col.extend(row1)
            discr.data.extend(-data1)

            discr.row.extend(row1)
            discr.col.extend(col1)
            discr.data.extend(-data2)

            discr.row.extend(col1)
            discr.col.extend(col1)
            discr.data.extend(data2)        

    def add_simple(discr, row, col, data):

        discr.row.extend(row)
        discr.col.extend(col)
        discr.data.extend(data)

    def add_multi(self,discr,case,lef1,rel1,val1,lef2 = None,rel2 = None,val2 = None):

        if case == 0:

            self.add_simple(discr,lef1,lef1,val1)
            self.add_simple(discr,rel1,lef1,-val1)
            self.add_simple(discr,lef1,rel1,-val2)
            self.add_simple(discr,rel1,rel1,val2)

        if case == 1:

            self.add_simple(discr,lef1,lef1,val1)
            self.add_simple(discr,rel1,lef1,val2)

    def mount():
        pass

    def solve():
        pass