import numpy as np

class Element:
    def __init__(self,ndof):
       self.M = np.zeros((ndof,ndof))
       self.C = np.zeros((ndof, ndof))
       self.K = np.zeros((ndof, ndof))
       self.s = np.zeros(2)
       self.f = np.zeros(2)
       self.x = np.zeros(2)