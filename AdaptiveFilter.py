import numpy as np

class EWRLS:

    def __init__(self,lamb,delta,dim,rolling_window = False):
        """
        Description:
        lamb : forgetting factor
        delta : regularisation parameter
        d : dimenion p of the data
        """
        self.lamb = lamb
        self.delta = delta
        self.dim = dim
        self.w = np.zeros((self.dim,1))
        self.P = 1 / delta * np.identity(self.dim)
        self.window = rolling_window
        self.n = 0
        self.U = 0

    def update(self,u,d):
        """
        Description:
        u : observation vector 
        d : target vector
        """
        pi = self.lamb ** (-1) * self.P @ u
        # kalman gain vector
        k = pi / (1 + u.T @ pi)
        # priori error
        xi = d - self.w.T @ u
        # update w
        self.w = self.w + k @ xi.T
        self.P = self.lamb ** (-1) * self.P - self.lamb ** (-1) * k @ u.T @ self.P
        # if self.n == 0:
        #     self.U = xi
        # if self.n > 0:
        #     self.U = np.hstack((self.U,xi))
        self.n += 1

    # # TODO write downdate
    # def downdate(self):
    #     u0 = self.U[:,0]
    #     k = self.P @ u0 / (self.lamb ** (self.window + 1) + u0.T @ self.P @ self.u0)
    #     self.P = self.P - k @ u0.T @ self.P
    #     self.w 

    # def step(self,u,d):
    #     self.update()

    def predict(self,u):
        return (self.w.T @ u).item()


        
        

        
