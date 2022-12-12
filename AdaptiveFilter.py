import numpy as np

class EWRLS:

    def __init__(self,lamb,delta,dim,rolling_window = 180):
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
        # storing all past observations
        self.U = 0
        # storing all past response
        self.D = 0

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
        if self.n == 0:
            self.U = u
            self.D = d
        if self.n > 0:
            self.U = np.hstack((self.U,u))
            self.D = np.hstack((self.D,d))
        self.n += 1

    def downdate(self):
        u0 = np.reshape(self.U[:,0],(5,-1))
        d0 = np.reshape(self.D[:,0],(1,-1))
        k_downdate = self.lamb ** (self.window + 1) * self.P @ u0 / (1 - self.lamb ** (self.window + 1) * u0.T @ self.P @ u0)
        self.w = self.w + k_downdate @ (u0.T @ self.w - d0)
        self.P = self.P + k_downdate @ u0.T @ self.P
        self.n -= 1

    def step(self,u,d):
        self.update(u,d)
        if self.n > self.window:
            self.downdate()


    def predict(self,u):
        return (self.w.T @ u).item()


        
        

        
