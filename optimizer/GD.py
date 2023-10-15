from optimizer.optimizer import Optimizer
import numpy as np
import time 

class Gd(Optimizer):
    """
    Gradient descent with constant learning rate or a line search procedure.
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr=None, *args, **kwargs):
        super(Gd, self).__init__(*args, **kwargs)
        self.lr = lr
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        if self.line_search is None:
            self.x -= self.lr * self.grad
            if self.use_prox:
                self.x = self.loss.regularizer.prox(self.x, self.lr)
        else:
            self.x = self.line_search(x=self.x, direction=-self.grad)
    
    def init_run(self, *args, **kwargs):
        super(Gd, self).init_run(*args, **kwargs)
        if self.lr is None:
            self.lr = 1 / self.loss.smoothness

class GD_LS(Optimizer):
    """
    Gradient descent with a line search procedure.
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr=None, alpha = 0.5, beta=0.5, *args, **kwargs):
        super(GD_LS, self).__init__(*args, **kwargs)
        self.lr = lr
        self.value = None
        self.beta = beta
        self.alpha = alpha
        
    def step(self):
        if self.value is None:
            self.value = self.loss.value(self.x)
        
        grad_start = time.time()
        self.grad = self.loss.gradient(self.x)
        grad_end = time.time()
        # print('Grad time: {}'.format(grad_end-grad_start))

        lr = self.lr/self.beta
        x_new = self.x - lr * self.grad
        value_new = self.loss.value(x_new)

        grad_squared_norm = np.linalg.norm(self.grad)**2

        # print(lr)
        # print(grad_squared_norm)
        while value_new-self.value >  - self.alpha * lr * grad_squared_norm:
            lr = lr * self.beta
            x_new = self.x - lr * self.grad
            value_new = self.loss.value(x_new)

        self.x = x_new
        self.lr = lr
        self.value = value_new

        # if self.line_search is None:
        #     self.x -= self.lr * self.grad
        #     if self.use_prox:
        #         self.x = self.loss.regularizer.prox(self.x, self.lr)
        # else:
        #     self.x = self.line_search(x=self.x, direction=-self.grad)
    
    def init_run(self, *args, **kwargs):
        super(GD_LS, self).init_run(*args, **kwargs)
        if self.lr is None:
            self.lr = 1 / self.loss.smoothness
