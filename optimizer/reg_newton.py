import copy
import numpy as np
import warnings

# from optmethods.line_search import RegNewtonLS
from optimizer.optimizer import Optimizer


def empirical_hess_lip(grad, grad_old, hess, x, x_old, loss):
    grad_error = grad - grad_old - hess@(x - x_old)
    r2 = loss.norm(x - x_old)**2
    if r2 > 0:
        return 2 * loss.norm(grad_error) / r2
    return np.finfo(float).eps

class LineSearch():
    """
    A universal Line Search class that allows for finding the best 
    scalar alpha such that x + alpha * delta is a good
    direction for optimization. The goodness of the new point can 
    be measured in many ways: decrease of functional values, 
    smaller gradient norm, Lipschitzness of an operator, etc.
    Arguments:
        lr0 (float, optional): the initial estimate (default: 1.0)
        count_first_it (bool, optional): to count the first iteration as requiring effort.
            This should be False for methods that reuse information, such as objective value, from the previous
            line search iteration. In contrast, most stochastic line searches
            should count the initial iteration too as information can't be reused (default: False)
        count_last_it (bool, optional): to count the last iteration as requiring effort.
            Not true for line searches that can use the produced information, such as matrix-vector
            product, to compute the next gradient or other important quantities. However, even then, 
            it is convenient to set to False to account for gradient computation (default: True)
        it_max (int, optional): maximal number of innert iterations per one call. 
            Prevents the line search from running for too long and from
            running into machine precision issues (default: 50)
        tolerance (float, optional): the allowed amount of condition violation (default: 0)
    """
    
    def __init__(self, lr0=1.0, count_first_it=False, count_last_it=True, it_max=50, tolerance=0):
        self.lr0 = lr0
        self.lr = lr0
        self.count_first_it = count_first_it
        self.count_last_it = count_last_it
        self.it = 0
        self.it_max = it_max
        self.tolerance = tolerance
        
    @property
    def it_per_call(self):
        return self.count_first_it + self.count_last_it
        
    def reset(self, optimizer):
        self.lr = self.lr0
        self.it = 0
        self.optimizer = optimizer
        self.loss = optimizer.loss
        self.use_prox = optimizer.use_prox
        
    def __call__(self, x=None, direction=None, x_new=None):
        pass
    
class RegNewton(Optimizer):
    """
    Regularized Newton algorithm for second-order minimization.
    By default returns the Regularized Newton method from paper
        "Regularized Newton Method with Global O(1/k^2) Convergence"
        https://arxiv.org/abs/2112.02089
    
    Arguments:
        loss (optmethods.loss.Oracle): loss oracle
        identity_coef (float, optional): initial regularization coefficient (default: None)
        hess_lip (float, optional): estimate for the Hessian Lipschitz constant. 
            If not provided, it is estimated or a small value is used (default: None)
        adaptive (bool, optional): use decreasing regularization based on either empirical Hessian-Lipschitz constant
            or a line-search procedure
        line_search (optmethods.LineSearch, optional): a callable line search. If None, line search is intialized
            automatically as an instance of RegNewtonLS (default: None)
        use_line_search (bool, optional): use line search to estimate the Lipschitz constan of the Hessian.
            If adaptive is True, line search will be non-monotonic and regularization may decrease (default: False)
        backtracking (float, optional): backtracking constant for the line search if line_search is None and
            use_line_search is True (default: 0.5)
    """
    def __init__(self, loss, identity_coef=None, hess_lip=None, adaptive=False, line_search=None,
                 use_line_search=False, backtracking=0.5, *args, **kwargs):
        if hess_lip is None:
            hess_lip = loss.hessian_lipschitz
            if loss.hessian_lipschitz is None:
                hess_lip = 1e-5
                warnings.warn(f"No estimate of Hessian-Lipschitzness is given, so a small value {hess_lip} is used as a heuristic.")
        self.hess_lip = hess_lip
        
        self.H = hess_lip / 2
            
        if use_line_search and line_search is None:
            if adaptive:
                line_search = RegNewtonLS(decrease_reg=adaptive, backtracking=backtracking, H0=self.H)
            else:
                # use a more optimistic initial estimate since hess_lip is often too optimistic
                line_search = RegNewtonLS(decrease_reg=adaptive, backtracking=backtracking, H0=self.H / 100)
        super(RegNewton, self).__init__(loss=loss, line_search=line_search, *args, **kwargs)
        
        self.identity_coef = identity_coef
        self.adaptive = adaptive
        self.use_line_search = use_line_search
        
    def step(self):
        self.grad = self.loss.gradient(self.x)
        if self.adaptive and self.hess is not None and not self.use_line_search:
            self.hess_lip /= 2
            empirical_lip = empirical_hess_lip(self.grad, self.grad_old, self.hess, self.x, self.x_old, self.loss)
            self.hess_lip = max(self.hess_lip, empirical_lip)
        self.hess = self.loss.hessian(self.x)
        
        if self.use_line_search:
            self.x = self.line_search(self.x, self.grad, self.hess)
        else:
            if self.adaptive:
                self.H = self.hess_lip / 2
            grad_norm = self.loss.norm(self.grad)
            self.identity_coef = (self.H * grad_norm)**0.5
            self.x_old = copy.deepcopy(self.x)
            self.grad_old = copy.deepcopy(self.grad)
            delta_x = -np.linalg.solve(self.hess + self.identity_coef*np.eye(self.loss.dim), self.grad)
            self.x += delta_x
        
    def init_run(self, *args, **kwargs):
        super(RegNewton, self).init_run(*args, **kwargs)
        self.x_old = None
        self.hess = None
        self.trace.lrs = []
        
    def update_trace(self, *args, **kwargs):
        super(RegNewton, self).update_trace(*args, **kwargs)
        if not self.use_line_search:
            self.trace.lrs.append(1 / self.identity_coef)



class RegNewtonLS(LineSearch):
    """
    This line search estimates the Hessian Lipschitz constant for the Global Regularized Newton.
    See the following paper for the details and convergence proof:
        "Regularized Newton Method with Global O(1/k^2) Convergence"
        https://arxiv.org/abs/2112.02089
    For consistency with other line searches, 'lr' parameter is used to denote the inverse of regularization.
    Arguments:
        decrease_reg (boolean, optional): multiply the previous regularization parameter by 1/backtracking (default: True)
        backtracking (float, optional): constant by which the current regularization is divided (default: 0.5)
    """
    
    def __init__(self, decrease_reg=True, backtracking=0.5, H0=None, *args, **kwargs):
        super(RegNewtonLS, self).__init__(*args, **kwargs)
        self.decrease_reg = decrease_reg
        self.backtracking = backtracking
        self.H0 = H0
        self.H = self.H0
        self.attempts = 0
        
    def condition(self, x_new, x, grad, identity_coef):
        if self.f_prev is None:
            self.f_prev = self.loss.value(x)
        self.f_new = self.loss.value(x_new)
        r = self.loss.norm(x_new - x)
        condition_f = self.f_new <= self.f_prev - 2/3 * identity_coef * r**2
        grad_new = self.loss.gradient(x_new)
        condition_grad = self.loss.norm(grad_new) <= 2 * identity_coef * r
        self.attempts = self.attempts + 1 if not condition_f or not condition_grad else 0
        return condition_f and condition_grad
        
    def __call__(self, x, grad, hess):
        if self.decrease_reg:
            self.H *= self.backtracking
        grad_norm = self.loss.norm(grad)
        identity_coef = np.sqrt(self.H * grad_norm)
        
        x_new = x - np.linalg.solve(hess + identity_coef*np.eye(self.loss.dim), grad)
        condition_met = self.condition(x_new, x, grad, identity_coef)
        self.it += self.it_per_call
        it_extra = 0
        it_max = min(self.it_max, self.optimizer.ls_it_max - self.it)
        while not condition_met and it_extra < it_max:
            self.H /= self.backtracking
            identity_coef = np.sqrt(self.H * grad_norm)
            x_new = x - np.linalg.solve(hess + identity_coef*np.eye(self.loss.dim), grad)
            condition_met = self.condition(x_new, x, grad, identity_coef)
            it_extra += 1
            if self.backtracking / self.H == 0:
                break
        self.f_prev = self.f_new
        self.it += it_extra
        self.lr = 1 / identity_coef
        return x_new

    def reset(self, *args, **kwargs):
        super(RegNewtonLS, self).reset(*args, **kwargs)
        self.f_prev = None



