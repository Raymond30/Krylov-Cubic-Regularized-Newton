import numpy as np
import numpy.linalg as la

import copy
from scipy.optimize import root_scalar
import time

import random
import os
import pickle

# from tqdm.notebook import tqdm
from tqdm import tqdm


from opt_trace import Trace


def set_seed(seed=42):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

class Optimizer:
    """
    Base class for optimization algorithms. Provides methods to run them,
    save the trace and plot the results.
    
    Arguments:
        loss (required): an instance of class Oracle, which will be used to produce gradients,
              loss values, or whatever else is required by the optimizer
        trace_len (int, optional): the number of checkpoints that will be stored in the
                                  trace. Larger value may slowdown the runtime (default: 200)
        use_prox (bool, optional): whether the optimizer should treat the regularizer
                                   using prox (default: True)
        tolerance (float, optional): stationarity level at which the method should interrupt.
                                     Stationarity is computed using the difference between
                                     two consecutive iterates(default: 0)
        line_search (optional): an instance of class LineSearch, which is used to tune stepsize,
                                or other parameters of the optimizer (default: None)
        save_first_iterations (int, optional): how many of the very first iterations should be
                                               saved as checkpoints in the trace. Useful when
                                               optimizer converges fast at the beginning 
                                               (default: 5)
        label (string, optional): label to be passed to the Trace attribute (default: None)
        seeds (list, optional): random seeds to be used to create random number generator (RNG).
                                If None, a single random seed 42 will be used (default: None)
        tqdm (bool, optional): whether to use tqdm to report progress of the run (default: True)
    """
    def __init__(self, loss, trace_len=200, use_prox=True, tolerance=0, line_search=None,
                 save_first_iterations=5, label=None, seeds=None, tqdm=True):
        self.loss = loss
        self.trace_len = trace_len
        self.use_prox = use_prox and (self.loss.regularizer is not None)
        self.tolerance = tolerance
        self.line_search = line_search
        self.save_first_iterations = save_first_iterations
        self.label = label
        self.tqdm = tqdm
        
        self.initialized = False
        self.x_old_tol = None
        self.trace = Trace(loss=loss, label=label)
        if seeds is None:
            self.seeds = [42]
        else:
            self.seeds = seeds
        self.finished_seeds = []
    
    def run(self, x0, t_max=np.inf, it_max=np.inf, ls_it_max=None):
        if t_max is np.inf and it_max is np.inf:
            it_max = 100
            print(f'{self.label}: The number of iterations is set to {it_max}.')
        self.t_max = t_max
        self.it_max = it_max
        
        for seed in self.seeds:
            if seed in self.finished_seeds:
                continue
            if len(self.seeds) > 1:
                print(f'{self.label}: Running seed {seed}')
            self.rng = np.random.default_rng(seed)
            if ls_it_max is None:
                self.ls_it_max = it_max
            if not self.initialized:
                self.init_run(x0)
                self.initialized = True
                
            it_criterion = self.ls_it_max is not np.inf
            tqdm_total = self.ls_it_max if it_criterion else self.t_max
            tqdm_val = 0
            with tqdm(total=tqdm_total) as pbar:
                while not self.check_convergence():
                    if self.tolerance > 0:
                        self.x_old_tol = copy.deepcopy(self.x)
                    self.step()
                    self.save_checkpoint()
                    if it_criterion and self.line_search is not None:
                        tqdm_val_new = self.ls_it
                    elif it_criterion:
                        tqdm_val_new = self.it
                    else:
                        tqdm_val_new = self.t
                    pbar.update(tqdm_val_new - tqdm_val)
                    tqdm_val = tqdm_val_new
            self.finished_seeds.append(seed)
            self.initialized = False

        return self.trace
        
    def check_convergence(self):
        no_it_left = self.it >= self.it_max
        if self.line_search is not None:
            no_it_left = no_it_left or (self.line_search.it >= self.ls_it_max)
        no_time_left = time.perf_counter()-self.t_start >= self.t_max
        if self.tolerance > 0:
            tolerance_met = self.x_old_tol is not None and self.loss.norm(self.x-self.x_old_tol) < self.tolerance
        else:
            tolerance_met = False
        return no_it_left or no_time_left or tolerance_met
        
    def step(self):
        pass
            
    def init_run(self, x0):
        self.dim = x0.shape[0]
        self.x = copy.deepcopy(x0)
        self.trace.xs = [copy.deepcopy(x0)]
        self.trace.its = [0]
        self.trace.ts = [0]
        if self.line_search is not None:
            self.trace.ls_its = [0]
            self.trace.lrs = [self.line_search.lr]
        self.it = 0
        self.t = 0
        self.t_start = time.perf_counter()
        self.time_progress = 0
        self.iterations_progress = 0
        self.max_progress = 0
        if self.line_search is not None:
            self.line_search.reset(self)
        
    def should_update_trace(self):
        if self.it <= self.save_first_iterations:
            return True
        self.time_progress = int((self.trace_len-self.save_first_iterations) * self.t / self.t_max)
        self.iterations_progress = int((self.trace_len-self.save_first_iterations) * (self.it / self.it_max))
        if self.line_search is not None:
            ls_it = self.line_search.it
            self.iterations_progress = max(self.iterations_progress, int((self.trace_len-self.save_first_iterations) * (ls_it / self.it_max)))
        enough_progress = max(self.time_progress, self.iterations_progress) > self.max_progress
        return enough_progress
        
    def save_checkpoint(self):
        self.it += 1
        if self.line_search is not None:
            self.ls_it = self.line_search.it
        self.t = time.perf_counter() - self.t_start
        if self.should_update_trace():
            self.update_trace()
        self.max_progress = max(self.time_progress, self.iterations_progress)
        
    def update_trace(self):
        self.trace.xs.append(copy.deepcopy(self.x))
        self.trace.ts.append(self.t)
        self.trace.its.append(self.it)
        if self.line_search is not None:
            self.trace.ls_its.append(self.line_search.it)
            self.trace.lrs.append(self.line_search.lr)
            
    def compute_loss_of_iterates(self):
        self.trace.compute_loss_of_iterates()
        
    def reset(self, loss):
        self.initialized = False
        self.x_old_tol = None
        self.trace = Trace(loss=loss, label=self.label)
        self.finished_seeds = []
    
    # @classmethod
    def from_pickle(self, path, loss=None):
        if not os.path.isfile(path):
            return None
        with open(path, 'rb') as f:
            trace = pickle.load(f)
            trace.loss = loss
            self.trace = trace
        # if loss is not None:
        #     loss.f_opt = min(self.best_loss_value, loss.f_opt)
        # return trace


def ls_cubic_solver(x, g, H, M, it_max=100, epsilon=1e-8, loss=None):
    """
    Solve min_z <g, z-x> + 1/2<z-x, H(z-x)> + M/3 ||z-x||^3
    
    For explanation of Cauchy point, see "Gradient Descent 
        Efficiently Finds the Cubic-Regularized Non-Convex Newton Step"
        https://arxiv.org/pdf/1612.00547.pdf
    Other potential implementations can be found in paper
        "Adaptive cubic regularisation methods"
        https://people.maths.ox.ac.uk/cartis/papers/ARCpI.pdf
    """
    solver_it = 1
    newton_step = -np.linalg.solve(H, g)
    if M == 0:
        return x + newton_step, solver_it
    def cauchy_point(g, H, M):
        if la.norm(g) == 0 or M == 0:
            return 0 * g
        g_dir = g / la.norm(g)
        H_g_g = H @ g_dir @ g_dir
        R = -H_g_g / (2*M) + np.sqrt((H_g_g/M)**2/4 + la.norm(g)/M)
        return -R * g_dir
    
    def conv_criterion(s, r):
        """
        The convergence criterion is an increasing and concave function in r
        and it is equal to 0 only if r is the solution to the cubic problem
        """
        s_norm = la.norm(s)
        return 1/s_norm - 1/r
    
    # Solution s satisfies ||s|| >= Cauchy_radius
    r_min = la.norm(cauchy_point(g, H, M))
    
    if loss is not None:
        x_new = x + newton_step
        if loss.value(x) > loss.value(x_new):
            return x_new, solver_it
        
    r_max = la.norm(newton_step)
    if r_max - r_min < epsilon:
        return x + newton_step, solver_it
    id_matrix = np.eye(len(g))
    for _ in range(it_max):
        r_try = (r_min + r_max) / 2
        lam = r_try * M
        s_lam = -np.linalg.solve(H + lam*id_matrix, g)
        solver_it += 1
        crit = conv_criterion(s_lam, r_try)
        if np.abs(crit) < epsilon:
            return x + s_lam, solver_it
        if crit < 0:
            r_min = r_try
        else:
            r_max = r_try
        if r_max - r_min < epsilon:
            break
    return x + s_lam, solver_it


def cubic_solver_root(x, g, H, M, V=None, it_max=100, epsilon=1e-8, loss=None, r0 = 0.1):
    id_matrix = np.eye(len(g))

    def func(lam):
        s_lam = -np.linalg.solve(H + lam*id_matrix, g)
        return lam**2 - M**2 * np.linalg.norm(s_lam)**2
    
    def grad(lam):
        s_lam = -np.linalg.solve(H + lam*id_matrix, g)
        phi_lam = np.linalg.norm(s_lam)**2
        phi_lam_grad = -2*np.dot(s_lam,np.linalg.solve(H + lam*id_matrix, s_lam))
        return 2*lam - M**2 * phi_lam_grad

    # s_lam = lambda lam: -np.linalg.solve(H + lam*id_matrix, g)
    # phi_lam = lambda lam: np.la.norm(s_lam(lam))^2
    # phi_lam_grad = lambda lam: -2*s_lam(lam)*np.linalg.solve(H + lam*id_matrix, s_lam(lam))
    # func = lambda r: np.la.norm(np.la.solve(H + M*r*id_matrix, g)) - r

    sol = root_scalar(func, fprime=grad, x0 = r0, method='newton')
    r = sol.root
    s = -np.linalg.solve(H + r*id_matrix, g)
    model_decrease = M/6*np.linalg.norm(s)**3 - np.dot(g,s)/2
    return x + s, sol.iterations, r, None, model_decrease

def Lanczos(A,v,m=10):
    # initialize beta and v
    beta = 0
    v_pre = np.zeros_like(v)
    # normalize v
    v = v / np.linalg.norm(v)
    # Use V to store the Lanczos vectors
    V = np.zeros((len(v),m))
    V[:,0] = v
    # Use alphas, betas to store the Lanczos parameters
    alphas = np.zeros(m)
    betas = np.zeros(m-1)
    for j in range(m-1):
        # w = np.dot(A,v) - beta * v_pre
        w = A(v) - beta * v_pre
        alpha = np.dot(v,w)
        alphas[j] = alpha
        w = w - alpha * v
        beta = np.linalg.norm(w)
        if np.abs(beta) < 1e-6:
            break
        betas[j] = beta
        v_pre = v
        v = w / beta
        V[:,j+1] = v
        
    if m > 1 and j < m-2:
        V = V[:,:j+1]
        alphas = alphas[:j+1]
        betas = betas[:j]
    alphas[-1] = np.dot(v, A(v))
    
    return V, alphas, betas, beta

def cubic_solver_krylov(x, g, H, M, V, it_max=5, epsilon=1e-8, loss=None, r0 = 0.1):
    """
    Solve min_z <g, z-x> + 1/2<z-x, H(z-x)> + M/3 ||z-x||^3
    """
    residual = None
    id_matrix = np.eye(len(g))
    # V, alphas, betas, beta = Lanczos(H, g, m=it_max)
    # T = np.diag(alphas) + np.diag(betas, -1) + np.diag(betas, 1)
    # id_matrix = np.eye(len(alphas))
    # e1 = np.zeros(len(alphas))
    # e1[0] = 1
    # g_tilde = np.linalg.norm(g)*e1
    # g_tilde = V.T @ g
    # print(g_tilde)

    # m_residual = V @ T @ V.T - H
    # eigvals =np.linalg.eigvalsh(m_residual)
    # residual = np.max(eigvals)
    
    def func(lam):
        s_lam = -np.linalg.solve(H + lam*id_matrix, g)
        return lam**2 - M**2 * np.linalg.norm(s_lam)**2
    
    def grad(lam):
        s_lam = -np.linalg.solve(H + lam*id_matrix, g)
        phi_lam = np.linalg.norm(s_lam)**2
        phi_lam_grad = -2*np.dot(s_lam,np.linalg.solve(H + lam*id_matrix, s_lam))
        return 2*lam - M**2 * phi_lam_grad

    # s_lam = lambda lam: -np.linalg.solve(H + lam*id_matrix, g)
    # phi_lam = lambda lam: np.la.norm(s_lam(lam))^2
    # phi_lam_grad = lambda lam: -2*s_lam(lam)*np.linalg.solve(H + lam*id_matrix, s_lam(lam))
    # func = lambda r: np.la.norm(np.la.solve(H + M*r*id_matrix, g)) - r

    sol = root_scalar(func, fprime=grad, x0 = r0, method='newton')
    r = sol.root
    s = -np.linalg.solve(H + r*id_matrix, g)
    model_decrease = M/6*np.linalg.norm(s)**3 - np.dot(g,s)/2
    return x + V @ s, sol.iterations, r, residual, model_decrease

class Cubic(Optimizer):
    """
    Newton method with cubic regularization for global convergence.
    The method was studied by Nesterov and Polyak in the following paper:
        "Cubic regularization of Newton method and its global performance"
        https://link.springer.com/article/10.1007/s10107-006-0706-8
    
    Arguments:
        reg_coef (float, optional): an estimate of the Hessian's Lipschitz constant
    """
    def __init__(self, reg_coef=None, solver_it_max=100, solver_eps=1e-8, cubic_solver="root", *args, **kwargs):
        super(Cubic, self).__init__(*args, **kwargs)
        self.reg_coef = reg_coef
        self.cubic_solver = cubic_solver
        self.solver_it = 0
        self.solver_it_max = solver_it_max
        self.solver_eps = solver_eps

        self.r0 = 0.1
        self.residuals = []
        self.cubic_solver = cubic_solver

        if reg_coef is None:
            self.reg_coef = self.loss.hessian_lipschitz
        if cubic_solver == "GD": 
            self.cubic_solver = ls_cubic_solver
        elif cubic_solver == "root":
            self.cubic_solver = cubic_solver_root
        elif cubic_solver == "krylov":
            self.cubic_solver = cubic_solver_krylov
        else:
            print("Error: cubic_solver not recognized")
        # if cubic_solver is None:
        #     # self.cubic_solver = ls_cubic_solver
        #     self.cubic_solver = cubic_solver_root
        
    def step(self):
        self.grad = self.loss.gradient(self.x)

        if self.cubic_solver is cubic_solver_krylov:
            self.hess = lambda v: self.loss.hess_vec_prod(self.x,v)
        else:
            self.hess = self.loss.hessian(self.x)
        # self.hess = self.loss.hessian(self.x)
        reg_coef = self.reg_coef
        self.x, solver_it, self.r0, residual, model_decrease = self.cubic_solver(self.x, self.grad, self.hess, reg_coef, self.solver_it_max, self.solver_eps, r0 = self.r0)
        self.solver_it += solver_it
        self.residuals.append(residual)
        
    def init_run(self, *args, **kwargs):
        super(Cubic, self).init_run(*args, **kwargs)
        self.trace.solver_its = [0]
        
    def update_trace(self):
        super(Cubic, self).update_trace()
        self.trace.solver_its.append(self.solver_it)


class Cubic_LS(Optimizer):
    """
    Newton method with cubic regularization with line search for global convergence.
    The method was studied by Nesterov and Polyak in the following paper:
        "Cubic regularization of Newton method and its global performance"
        https://link.springer.com/article/10.1007/s10107-006-0706-8
    
    Arguments:
        reg_coef (float, optional): an estimate of the Hessian's Lipschitz constant
    """
    def __init__(self, reg_coef=None, solver_it_max=100, solver_eps=1e-8, cubic_solver="root", beta=0.5, *args, **kwargs):
        super(Cubic_LS, self).__init__(*args, **kwargs)
        self.reg_coef = reg_coef
        self.solver_it = 0
        self.solver_it_max = solver_it_max
        self.solver_eps = solver_eps

        self.beta = beta
        self.r0 = 0.1
        self.residuals = []
        self.value = None

        if reg_coef is None:
            self.reg_coef = self.loss.hessian_lipschitz
        if cubic_solver == "GD": 
            self.cubic_solver = ls_cubic_solver
        elif cubic_solver == "root":
            self.cubic_solver = cubic_solver_root
        elif cubic_solver == "krylov":
            self.cubic_solver = cubic_solver_krylov
        else:
            print("Error: cubic_solver not recognized")
        # if cubic_solver is None:
        #     # self.cubic_solver = ls_cubic_solver
        #     self.cubic_solver = cubic_solver_root
    
    
    def step(self):
        if self.value is None:
            self.value = self.loss.value(self.x)
        
        self.grad = self.loss.gradient(self.x)

        if self.cubic_solver is cubic_solver_krylov:
            self.hess = lambda v: self.loss.hess_vec_prod(self.x,v)
            V, alphas, betas, beta = Lanczos(self.hess, self.grad, m=self.solver_it_max)
            self.hess = np.diag(alphas) + np.diag(betas, -1) + np.diag(betas, 1)
            id_matrix = np.eye(len(alphas))
            e1 = np.zeros(len(alphas))
            e1[0] = 1
            self.grad = np.linalg.norm(self.grad)*e1
        else:
            self.hess = self.loss.hessian(self.x)
            V = None

        if np.linalg.norm(self.grad) < self.tolerance:
            return
        # set the initial value of the regularization coefficient
        reg_coef = self.reg_coef*self.beta

        x_new, solver_it, r0_new, residual, model_decrease = self.cubic_solver(self.x, self.grad, self.hess, 
        reg_coef, V, self.solver_it_max, self.solver_eps, r0 = self.r0)
        value_new = self.loss.value(x_new)
        while value_new > self.value - model_decrease:
            reg_coef = reg_coef/self.beta
            x_new, solver_it, r0_new, residual, model_decrease = self.cubic_solver(self.x, self.grad, self.hess, 
            reg_coef, V, self.solver_it_max, self.solver_eps, r0 = self.r0)
            value_new = self.loss.value(x_new)
        self.x = x_new
        self.reg_coef = reg_coef
        self.value = value_new
        self.r0 = r0_new
        
        self.solver_it += solver_it
        self.residuals.append(residual)
        
    def init_run(self, *args, **kwargs):
        super(Cubic_LS, self).init_run(*args, **kwargs)
        self.trace.solver_its = [0]
        
    def update_trace(self):
        super(Cubic_LS, self).update_trace()
        self.trace.solver_its.append(self.solver_it)