import numpy as np
import numpy.linalg as la
import copy

from scipy.optimize import root_scalar
from scipy.linalg import eigh, solve 

from scipy import sparse
from scipy.sparse.linalg import cg, LinearOperator, spsolve


import random
import os
import time

from optimizer.optimizer import Optimizer





def set_seed(seed=42):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)




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


def cubic_solver_root(g, H, M, it_max=100, epsilon=1e-8, r0 = 0.1):
    if sparse.issparse(H):
        id_matrix = sparse.eye(len(g))
        lp_solve = lambda A,b : spsolve(A,b)
    else:
        id_matrix = np.eye(len(g))
        lp_solve = lambda A,b: solve(A,b, assume_a= 'pos')

    def func(lam):
        s_lam = -lp_solve(H + lam*id_matrix, g)
        return lam**2 - M**2 * np.linalg.norm(s_lam)**2
    
    def grad(lam):
        s_lam = -lp_solve(H + lam*id_matrix, g)
        # phi_lam = la.norm(s_lam)**2
        phi_lam_grad = -2*np.dot(s_lam,lp_solve(H + lam*id_matrix, s_lam))
        return 2*lam - M**2 * phi_lam_grad

    # s_lam = lambda lam: -np.linalg.solve(H + lam*id_matrix, g)
    # phi_lam = lambda lam: np.la.norm(s_lam(lam))^2
    # phi_lam_grad = lambda lam: -2*s_lam(lam)*np.linalg.solve(H + lam*id_matrix, s_lam(lam))
    # func = lambda r: np.la.norm(np.la.solve(H + M*r*id_matrix, g)) - r

    sol = root_scalar(func, fprime=grad, x0 = r0, method='newton', maxiter=it_max, xtol=epsilon)
    r = sol.root
    s = -lp_solve(H + r*id_matrix, g)
    norm_s = la.norm(s)
    model_decrease = r/2*norm_s**2-M/3*norm_s**3 - np.dot(g,s)/2
    return s, sol.iterations, r, model_decrease

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

# def cubic_solver_krylov(x, g, H, M, V, it_max=5, epsilon=1e-8, loss=None, r0 = 0.1):
#     """
#     Solve min_z <g, z-x> + 1/2<z-x, H(z-x)> + M/3 ||z-x||^3
#     """
#     residual = None
#     id_matrix = np.eye(len(g))
#     # V, alphas, betas, beta = Lanczos(H, g, m=it_max)
#     # T = np.diag(alphas) + np.diag(betas, -1) + np.diag(betas, 1)
#     # id_matrix = np.eye(len(alphas))
#     # e1 = np.zeros(len(alphas))
#     # e1[0] = 1
#     # g_tilde = np.linalg.norm(g)*e1
#     # g_tilde = V.T @ g
#     # print(g_tilde)

#     # m_residual = V @ T @ V.T - H
#     # eigvals =np.linalg.eigvalsh(m_residual)
#     # residual = np.max(eigvals)
    
#     def func(lam):
#         s_lam = -np.linalg.solve(H + lam*id_matrix, g)
#         return lam**2 - M**2 * np.linalg.norm(s_lam)**2
    
#     def grad(lam):
#         s_lam = -np.linalg.solve(H + lam*id_matrix, g)
#         phi_lam = np.linalg.norm(s_lam)**2
#         phi_lam_grad = -2*np.dot(s_lam,np.linalg.solve(H + lam*id_matrix, s_lam))
#         return 2*lam - M**2 * phi_lam_grad

#     # s_lam = lambda lam: -np.linalg.solve(H + lam*id_matrix, g)
#     # phi_lam = lambda lam: np.la.norm(s_lam(lam))^2
#     # phi_lam_grad = lambda lam: -2*s_lam(lam)*np.linalg.solve(H + lam*id_matrix, s_lam(lam))
#     # func = lambda r: np.la.norm(np.la.solve(H + M*r*id_matrix, g)) - r

#     sol = root_scalar(func, fprime=grad, x0 = r0, method='newton')
#     r = sol.root
#     s = -np.linalg.solve(H + r*id_matrix, g)
#     norm_s = np.linalg.norm(s)
#     model_decrease = r/2*norm_s**2-M/3*norm_s**3 - np.dot(g,s)/2
#     return x + V @ s, sol.iterations, r, residual, model_decrease

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
        self.loss.reset()
        
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
    def __init__(self, reg_coef=None, cubic_solver="CG", solver_it_max=100, solver_eps=1e-8, beta=0.5, *args, **kwargs):
        super(Cubic_LS, self).__init__(*args, **kwargs)
        self.solver_it = 0
        self.solver_it_max = solver_it_max
        self.solver_eps = solver_eps

        self.beta = beta
        self.r0 = 0.1
        self.residuals = []
        self.value = None

        if reg_coef is None:
            self.reg_coef = self.loss.hessian_lipschitz
        else:
            self.reg_coef = reg_coef
        
        if cubic_solver == "CG":
            self.cubic_solver = self.cubic_solver_root_CG
        elif cubic_solver == "full":
            self.cubic_solver = self.cubic_solver_root_full
        else:
            print("Error: cubic_solver not recognized")
        # if cubic_solver == "GD": 
        #     self.cubic_solver = ls_cubic_solver
        # elif cubic_solver == "root":
        #     self.cubic_solver = cubic_solver_root
        # elif cubic_solver == "krylov":
        #     self.cubic_solver = cubic_solver_krylov
        # else:
        #     print("Error: cubic_solver not recognized")
        # if cubic_solver is None:
        #     # self.cubic_solver = ls_cubic_solver
        #     self.cubic_solver = cubic_solver_root
    
    def cubic_solver_root_CG(self, M, it_max=100, epsilon=1e-8, r0 = 0.1):
        g = self.grad
        def func(lam):
            # s_lam = -solve(H + lam*id_matrix, g, assume_a= 'pos')
            def mv(v):
                return self.loss.hess_vec_prod(self.x,v) + lam*v
            H_lambda = LinearOperator(shape =(len(g),len(g)), matvec=mv)
            s_lam, exit_code = cg(H_lambda, -g, tol=epsilon)
            # s_lam = -s_lam
            return lam**2 - M**2 * np.linalg.norm(s_lam)**2
        
        def grad(lam):
            # s_lam = -solve(H + lam*id_matrix, g, assume_a= 'pos')
            # phi_lam = la.norm(s_lam)**2
            def mv(v):
                return self.loss.hess_vec_prod(self.x,v) + lam*v
            H_lambda = LinearOperator(shape =(len(g),len(g)), matvec=mv)
            s_lam, exit_code = cg(H_lambda, -g, tol=epsilon)
            # s_lam = -s_lam
            # phi_lam_grad = -2*np.dot(s_lam, self.loss.hess_solve(self.x, s_lam, lam))
            Hinv_s_lam, exit_code = cg(H_lambda, s_lam, tol=epsilon)
            phi_lam_grad = -2*np.dot(s_lam, Hinv_s_lam)
            return 2*lam - M**2 * phi_lam_grad

        # s_lam = lambda lam: -np.linalg.solve(H + lam*id_matrix, g)
        # phi_lam = lambda lam: np.la.norm(s_lam(lam))^2
        # phi_lam_grad = lambda lam: -2*s_lam(lam)*np.linalg.solve(H + lam*id_matrix, s_lam(lam))
        # func = lambda r: np.la.norm(np.la.solve(H + M*r*id_matrix, g)) - r

        sol = root_scalar(func, fprime=grad, x0 = r0, method='newton', maxiter=it_max, xtol=epsilon)
        r = sol.root
        # s = -solve(H + r*id_matrix, g, assume_a= 'pos')

        def mv(v):
            return self.loss.hess_vec_prod(self.x,v) + r*v
        H_lambda = LinearOperator(shape =(len(g),len(g)), matvec=mv)
        # s = -self.loss.hess_solve(self.x, g, r)
        s, exit_code = cg(H_lambda, -g, tol=epsilon)
        # s = -s
        norm_s = la.norm(s)
        model_decrease = r/2*norm_s**2-M/3*norm_s**3 - np.dot(g,s)/2
        return s, sol.iterations, r, model_decrease
    
    def cubic_solver_root_full(self, M, it_max=100, epsilon=1e-8, r0 = 0.1):
        g = self.grad
        # id_matrix = np.eye(len(g))
        H = self.hess
        return cubic_solver_root(g, H, M, it_max=it_max, epsilon=epsilon, r0=r0)
    
    def step(self):

        if self.value is None:
            self.value = self.loss.value(self.x)
        
        self.grad = self.loss.gradient(self.x)

        if self.cubic_solver == self.cubic_solver_root_full:
            self.hess = self.loss.hessian(self.x)

        # if self.cubic_solver is cubic_solver_krylov:    
        # self.hess = lambda v: self.loss.hess_vec_prod(self.x,v)
        # # krylov_start = time.time()
        # V, alphas, betas, beta = Lanczos(self.hess, self.grad, m=self.solver_it_max)
        # # krylov_end = time.time()
        # # print('Krylov Time {time:.3f}'.format(time=krylov_end - krylov_start))
        # self.hess = np.diag(alphas) + np.diag(betas, -1) + np.diag(betas, 1)
        # id_matrix = np.eye(len(alphas))
        # e1 = np.zeros(len(alphas))
        # e1[0] = 1
        # self.grad = np.linalg.norm(self.grad)*e1
            
        # self.hess = self.loss.hessian(self.x)

        if np.linalg.norm(self.grad) < self.tolerance:
            return
        # set the initial value of the regularization coefficient
        reg_coef = self.reg_coef*self.beta

        # LS_start = time.time()

        s_new, solver_it, r0_new, model_decrease = self.cubic_solver( 
        reg_coef, self.solver_it_max, self.solver_eps, r0 = self.r0)

        x_new = self.x + s_new
        value_new = self.loss.value(x_new)
        while value_new > self.value - model_decrease:
            reg_coef = reg_coef/self.beta
            s_new, solver_it, r0_new, model_decrease = self.cubic_solver( 
            reg_coef, self.solver_it_max, self.solver_eps, r0 = self.r0)
            x_new = self.x + s_new
            value_new = self.loss.value(x_new)
        self.x = x_new
        self.reg_coef = reg_coef
        self.value = value_new
        self.r0 = r0_new
        
        self.solver_it += solver_it
        # self.residuals.append(residual)
        # LS_end = time.time()
        # print('LS Time {time:.3f}'.format(time=LS_end - LS_start))
        
    def init_run(self, *args, **kwargs):
        super(Cubic_LS, self).init_run(*args, **kwargs)
        self.trace.solver_its = [0]
        self.loss.reset()
        
    def update_trace(self):
        super(Cubic_LS, self).update_trace()
        self.trace.solver_its.append(self.solver_it)


class Cubic_Krylov_LS(Optimizer):
    """
    Krylov subspace cubic Newton method with line search
    
    Arguments:
        
    """
    def __init__(self, reg_coef=None, subspace_dim=100, solver_eps=1e-8, beta=0.5, *args, **kwargs):
        super(Cubic_Krylov_LS, self).__init__(*args, **kwargs)    
        self.solver_it = 0
        self.subspace_dim = subspace_dim
        self.solver_eps = solver_eps

        self.beta = beta
        self.r0 = 0.1
        # self.residuals = []
        self.value = None

        if reg_coef is None:
            self.reg_coef = self.loss.hessian_lipschitz
        else:
            self.reg_coef = reg_coef
        
        # if cubic_solver == "GD": 
        #     self.cubic_solver = ls_cubic_solver
        # elif cubic_solver == "root":
        #     self.cubic_solver = cubic_solver_root
        # elif cubic_solver == "krylov":
        #     self.cubic_solver = cubic_solver_krylov
        # else:
        #     print("Error: cubic_solver not recognized")

        # if cubic_solver is None:
        #     # self.cubic_solver = ls_cubic_solver
        #     self.cubic_solver = cubic_solver_root
    
    
    def step(self):

        if self.value is None:
            self.value = self.loss.value(self.x)
        
        time_start = time.time()
        self.grad = self.loss.gradient(self.x)
        # print('Grad time: {:.3f}'.format(time.time()-time_start))

        # if self.cubic_solver is cubic_solver_krylov:    
        self.hess = lambda v: self.loss.hess_vec_prod(self.x,v)
        krylov_start = time.time()
        V, alphas, betas, beta = Lanczos(self.hess, self.grad, m=self.subspace_dim)
        krylov_end = time.time()
        # print('Krylov Time {time:.3f}'.format(time=krylov_end - krylov_start))

        self.hess = np.diag(alphas) + np.diag(betas, -1) + np.diag(betas, 1)

        e1 = np.zeros(len(alphas))
        e1[0] = 1
        self.grad = np.linalg.norm(self.grad)*e1

        # if np.linalg.norm(self.grad) < self.tolerance:
        #     return
        # set the initial value of the regularization coefficient
        reg_coef = self.reg_coef*self.beta

        # LS_start = time.time()

        s_new, solver_it, r0_new, model_decrease = cubic_solver_root(self.grad, self.hess, 
        reg_coef, epsilon = self.solver_eps, r0 = self.r0)
        x_new = self.x + V @ s_new
        value_new = self.loss.value(x_new)

        iter_count = 0
        max_iter = 20
        while value_new > self.value - model_decrease and iter_count < max_iter:
            reg_coef = reg_coef/self.beta
            s_new, solver_it, r0_new, model_decrease = cubic_solver_root(self.grad, self.hess, 
            reg_coef, epsilon = self.solver_eps, r0 = self.r0)
            x_new = self.x + V @ s_new
            value_new = self.loss.value(x_new)
            iter_count += 1
        self.x = x_new
        self.reg_coef = reg_coef
        self.value = value_new
        self.r0 = r0_new
        
        self.solver_it += solver_it

        # print('Iteration time: {:.3f}'.format(time.time()-time_start))
        # if model_decrease < 1e-6:
        #     self.subspace_dim = 1
        # self.residuals.append(residual)
        # LS_end = time.time()
        # print('LS Time {time:.3f}'.format(time=LS_end - LS_start))
        
    def init_run(self, *args, **kwargs):
        super(Cubic_Krylov_LS, self).init_run(*args, **kwargs)
        self.trace.solver_its = [0]
        self.loss.reset()
        
    def update_trace(self):
        super(Cubic_Krylov_LS, self).update_trace()
        self.trace.solver_its.append(self.solver_it)

class Cubic_Stoch_Krylov_LS(Optimizer):
    """
    Krylov subspace cubic Newton method with line search
    
    Arguments:
        
    """
    def __init__(self, reg_coef=None, subspace_dim=100, subsampling = 100, solver_eps=1e-8, beta=0.5, *args, **kwargs):
        super(Cubic_Stoch_Krylov_LS, self).__init__(*args, **kwargs)    
        self.solver_it = 0
        self.subspace_dim = subspace_dim
        self.subsampling = subsampling
        self.sampling_dim = None
        self.solver_eps = solver_eps

        self.beta = beta
        self.r0 = 0.1
        # self.residuals = []
        self.value = None

        if reg_coef is None:
            self.reg_coef = self.loss.hessian_lipschitz
        else:
            self.reg_coef = reg_coef
        
        # if cubic_solver == "GD": 
        #     self.cubic_solver = ls_cubic_solver
        # elif cubic_solver == "root":
        #     self.cubic_solver = cubic_solver_root
        # elif cubic_solver == "krylov":
        #     self.cubic_solver = cubic_solver_krylov
        # else:
        #     print("Error: cubic_solver not recognized")

        # if cubic_solver is None:
        #     # self.cubic_solver = ls_cubic_solver
        #     self.cubic_solver = cubic_solver_root
    
    
    def step(self):

        if self.value is None:
            self.value = self.loss.value(self.x)
        if self.sampling_dim is None: 
            self.sampling_dim = int(self.dim/self.subsampling)
        # sample random coordinates
        I = self.rng.choice(self.dim, size=self.sampling_dim, replace=False)
        
        self.grad = self.loss.partial_gradient(self.x, I)

        # if self.cubic_solver is cubic_solver_krylov:    
        self.hess = lambda v: self.loss.partial_hess_vec_prod(self.x,v,I)
            # krylov_start = time.time()
        V, alphas, betas, beta = Lanczos(self.hess, self.grad, m=self.subspace_dim)
            # krylov_end = time.time()
            # print('Krylov Time {time:.3f}'.format(time=krylov_end - krylov_start))
        self.hess = np.diag(alphas) + np.diag(betas, -1) + np.diag(betas, 1)

        e1 = np.zeros(len(alphas))
        e1[0] = 1
        self.grad = np.linalg.norm(self.grad)*e1

        # if np.linalg.norm(self.grad) < self.tolerance:
        #     return
        # set the initial value of the regularization coefficient
        reg_coef = self.reg_coef*self.beta

        # LS_start = time.time()
        x_new = copy.deepcopy(self.x)
        s_new, solver_it, r0_new, model_decrease = cubic_solver_root(self.grad, self.hess, 
        reg_coef, epsilon = self.solver_eps, r0 = self.r0)
        x_new[I] = self.x[I] + V @ s_new
        value_new = self.loss.value(x_new)

        iter_count = 0
        max_iter = 20
        while value_new > self.value - model_decrease and iter_count < max_iter:
            reg_coef = reg_coef/self.beta
            s_new, solver_it, r0_new, model_decrease = cubic_solver_root(self.grad, self.hess, 
            reg_coef, epsilon = self.solver_eps, r0 = self.r0)
            x_new[I] = self.x[I] + V @ s_new
            value_new = self.loss.value(x_new)
            iter_count += 1
        self.x = x_new
        self.reg_coef = reg_coef
        self.value = value_new
        self.r0 = r0_new
        
        self.solver_it += solver_it

        # if model_decrease < 1e-6:
        #     self.subspace_dim = 1
        # self.residuals.append(residual)
        # LS_end = time.time()
        # print('LS Time {time:.3f}'.format(time=LS_end - LS_start))
        
    def init_run(self, *args, **kwargs):
        super(Cubic_Stoch_Krylov_LS, self).init_run(*args, **kwargs)
        self.trace.solver_its = [0]
        self.loss.reset()
        
    def update_trace(self):
        super(Cubic_Stoch_Krylov_LS, self).update_trace()
        self.trace.solver_its.append(self.solver_it)

## Test 

class Cubic_Krylov_LS_test(Optimizer):
    """
    Krylov subspace cubic Newton method with line search
    
    Arguments:
        
    """
    def __init__(self, reg_coef=None, subspace_dim=100, solver_eps=1e-8, beta=0.5, *args, **kwargs):
        super(Cubic_Krylov_LS, self).__init__(*args, **kwargs)    
        self.solver_it = 0
        self.subspace_dim = subspace_dim
        self.solver_eps = solver_eps

        self.beta = beta
        self.r0 = 0.1
        # self.residuals = []
        self.value = None

        if reg_coef is None:
            self.reg_coef = self.loss.hessian_lipschitz
        else:
            self.reg_coef = reg_coef
        
        # if cubic_solver == "GD": 
        #     self.cubic_solver = ls_cubic_solver
        # elif cubic_solver == "root":
        #     self.cubic_solver = cubic_solver_root
        # elif cubic_solver == "krylov":
        #     self.cubic_solver = cubic_solver_krylov
        # else:
        #     print("Error: cubic_solver not recognized")

        # if cubic_solver is None:
        #     # self.cubic_solver = ls_cubic_solver
        #     self.cubic_solver = cubic_solver_root
        self.spectra = []

    def step(self):
        
        if self.value is None:
            self.value = self.loss.value(self.x)
        
        
        self.grad = self.loss.gradient(self.x)


        self.hess = self.hess(self.x)
        h_eigs = eigh(self.hess, eigvals_only=True)
        self.spectra.append(h_eigs)

        # if self.cubic_solver is cubic_solver_krylov:    
        self.hess = lambda v: self.loss.hess_vec_prod(self.x,v)
            # krylov_start = time.time()
        V, alphas, betas, beta = Lanczos(self.hess, self.grad, m=self.subspace_dim)
            # krylov_end = time.time()
            # print('Krylov Time {time:.3f}'.format(time=krylov_end - krylov_start))
        self.hess = np.diag(alphas) + np.diag(betas, -1) + np.diag(betas, 1)

        e1 = np.zeros(len(alphas))
        e1[0] = 1
        self.grad = np.linalg.norm(self.grad)*e1

        if np.linalg.norm(self.grad) < self.tolerance:
            return
        # set the initial value of the regularization coefficient
        reg_coef = self.reg_coef*self.beta

        # LS_start = time.time()

        s_new, solver_it, r0_new, model_decrease = cubic_solver_root(self.grad, self.hess, 
        reg_coef, epsilon = self.solver_eps, r0 = self.r0)
        x_new = self.x + V @ s_new
        value_new = self.loss.value(x_new)
        while value_new > self.value - model_decrease:
            reg_coef = reg_coef/self.beta
            s_new, solver_it, r0_new, model_decrease = cubic_solver_root(self.grad, self.hess, 
            reg_coef, epsilon = self.solver_eps, r0 = self.r0)
            x_new = self.x + V @ s_new
            value_new = self.loss.value(x_new)
        self.x = x_new
        self.reg_coef = reg_coef
        self.value = value_new
        self.r0 = r0_new
        
        self.solver_it += solver_it

        # if model_decrease < 1e-6:
        #     self.subspace_dim = 1
        # self.residuals.append(residual)
        # LS_end = time.time()
        # print('LS Time {time:.3f}'.format(time=LS_end - LS_start))
        
    def init_run(self, *args, **kwargs):
        super(Cubic_Krylov_LS, self).init_run(*args, **kwargs)
        self.trace.solver_its = [0]
        self.loss.reset()
        
    def update_trace(self):
        super(Cubic_Krylov_LS, self).update_trace()
        self.trace.solver_its.append(self.solver_it)

class SSCN(Optimizer):
    """
    Newton method with cubic regularization with line search for global convergence.
    The method was studied by Nesterov and Polyak in the following paper:
        "Cubic regularization of Newton method and its global performance"
        https://link.springer.com/article/10.1007/s10107-006-0706-8
    
    Arguments:
        reg_coef (float, optional): an estimate of the Hessian's Lipschitz constant
    """
    def __init__(self, reg_coef=None, subspace_dim=100, solver_eps=1e-8, beta=0.5, *args, **kwargs):
        super(SSCN, self).__init__(*args, **kwargs)
        self.reg_coef = reg_coef
        self.solver_it = 0
        self.subspace_dim = subspace_dim
        self.solver_eps = solver_eps

        self.beta = beta
        self.r0 = 0.1
        self.residuals = []
        self.value = None
        self.tolerance = 0

        if reg_coef is None:
            self.reg_coef = self.loss.hessian_lipschitz

        self.reuse = False
    
    def step(self):
        if self.value is None:
            self.value = self.loss.value(self.x)
        
        # sample random coordinates
        I = self.rng.choice(self.dim, size=self.subspace_dim, replace=False)
        
        # partial derivative information

        grad_start = time.time()
        self.grad = self.loss.partial_gradient(self.x, I)
        grad_end = time.time()
        # print('Gradient time: {}'.format(grad_end-grad_start))

        hess_start = time.time()
        self.hess = self.loss.partial_hessian(self.x, I)
        hess_end = time.time()
        # print('Hessian time: {}'.format(hess_end-hess_start))

        # if np.linalg.norm(self.grad) < self.tolerance:
        #     return
        # set the initial value of the regularization coefficient
        reg_coef = max(self.reg_coef*self.beta, np.finfo(float).eps)


        # x_sub = self.x[I]
        x_new = copy.deepcopy(self.x)
        Ax = copy.deepcopy(self.loss._mat_vec_prod)

        time_start = time.time()
        s_new_sub, solver_it, r0_new, model_decrease = cubic_solver_root(self.grad, self.hess, 
        reg_coef, r0 = self.r0, epsilon=np.finfo(float).eps)
        x_new[I] = self.x[I] + s_new_sub
        # print('Cubic solving time: {}'.format(time.time()-time_start))

        self.loss.update_mat_vec_product(Ax, s_new_sub, I)
        
        value_new = self.loss.value(x_new)
        while value_new > self.value - model_decrease:
            reg_coef = reg_coef/self.beta
            s_new_sub, solver_it, r0_new, model_decrease = cubic_solver_root(self.grad, self.hess, 
            reg_coef, r0 = self.r0, epsilon=np.finfo(float).eps)
            x_new[I] = self.x[I] + s_new_sub

            self.loss.update_mat_vec_product(Ax, s_new_sub, I)
            value_new = self.loss.value(x_new)
        self.x = x_new
        self.reg_coef = reg_coef
        self.value = value_new
        self.r0 = r0_new

        
        
        self.solver_it += solver_it
        # self.residuals.append(residual)
        
    def init_run(self, *args, **kwargs):
        super(SSCN, self).init_run(*args, **kwargs)
        self.trace.solver_its = [0]
        self.loss.reset()
        
    def update_trace(self):
        super(SSCN, self).update_trace()
        self.trace.solver_its.append(self.solver_it)