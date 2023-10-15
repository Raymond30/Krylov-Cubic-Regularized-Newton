import sklearn.datasets
import urllib.request
import os.path

import numpy as np
import matplotlib.pyplot as plt

from optimizer.loss import LogisticRegression
from optimizer.cubic import Cubic, Cubic_LS, Cubic_Krylov_LS, Cubic_Stoch_Krylov_LS, SSCN
from optimizer.LBFGS import Lbfgs
from optimizer.GD import Gd, GD_LS
from optimizer.reg_newton import RegNewton

import time

if __name__ == '__main__':
    # Define the loss function
    # dataset = 'gisette_scale'
    # dataset = 'madelon'
    # dataset = 'w8a'
    # dataset = 'rcv1_train.binary'
    dataset = 'news20.binary'
    # if dataset == 'mushrooms':
    #     data_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms"
    #     data_path = './mushrooms'
    # else:
    #     data_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a"
    #     data_path = './w8a'
    data_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{}".format(dataset)

    if dataset == 'gisette_scale' or dataset == 'duke' or dataset == 'rcv1_train.binary' or dataset == 'news20.binary': # or dataset == 'epsilon_normalized':
        data_path = './{}.bz2'.format(dataset)
    else:
        data_path = './{}'.format(dataset)
    if not os.path.exists(data_path):
        f = urllib.request.urlretrieve(data_url, data_path)
    A, b = sklearn.datasets.load_svmlight_file(data_path)

    time_start = time.time()
    A_csc = A.tocsc()
    print('tocsc time: {}'.format(time.time()-time_start))
    # A = A.toarray()

    # Logistic regression problem
    loss = LogisticRegression(A, b, l1=0, l2=0, store_mat_vec_prod=True)

    loss_csc = LogisticRegression(A_csc, b, l1=0, l2=0, store_mat_vec_prod=True)
    n, dim = A.shape
    L = loss.smoothness
    # l2 = 1e-10 * L # make the problem ill-conditioned
    # loss.l2 = l2
    x0 = np.ones(dim) * 0.5

    flag_time = True # True for time, False for iteration
    it_max = 50000
    time_max = 30

    # Define the optimization algs
    flag_LS = True # True for LS, False for fixed learning rate/regularization parameter

    memory_size = 10
    
    
    gd = GD_LS(loss=loss, label='GD LS')

    krylov_dim = 10
    cub_krylov = Cubic_Krylov_LS(loss=loss, reg_coef= 1e-3, label='Cubic Newton LS (Krylov dim = {})'.format(krylov_dim),
                            subspace_dim=krylov_dim, tolerance = 1e-9)
    
    # cub_krylov = Cubic_Stoch_Krylov_LS(loss=loss, reg_coef = 1, label='Stochastic Cubic Newton LS (Krylov dim = {})'.format(memory_size),
    #                     subsampling=memory_size, 
    #                     subspace_dim=memory_size, tolerance = 1e-9)
    
    # cub_root = Cubic_LS(loss=loss, reg_coef = 1, cubic_solver= "full", label='Cubic Newton LS', tolerance = 1e-8)
    sscn = SSCN(loss=loss_csc, reg_coef= 1e-3, label='SSCN (subspace dim = {})'.format(memory_size),
                            subspace_dim=memory_size, tolerance = 1e-9)

    # A benchmark algorithm that is used to compute the optimal solution
    adan = RegNewton(loss=loss, adaptive=True, use_line_search=True, 
                     label='AdaN')
    
    lbfgs = Lbfgs(loss=loss,mem_size=memory_size, label='LBFGS (m={})'.format(memory_size))
    
    # print(f'Running optimizer: {cub_krylov.label}')
    # cub_krylov.run(x0=x0, it_max=it_max, t_max=time_max)
    # cub_krylov.compute_loss_of_iterates()

    # print(f'Running optimizer: {cub_root.label}')
    # cub_root.run(x0=x0, it_max=it_max, t_max=time_max)
    # cub_root.compute_loss_of_iterates()

    
    # Running algs

 

    # print(f'Running optimizer: {cub_root.label}')
    # cub_root.run(x0=x0, it_max=it_max, t_max=time_max)
    # cub_root.compute_loss_of_iterates()

    # print(f'Running optimizer: {gd.label}')
    gd.run(x0=x0, it_max=it_max, t_max=time_max)
    gd.compute_loss_of_iterates()

    # # benchmark: SSCN
    print(f'Running optimizer: {sscn.label}')
    sscn.run(x0=x0, it_max=it_max, t_max=time_max)
    sscn.compute_loss_of_iterates()

    print(f'Running optimizer: {cub_krylov.label}')
    cub_krylov.run(x0=x0, it_max=it_max, t_max=time_max)
    cub_krylov.compute_loss_of_iterates()

    

    # # benchmark: SSCN
    # print(f'Running optimizer: {lbfgs.label}')
    # lbfgs.run(x0=x0, it_max=it_max, t_max=time_max)
    # lbfgs.compute_loss_of_iterates()

    


    # it_max_adan = 100
    # print(f'Running optimizer: {adan.label}')
    # adan.run(x0=x0, it_max=it_max_adan,t_max=time_max)
    # adan.compute_loss_of_iterates()

    # print(gd.trace.loss_vals)



    # print(f'Running optimizer: {cub_root.label}')
    # cub_root.run(x0=x0, it_max=it_max)
    # cub_root.compute_loss_of_iterates()

    

    
    # print(f'Running optimizer: {cub_krylov.label}')
    # cub_krylov.run(x0=x0, it_max=it_max, t_max=time_max)
    # cub_krylov.compute_loss_of_iterates()

    # print(f'Running optimizer: {cub_root.label}')
    # cub_root.run(x0=x0, it_max=it_max, t_max=time_max)
    # cub_root.compute_loss_of_iterates()


    f_opt = min(loss.f_opt, loss_csc.f_opt)

    # Plot the loss curve
    # plt.style.use('tableau-colorblind10')
    gd.trace.plot_losses(marker='^', f_opt=f_opt, time=flag_time)
    # cub_krylov.trace.plot_losses(marker='>', label='cubic Newton (Krylov subspace)')
    # cub_root.trace.plot_losses(marker='o', label='cubic Newton (exact)')

    

    cub_krylov.trace.plot_losses(marker='d', f_opt=f_opt, time=flag_time)

    sscn.trace.plot_losses(marker='o', f_opt=f_opt, time=flag_time)

    # cub_root.trace.plot_losses(marker='s', time=flag_time)

    # lbfgs.trace.plot_losses(marker='X', time=flag_time)

    # print(cub.trace.loss_vals)
    ## plt.xscale('log')
    if flag_time:
        plt.xlabel('Time')
    else:
        plt.xlabel('Iterations')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    # if flag_time:
    #     plt.savefig('figs/logistic_time_{}.pdf'.format(dataset))
    # else:
    #     plt.savefig('figs/logistic_iteration_{}.pdf'.format(dataset))
    plt.show()