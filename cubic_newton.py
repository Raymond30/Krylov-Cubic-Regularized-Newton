import sklearn.datasets
import urllib.request

import numpy as np
import matplotlib.pyplot as plt

from optimizer.loss import LogisticRegression
from optimizer.cubic import Cubic, Cubic_LS, SSCN
from optimizer.GD import Gd, GD_LS
from optimizer.reg_newton import RegNewton


if __name__ == '__main__':
    # Define the loss function
    dataset = 'madelon'
    # if dataset == 'mushrooms':
    #     data_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms"
    #     data_path = './mushrooms'
    # else:
    #     data_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a"
    #     data_path = './w8a'
    data_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{}".format(dataset)
    data_path = './{}'.format(dataset)

    f = urllib.request.urlretrieve(data_url, data_path)
    A, b = sklearn.datasets.load_svmlight_file(data_path)
    A = A.toarray()

    # Logistic regression problem
    loss = LogisticRegression(A, b, l1=0, l2=0, store_mat_vec_prod=True)
    n, dim = A.shape
    L = loss.smoothness
    # l2 = 1e-10 * L # make the problem ill-conditioned
    # loss.l2 = l2
    x0 = np.ones(dim) * 0.5
    it_max = 200

    # Define the optimization algs
    flag_LS = True # True for LS, False for fixed learning rate/regularization parameter

    memory_size = 10
    if not flag_LS:
        gd = Gd(loss=loss, label='GD')
        # grid search for lr
        lr_gd = np.geomspace(start=1e0, stop=1e6, num=7) / L
        cub_krylov = Cubic(loss=loss, label='Cubic Newton (Krylov dim = {})'.format(memory_size),
                           cubic_solver="krylov", solver_it_max=memory_size)
        cub_root = Cubic(loss=loss, label='Cubic Newton',cubic_solver="root")
        # grid search for regularization parameter
        reg_cub = np.geomspace(start=1e-8, stop=1e-2, num=7) * loss.hessian_lipschitz
    else:
        gd = GD_LS(loss=loss, label='GD LS')
        cub_krylov = Cubic_LS(loss=loss, label='Cubic Newton LS (Krylov dim = {})'.format(memory_size),
                              cubic_solver="krylov", solver_it_max=memory_size, tolerance = 1e-9)
        cub_root = Cubic_LS(loss=loss, label='Cubic Newton LS',cubic_solver="root", tolerance = 1e-8)
        sscn = SSCN(loss=loss, label='SSCN (subspace dim = {})'.format(memory_size),
                               sub_dim=memory_size, tolerance = 1e-9)

    # A benchmark algorithm that is used to compute the optimal solution
    adan = RegNewton(loss=loss, adaptive=True, use_line_search=True, 
                     label='AdaN')
    
    # benchmark: SSCN
    print(f'Running optimizer: {sscn.label}')
    sscn.run(x0=x0, it_max=it_max)
    sscn.compute_loss_of_iterates()
    
    # Running algs
    if not flag_LS:
        best_loss_gd = np.inf
        best_lr_gd = None
        for lr in lr_gd:
            print(f'Running optimizer: {gd.label} with lr={lr}')
            gd.lr = lr
            gd.run(x0=x0, it_max=it_max)
            gd.compute_loss_of_iterates()
            loss_gd = gd.trace.best_loss_value
            print(f'Loss value: {loss_gd}')
            if loss_gd < best_loss_gd:
                best_loss_gd = loss_gd
                best_lr_gd = lr
                # save the best result    
                gd.trace.save('GD_best')
            gd.reset(loss=loss)
        
        # load the best result
        gd.from_pickle('./results/GD_best',loss=loss)
        # gd.trace.label = f'GD (lr={best_lr_gd})'
    else:
        print(f'Running optimizer: {gd.label}')
        gd.run(x0=x0, it_max=it_max)
        gd.compute_loss_of_iterates()

    print(f'Running optimizer: {adan.label}')
    adan.run(x0=x0, it_max=100)
    adan.compute_loss_of_iterates()

    # print(gd.trace.loss_vals)

    # print(f'Running optimizer: {cub_krylov.label}-Krylov')
    # cub_krylov.run(x0=x0, it_max=it_max)
    # cub_krylov.compute_loss_of_iterates() 

    # print(f'Running optimizer: {cub_root.label}')
    # cub_root.run(x0=x0, it_max=it_max)
    # cub_root.compute_loss_of_iterates()

    if not flag_LS:
        best_loss_cub_root = np.inf
        best_reg_cub_root = None
        for reg in reg_cub:
            print(f'Running optimizer: {cub_root.label} with reg={reg}')
            cub_root.reg_coef = reg
            cub_root.run(x0=x0, it_max=it_max)
            cub_root.compute_loss_of_iterates()
            loss_cub_root = cub_root.trace.best_loss_value
            print(f'Best loss value: {loss_cub_root}')
            if loss_cub_root < best_loss_cub_root:
                best_loss_cub_root = loss_cub_root
                best_reg_cub_root = reg    
                cub_root.trace.save('Cubic_best')
            cub_root.reset(loss=loss)
        
        cub_root.from_pickle('./results/Cubic_best',loss=loss)
        # cub_root.trace.label = f'Cubic Newton (Reg={best_reg_cub_root})'

        best_loss_cub_krylov = np.inf
        best_reg_cub_krylov = None
        for reg in reg_cub:
            print(f'Running optimizer: {cub_krylov.label} with reg={reg}')
            cub_krylov.reg_coef = reg
            cub_krylov.run(x0=x0, it_max=it_max)
            cub_krylov.compute_loss_of_iterates()
            loss_cub_krylov = cub_krylov.trace.best_loss_value
            print(f'Best loss value: {loss_cub_krylov}')
            if loss_cub_krylov < best_loss_cub_krylov:
                best_loss_cub_krylov = loss_cub_krylov
                best_reg_cub_krylov = reg    
                cub_krylov.trace.save('Cubic_krylov_best')
            cub_krylov.reset(loss=loss)
        
        cub_krylov.from_pickle('./results/Cubic_krylov_best',loss=loss)
        # cub_krylov.trace.label = f'Cubic Newton (Krylov dim ={memeory_size}, Reg={best_reg_cub_root})'

    else:
        print(f'Running optimizer: {cub_krylov.label}')
        cub_krylov.run(x0=x0, it_max=it_max)
        cub_krylov.compute_loss_of_iterates()

        print(f'Running optimizer: {cub_root.label}')
        cub_root.run(x0=x0, it_max=it_max)
        cub_root.compute_loss_of_iterates()




    # Plot the loss curve
    flag_time = True # True for time, False for iteration
    gd.trace.plot_losses(marker='^', time=flag_time)
    # cub_krylov.trace.plot_losses(marker='>', label='cubic Newton (Krylov subspace)')
    # cub_root.trace.plot_losses(marker='o', label='cubic Newton (exact)')
    cub_root.trace.plot_losses(marker='s', time=flag_time)
    cub_krylov.trace.plot_losses(marker='d', time=flag_time)

    sscn.trace.plot_losses(marker='o', time=flag_time)

    # print(cub.trace.loss_vals)
    ## plt.xscale('log')
    if flag_time:
        plt.xlabel('Time')
    else:
        plt.xlabel('Iterations')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig('figs/logistic.pdf')
    plt.show()