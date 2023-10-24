import argparse
import sklearn.datasets
import urllib.request
import os.path

import numpy as np
import matplotlib.pyplot as plt

from optimizer.loss import LogisticRegression
from optimizer.cubic import Cubic_LS, Cubic_Krylov_LS, SSCN

import seaborn as sns

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cubic Regularized Newton Methods')
    parser.add_argument('--dataset', metavar='DATASETS', default='w8a', type=str,
                    help='The dataset')
    parser.add_argument('--time', dest='flag_time', action='store_true',
                    help='Plot with respect to time')
    parser.add_argument('--it_max', default=50000, type=int, metavar='IT',
                    help='max iteration')
    parser.add_argument('--time_max', default=60, type=float, metavar='T',
                    help='max time')
    
    args = parser.parse_args()
    dataset = args.dataset
    flag_time = args.flag_time
    it_max = args.it_max
    time_max = args.time_max
    # Define the loss function
    # dataset = 'gisette_scale'
    # dataset = 'madelon'
    
    # dataset = 'rcv1_train.binary'
    # dataset = 'news20.binary'
    data_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{}".format(dataset)

    if dataset in {'gisette_scale','duke','rcv1_train.binary','news20.binary'}: # or dataset == 'epsilon_normalized':
        data_path = './{}.bz2'.format(dataset)
    else:
        data_path = './{}'.format(dataset)
    if not os.path.exists(data_path):
        f = urllib.request.urlretrieve(data_url, data_path)
    A, b = sklearn.datasets.load_svmlight_file(data_path)
    A_csc = A.tocsc()

    # Define loss functions
    loss = LogisticRegression(A, b, l1=0, l2=0, store_mat_vec_prod=True)
    loss_csc = LogisticRegression(A_csc, b, l1=0, l2=0, store_mat_vec_prod=True)
    n, dim = A.shape
    x0 = np.ones(dim) * 0.5

    # flag_time = True # True for time, False for iteration
    # it_max = 50000
    # time_max = 60

    # Define the optimization algs
    memory_size = 10
    cub_krylov = Cubic_Krylov_LS(loss=loss, reg_coef = 1e-3, label='Krylov CRN (m = {})'.format(memory_size),
                            subspace_dim=memory_size, tolerance = 1e-9)
    
    cub_root = Cubic_LS(loss=loss, reg_coef = 1e-3, label='CRN', cubic_solver="full", tolerance = 1e-8)
    sscn = SSCN(loss=loss_csc, reg_coef = 1e-3, label='SSCN (m = {})'.format(memory_size),
                            subspace_dim=memory_size, tolerance = 1e-9)


    
    # Running algs

    # SSCN
    print(f'Running optimizer: {sscn.label}')
    sscn.run(x0=x0, it_max=it_max, t_max=time_max)
    sscn.compute_loss_of_iterates()


    print(f'Running optimizer: {cub_krylov.label}')
    cub_krylov.run(x0=x0, it_max=it_max, t_max=time_max)
    cub_krylov.compute_loss_of_iterates()

    print(f'Running optimizer: {cub_root.label}')
    cub_root.run(x0=x0, it_max=it_max, t_max=time_max)
    cub_root.compute_loss_of_iterates()




    # Plot the loss curve
    # plt.style.use('tableau-colorblind10')
    sns.set_style('ticks') # setting style
    # sns.set_context('paper') # setting context
    sns.set_palette('colorblind') # setting palette

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    f_opt = min(loss.f_opt, loss_csc.f_opt)
    # gd.trace.plot_losses(marker='^', time=flag_time)
    # cub_krylov.trace.plot_losses(marker='>', label='cubic Newton (Krylov subspace)')
    # cub_root.trace.plot_losses(marker='o', label='cubic Newton (exact)')


    cub_root.trace.plot_losses(marker='o', markersize=5, f_opt=f_opt, time=flag_time, label='CRN')

    sscn.trace.plot_losses(marker='^', markersize=6, f_opt=f_opt, time=flag_time)

    cub_krylov.trace.plot_losses(marker='v', markersize=6, f_opt=f_opt, time=flag_time, color = color_cycle[7], label='Krylov CRN (m = 10)')
    if flag_time:
        plt.xlabel('Time (s)')
    else:
        plt.xlabel('Iteration')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.title('{} ($n={}$, $d={}$)'.format(dataset,n,dim))
    if flag_time:
        plt.savefig('figs/time_{}.pdf'.format(dataset))
    else:
        plt.savefig('figs/iteration_{}.pdf'.format(dataset))
    plt.show()