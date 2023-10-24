# Krylov Cubic Regularized Newton: A Subspace Second-Order Method with Dimension-Free Convergence Rate

This repository implements the Krylov cubic regularized Newton method.   
## How to run
To reproduce the plots on the "w8a" dataset (the first subfigures in Figures 2 and 3), run "cubic_newton.py": 
```
python cubic_newton.py --time --it_max 50000 --time_max 60
python cubic_newton.py --it_max 50 --time_max 60000
```

To reproduce the plots on the "rcv1_train" and the "news20" datasets (the other subfigures in Figures 2 and 3), run the Jupyter notebook "cubic_newton.ipynb".

## Acknowledgment
Our implementation is partially based on the code by XXX (We will deanonymize the link and include the proper license if the submission is accepted). 
