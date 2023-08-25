# Krylov subspace Newton Method

This repository implements Krylov subspace cubic Newton method ([Overleaf document](https://www.overleaf.com/5933328532vkddhzzkhkjb)). 
## How to run
To run the algorithms and produce the plots, run "cubic_newton.py": 
```
python cubic_newton.py
```
## File structure
- Under "optimizer": Implementations of loss functions and optimziation algorithms. They are adapted from https://github.com/konstmish/opt_methods
- [cubic_newton.ipynb](cubic_newton.ipynb): the jupyter notebook version of "python cubic_newton.py", for quick debugging and exploration. 
- [Lanczos.ipynb](Lanczos.ipynb): the jupyter notebook for implementing Lanczos method. 
- [newton.ipynb](newton.ipynb): the regularized Newton method from https://github.com/konstmish/global-newton
