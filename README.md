# TR-StoSQP

This respository reproduces the numerical results of the paper [Fully Stochastic Trust-Region Sequential Quadratic Programming for Equality-Constrained Optimization Problems](https://arxiv.org/abs/2211.15943). 

We implement TR-StoSQP method to solve (1) 47 probblems with equality constraints from CUTEst test set, and (2) 8 Logistic regression problems with equality constraints using data from LIBSVM collection.

The setup of CUTEst is available at [https://github.com/JuliaSmoothOptimizers/CUTEst.jl.](https://github.com/JuliaSmoothOptimizers/CUTEst.jl.)

The datasets of LIBSVM collection can be downloaded from [https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)

To use code, please cite the paper
```
@article{fang2022fully,
  title={Fully Stochastic Trust-Region Sequential Quadratic Programming for Equality-Constrained Optimization Problems},
  author={Fang, Yuchen and Na, Sen and Mahoney, Michael W and Kolar, Mladen},
  journal={arXiv preprint arXiv:2211.15943},
  year={2022}
}
```
