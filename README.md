MultiClassLogitBoost
==============
(Various) LogitBoost for multiclass classification with ensemble of vector-valued trees.

About one year after we published AOSOLogitBoost [1], we found that a simple LogitBoost variant also works very well for multiclass classification, sometimes even outperforms AOSOLogitBoost. This new variant, called `VTLogitBoost`, can be explained in a few lines of words if you are already very familiar with multiclass LogitBoost:

* The tree is still vector valued, as in AOSOLogitBoost
* At each node, all the `K` classes are updated
* The gain for node split is derived by diagonal approximation of the Hessian matrix

We never published `VTLogitBoost` anywhere, so we simply put the code here (for research purpose). In the folder there are also many other `VTLogitBoost` variants, including those with importance sampling, total correction, etc, although some of them see a degraded performance, which, we doubt, should attribute to our problematic implementation. However, the `VTLogitBoost` is guaranteed to  improve over/be on par with `AOSOLogitBoost`. 

The code is somewhat messy, but does the job. Also, it is not well documented, but the calling convention is similar to `AOSOLogitBoost`, see [the doc there](https://github.com/pengsun/AOSOLogitBoost.git). Feel free to contact me if you need any further help.

References
----------
[1] "Peng Sun, Mark D. Reid, Jie Zhou. "AOSO-LogitBoost: Adaptive One-Vs-One LogitBoost for Multi-Class Problems", International Conference on Machine Learning (ICML 2012)"
