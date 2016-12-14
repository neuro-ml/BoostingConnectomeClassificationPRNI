Source code for paper 'Boosting connectome classification via combination of geometric and topological normalizations', poster on 2016 International Workshop on Pattern Recognition in Neuroimaging (PRNI).

Link to paper: http://ieeexplore.ieee.org/abstract/document/7552353/

## How to run docker

Clone:

``` bash
git clone https://github.com/neuro-ml/BoostingConnectomeClassificationPRNI.git
cd BoostingConnectomeClassificationPRNI
```

Build:

``` bash
docker build -t boost_clf_prni:repr -f Dockerfile .
```

Run Container:

``` bash
docker run -it -p 8809:8888 boost_clf_prni:repr bash jupyter notebook --no-browser --ip="*"
```

Open http://localhost:8809 on your local machine.
