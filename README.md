# GAC
This is our Tensorflow implementation for our Knowledge-Based Systems 2021 paper(under review) and a part of baselines:

>Qi Zhang, Bin Wu, Zhongchuan Sun, Yangdong Ye(2021).

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.14.0
* numpy == 1.16.4
* scipy == 1.3.1
* pandas == 0.17

## C++ evaluator
We have implemented C++ code to output metrics during and after training, which is much more efficient than python evaluator. It needs to be compiled first using the following command. 
```
python setup.py build_ext --inplace
```
If the compilation is successful, the evaluator of cpp implementation will be called automatically.
Otherwise, the evaluator of python implementation will be called.
NOTE: The cpp implementation is much faster than python.**

## Examples to run GAC:
run [main.py](./main.py) in IDE or with command line:
```
python main.py
```

## Baselines
The list of available models in GAC, along with their paper citations, are shown below:

| General Recommender | Paper                                                                                                         |
|---------------------|---------------------------------------------------------------------------------------------------------------|
| BPRMF               | Steffen Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.                   |

| Sequential Recommender | Paper                                                                                                      |
|------------------------|------------------------------------------------------------------------------------------------------------|
| FPMC                   | Steffen Rendle et al., Factorizing Personalized Markov Chains for Next-Basket Recommendation, WWW 2010.    |
| SASRec                 | Wangcheng Kang et al., Self-Attentive Sequential Recommendation, ICDM 2018.                                |
