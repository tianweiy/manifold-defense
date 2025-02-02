# Adversarial Training Using the Overpowered Attack
This repository contains code to perform adversarial training using the overpowered attack proposed in our paper
[The Robust Manifold Defense: Adversarial Training Using Generative Models](https://arxiv.org/abs/1712.09196)

# Setup
---
1. To set up the necessary environment assuming you use Python3.5, please do:
```shell
$./setup.sh
```
If you don't use Python3.5, please edit ```setup.sh``` appropriately.

2. Once the environment is set up, please run the following to activate it:
```shell
$ source env/bin/activate
```
---

# Training Baseline Madry Model
To train a baseline model similar to Madry et al., do
```shell
$ python adv_train.py --dataset mnist --mode l2 --eps 1.5 --validation-set --opt adam --sgd-lr 1e-4 --no-norm --save-str <string> --num-epochs <num-epochs> --save-iters <save-frequency> --random-step
```

This will train a classifier on the mnist dataset with white box PGD L2 perturbation with norm 1.5. The checkpoint with best validation set accuracy will be saved in ```results/``` as ```results/mnist_l2_<string>_best```. 

If you choose to save checkpoints at regular intervals, then the checkpoints will be saved as ```results/mnist_l2_<string>_<epoch>``` where ```<epoch>``` is an integer multiple of ```<save-frequency>```.

If you only want to save the best model, choose ```<save-frequency>``` to be greater than ```<num-epochs>```.

The baseline checkpoint available in the folder ```results/``` was trained using the following command:
```shell
$ python adv_train.py --dataset mnist --mode l2 --eps 1.5 --num-pgd 40 --validation-set --opt adam --sgd-lr 1e-4 --no-norm --save-str baseline --num-epochs 500 --save-iters 501 --random-step 
```

# Adversarial Training Using Overpowered Attack
To train using the ovepowered attack, run the following:
```shell
$ python adv_train.py --dataset mnist --mode l2 --eps 1.5 --num-pgd 40 --validation-set --opt adam --sgd-lr 1e-4 --no-norm --save-str <string> --resume baseline_best --num-epochs <num-epochs> --save-iters <save-frequency> --random-step --op-attack --op-generator checkpoints/trained_vae_leakyrelu_20_500_500_784.pth --op-embed-feats 20 --op-iter 5 --op-weight 1e-2
```

The checkpoint available in the folder ```results/``` was trained using the following command:
```shell
$ python adv_train.py --dataset mnist --mode l2 --eps 1.5 --num-pgd 40 --validation-set --opt adam --sgd-lr 1e-4 --no-norm --save-str op --resume baseline_best --random-step --op-attack --op-generator checkpoints/trained_vae_leakyrelu_20_500_500_784.pth --op-embed-feats 20 --op-iter 5 --op-weight 1e-2
```

# Evaluation
To evaluate robustness of a classifier:
```shell
$ python eval_network.py --dataset mnist --net-path <PATH-TO-CLASSIFIER> --mode l2 --num-steps <NUMBER-OF-STEPS> --eps <EPS> --pgd-lr <PGD-LR> --random-step 
```

For example, to evaluate our overpowered model with epsilon=2.5, and 100 step PGD (the default step size is 2*epsilon/steps, but can be changed), do:
```shell
$ python eval_network.py --dataset mnist --net-path results/mnist_l2_op_best --eps 2.5 --mode l2 --num-steps 100 --random-step
```
