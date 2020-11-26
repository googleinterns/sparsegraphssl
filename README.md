# Improving semi-supervised learning with sparse attention and pseudo-labels propagation.

## Package description:

Here is the original repository for the models introduced in "Improving
semi-supervised learning with sparse attention and pseudo-labels propagation."
Our graph neural networks model utilizes SparseMax attention to prune
unessential connections to let the networks focus on a few edges only.

Our Sparsity-inducing graph neural network (SIGN) improves the selection of
meaningful samples significantly by up to 8.01 % compared to the Graph Attention
networks baselines. The model is up to 20% more robust to added noisy edges
while achieving similar or slightly improved classification performance. As a
side effect, SIGN also enhances the explanation of the classification decisions.
With our model, the attention mass is concentrated only on a few crucial samples
since SIGN sparsifies the graph connections. For selected samples, SIGN can
reduce the interpretation graph by more than 99%.

## Set-up the environment and required packages.

The easiest way is to set up a virtual environment with pip or conda. Below we
show an example to set up the pip environment.

### Installation

You will need the packages virtualenv, graphviz, python3-tk. graphviz python3-tk
are used for visualizations of the learned attention weights.

```bash
sudo apt-get --assume-yes install virtualenv graphviz-dev python3-tk
```

### Virtual environment to run on GPUs:

```bash
virtualenv --python=python3.7 [YourENV]
source [YourENV]/bin/activate
pip install -r requirements.txt
```

### Running experiments on CPU:

```bash
pip uninstall dgl-cu101
pip install dgl
```

## Running experiments:

You can run the experiments in low-labeling regimes with the standard settings
as below. More examples can be found in the directory run_scripts. Currently,
the citation graph datasets: Cora, Citeseer, Pubmed are supported. Set gpu = -1
to run on cpu.

Here is how to run the GAT-model: ```bash

%Run the GAT model with the following command.

python training/train_ctgr.py --model GAT --data cora --labeling_rate 0.1 --gpu
0 ```

For the SIGN-model, simply change the model flag. ```bash

%Run SIGN model with the following command.

python training/train_ctgr.py --model SparseGAT --data cora --labeling_rate 0.1
--gpu 0 ```

For the LabelPropagationtGAT-model, simply change the model flag. ```bash

%Run LP-GAT model with three steps of label-propagation

python training/train_ctgr.py --model LabelPropagationGAT --data cora
--labeling_rate 0.1 --label_prop_steps 3 --gpu 0 ```

For the LabelPropagationtSIGN-model: ```bash

%Run LP-GAT model with 3 steps of label-propagation

python training/train_ctgr.py --model LabelPropagationSparseGAT --data cora
--labeling_rate 0.1 --label_prop_steps 3 --gpu 0 ```
