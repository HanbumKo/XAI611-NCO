

# XAI611_NCO_generalization

Lecture XAI611 proejct repo.

Related topics: Neural combinatorial optimization; Maximum cut; Graph neural netowrks; Generalization

## Introduction

Many discrete optimization problems in the real world, such as last-mile delivery route optimization and chip design, can be expressed as graph combinatorial optimization (CO) problems. These problems are often classified as NP-complete, lacking a deterministic algorithm, and their complexity grows exponentially as the problem size increases. Especially, Karp's 21 NP-complete problems has attracted attention due to its reducibility between the CO problem and its computational intractability. In recent years, advancements in deep learning have demonstrated the potential to find near-optimal solutions within a short time while minimizing computational complexity.

But there are many obstacles to apply the model to deal with real-world problems. One of the problem is generalization. In particular, the number of nodes and their structure (how the nodes are connected) vary depending on the graph. It is still an open question which graph neural network structure has good generalization performance. Although many training and inference methods have been suggested to tackle the generalization problems, the GNN architecture remain old-fashioned. I would like to find the better GNN architecture that can be generalizable between graphs so that the trained GNN model can solve real-world problems represented in graphs.


### Problem: Maximum cut
Here, we focus on maximum cut (MC) problem, which is one of representative graph CO problems, maximizing size of cut-set. Cut-set is a set of edges connecting two complementary vertices sets S and T. Maximum cut problem is a problem finding a cut-set having maximum summation edge weights in the cut-set. Please refer following figure.

![maximum_cut](https://github.com/HanbumKo/XAI606_NCO_generalization/blob/main/images/maximum_cut.png?raw=true)

## Dataset
The experiment dataset consists of relatively small-sized graphs. I have made preprocessed graphs and its node/edge feature is saved via .pkl files. There are 50,000 training graphs and four different 100 validation/test graphs.

|          	| # of nodes 	| # of edges 	| # of data 	|
|----------	|------------	|------------	|-----------	|
| training 	| 20         	| small      	| 50,000    	|
| val1     	| 20         	| small      	| 100       	|
| val2     	| 50         	| small      	| 100       	|
| val3     	| 20         	| medium     	| 100       	|
| val4     	| 50         	| medium     	| 100       	|
| test1    	| 20         	| small      	| 100       	|
| test2    	| 50         	| small      	| 100       	|
| test3    	| 20         	| medium     	| 100       	|
| test4    	| 50         	| medium     	| 100       	|

Download the preprocessed datasets via following Google Drive, and unzip it. https://drive.google.com/file/d/1Wlemq8JSATaLnujODvBzczYNpN-hGlkz/view?usp=sharing

    tar -xzvf datasets.tar.gz

And you can see 'instance_{}.pkl' files in each directory. A .pkl file consists of ([NetworkX](https://networkx.org/documentation/stable/index.html) graph, [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) graph, label) to deal with graphs and its label. NetworkX graph class gives useful functions related to graph, and PyTorch Geometric graph is used for input of the GNN model.

You can import the data via following code.
``` python
import pickle

instance_path = './datasets_xai606/train/instance_0.pkl'
with  open(instance_path, 'rb') as  inp:
    networkx_graph, torch_graph, label = pickle.load(inp)
```


## Metric

Average binary cross entropy loss value and objective value made by trained GNN is measured. The metrics are calculated by your submitted selected nodes of all graphs in four test dataset.

|       	| Avg. loss 	| Avg. objective 	|
|-------	|-----------	|----------------	|
| test1 	| x         	| x              	|
| test2 	| x         	| x              	|
| test3 	| x         	| x              	|
| test4 	| x         	| x              	|



## Example training code
I share minimal train/validation code using simple graph convolutional network as an example.

    python train.py



## Requirements
Specific version is not needed. Please install missing library via pip. Major libraries are

    torch_geometric
    torch
    pickle
    networkx


## Contact

hanbumko95@korea.ac.kr
