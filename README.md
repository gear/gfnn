# Revisiting Graph Neural Networks: All We Have is Low-Pass Filters
In this work, we study graph neural networks for vertex classification. This work is conducted at [RIKEN AIP](https://aip.riken.jp/). [Our preprint](https://arxiv.org/abs/1905.09550) is uploaded on arXiv.

## Requirements

Create an Python 3.6 environment and install these packages:

```python
numpy
scipy
networkx==1.11
scikit-learn
pytorch
torchvision
```
Or install from `requirements.txt`.

Move files in `data/*` to your `~/data/` folder, or change the paths in `utils.py` for Cora, Citeseer, Pubmed. The data files can be found [here (Gdrive)](https://drive.google.com/open?id=1ruxemElzwiXXErR7c7DE8BhRtZ7cxnP4).

For Reddit, download [reddit.adj (Gdrive)](https://drive.google.com/file/d/174vb0Ws7Vxk_QTUtxqTgDHSQ4El4qDHt/view) and [reddit.npz (Gdrive)](https://drive.google.com/file/d/19SphVl_Oe8SJ1r87Hr5a6znx3nJu1F2J/view) and put them under `~/data/reddit/`, or change paths in `utils.py`.

We will pack data for PPI in the later version.

## Experiments

Check `args.py` file to change options for noise and other filters.

To run with Cora, Citeseer, Pubmed:
```python
python citation.py --no-cuda --model gfNN --dataset cora
python citation.py --no-cuda --model SGC --dataset cora
python citation.py --no-cuda --model GCN --dataset cora
```

To run with synthetic dataset (4000 data points):
```python
python synthetic.py --no-cuda --model gfNN
python synthetic.py --no-cuda --model SGC
python synthetic.py --no-cuda --model GCN
```

The frequency experiment (Figure 3) can be found in `transformation.py` and `raw_feats_citation.py`.

## Acknowledgement
This reposipory is built upon SGC, FastGCN, and GCN. I would like to say thanks
to the authors of these repository for making their code available.
