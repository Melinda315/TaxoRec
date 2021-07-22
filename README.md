# TaxoRec
This repository contains a implementation of our "Enhancing Recommendation with Automated TagTaxonomy Construction in Hyperbolic Space".

## Environment Setup
1. Pytorch 1.8.1
2. Python 3.7.3

## Guideline

### data

We provide one dataset, ciao.

```adj_csr.npz``` adj matrix built for training gcn 
```item_tag_matrix.npz``` items attributes matrix
```tag_map.json``` tag idx to tag name mapping.
```train.pkl``` train set
```test.pkl``` test set
```user_item_list.pkl``` user-item dict for the complete dataset.

### models

The implementation of model(```model.py```); 

code to implement Hyperbolic gcn (```encoders.py, hyp_layers.py```)

### utils

```data_generator.py``` read and organize data
```helper.py``` some method for helping preprocess data or set seeds and devices
```sampler.py``` a parallel sampler to sample batches for training
```taxogen.py``` build taxonomy
```train_utils.py``` read and parse the config arguments

## Example to run the codes

```
python run.py
```
