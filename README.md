# Degree-Quant

This repo provides a clean re-implementation of the code associated with the paper [Degree-Quant:
Quantization-Aware Training for Graph Neural Networks](https://arxiv.org/abs/2008.05000). At time of writing, only the core method + experiments on Reddit-Binary have been ported over. We may add the remaining experiments at a later date; however, extensive experiment details are supplied in the appendix of the camera ready paper. We do not include the nQAT method in the codebase either for the sake of cleanliness; see the `fairseq` repo if you are interested in implementing this yourself.

This code is useful primarily for downstream users who want to quickly experiment with different quantization methods applied to GNNs. You will most likely be interested in the `dq` folder. For each layer, you can supply a dictionary of functions that when called returns a quantizer; see `reddit_binary/gin.py` for an example. This should enable you to quickly plug-in your own quantization implementations without needing to modify the layers we supplied.

Running the `runall_reddit_binary.sh` script will launch the quantization experiments for reddit binary. We include some output from running our code on reddit binary in the `runs` folder.

## Dependencies

This code has been tested to work with PyTorch 1.7 and Torch Geometric 1.6.3

## Improving this Work

This work is by no means complete. Our study merely identified the issues that will arise when trying to quantize GNNs, and it is likely that you can improve upon our methods in several ways:

1. The runtime of our method is very slow. It is tolerable for the results, but ideally we would make the method faster. We supply a `--sample_prop` flag for the Reddit-Binary experiments that allows you to use sampling on tensors before running the percentile operation. We supply no guarantees on this, but it does seem to offer some improvements to runtime with little noticeable change in accuracy.
2. You may want to consider using learned step sizes -- see the paper on this topic by Esser et al. at ICLR 2020.
3. Robust quantization is another approach that might help -- these works focus on making the network less sensitive to changes in the quantization parameter choices.

We expand on all this in the appendix of the camera ready paper.

## Citing this Work

```
@inproceedings{
tailor2021degreequant,
title={Degree-Quant: Quantization-Aware Training for Graph Neural Networks},
author={Shyam A. Tailor and Javier Fernandez-Marques and Nicholas D. Lane},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=NSBrFgJAHg}
}
```
