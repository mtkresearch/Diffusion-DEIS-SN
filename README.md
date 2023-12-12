# Diffusion-DEIS-SN
Code repository for the paper titled "_[Score Normalization for a Faster Diffusion Exponential Integrator Sampler](https://arxiv.org/abs/2311.00157)_" from [MediaTek Research UK](https://i.mediatek.com/mediatekresearch) accepted at [NeurIPS 23 Diffusion workshop](https://openreview.net/group?id=NeurIPS.cc/2023/Workshop/Diffusion).

## Running the code

To run the code and reproduce experiments of the paper, you need to train a model and execute inference on it. As in the paper, the code allows testing on two datasets, CIFAR10-32 and LSUN-Church-64. We provide the cached "average" of the score estimates for these two datasets.

### Training

Training can be done by simply invoking the `uncond.py` file. You can use multi-GPU if you'd like

```
torchrun --nnodes 1 --nproc_per_node <gpus> --no_python python3 uncond.py --config ./deis_<dataset>.yml
```

The `<dataset>` can be one of ['cifar', 'church']. The config file contain all necessary hyperparameters needed. The options in the `.yml` file marked with `# ...` must be filled by yourself. You can also try changing the `--inference_num_steps xx` for experimentation. The training code computes FID in an on-the-fly manner. It samples every `saving_epochs` epochs, a batch of `num_samples` samples and compares against the reference batch provided in `reference_batch_path` argument. We showed how to create reference batch [in an older codebase](https://github.com/mtkresearch/shortest-path-diffusion#reference-batch-for-fid).

### Sampling

While training already computes FID, you may explicitly run sampling by some extra arguments

```
torchrun --nnodes 1 --nproc_per_node <gpus> --no_python python3 uncond.py --config ./deis_<dataset>.yml --infer <sampler> --inference_num_steps 10 --sample_with pipeline-xxx <infer_tag>
```

By changing the `--infer`, you can alter the sampling algorithm. Possible values are ['ddim', 'deisabode', 'deissnode'], of which, first one is standard DDIM, second one being original DEIS (Adam-Bashforth) and the third one is ours. The `--sample_with` argument takes two arguments, first one being the exact saved pipeline to use (HF pipelines will be saved during training) and the second one being a sampling specific tag.

## Notes

1. This repo is a subset of a larger repo. Please expect codes that are not part of the paper.
2. We provide the pre-computed cache stats for two datasets as `cifar10_Ls_score_abs.pt` and `lsunchurch_Ls_score_abs.pt`. You may compute this on your own with any saved model (pipeline) by [uncommenting these lines](https://github.com/mtkresearch/Diffusion-DEIS-SN/blob/8fd766a3a3ea603aaaac5a2583922508a70a332a/core/pipelines/deisabode_pipeline.py#L112) while running an inference with `--infer deisabode --timestep_spacing trailing --inference_num_steps 1000`.
3. Please also create a virtual env following the `requirements.txt`.

## Citation

Please cite this work using the following bib entry

```
@inproceedings{
xia2023score,
title={Score Normalization for a Faster Diffusion Exponential Integrator Sampler ({DEIS})},
author={Guoxuan Xia and
        Duolikun Danier and
        Ayan Das and
        Stathi Fotiadis and
        Farhang Nabiei and
        Ushnish Sengupta and
        Alberto Bernacchia},
booktitle={NeurIPS 2023 Workshop on Diffusion Models},
year={2023},
url={https://openreview.net/forum?id=AQvPfN33g9}
}
```
