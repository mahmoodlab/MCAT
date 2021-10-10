Commands for Running Ablation Experiments.
===========
### Deep Sets
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca --mode path --model_type deepset --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca --mode path --model_type deepset --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode path --model_type deepset --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad --mode path --model_type deepset --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode path --model_type deepset --apply_sigfeats

CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca --mode pathomic --model_type deepset --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca --mode pathomic --model_type deepset --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode pathomic --model_type deepset --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad --mode pathomic --model_type deepset --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode pathomic --model_type deepset --fusion concat --apply_sigfeats

CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca --mode pathomic --model_type deepset --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca --mode pathomic --model_type deepset --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode pathomic --model_type deepset --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad --mode pathomic --model_type deepset --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode pathomic --model_type deepset --fusion bilinear --apply_sigfeats
```

### Attention MIL
```bash
CUDA_VISIBLE_DEVICES=2 python main.py --which_splits 5foldcv --split_dir tcga_blca --mode path --model_type amil --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_brca --mode path --model_type amil --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode path --model_type amil --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_luad --mode path --model_type amil --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode path --model_type amil --apply_sigfeats

CUDA_VISIBLE_DEVICES=2 python main.py --which_splits 5foldcv --split_dir tcga_blca --mode pathomic --model_type amil --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_brca --mode pathomic --model_type amil --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode pathomic --model_type amil --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_luad --mode pathomic --model_type amil --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode pathomic --model_type amil --fusion concat --apply_sigfeats

CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_blca --mode pathomic --model_type amil --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_brca --mode pathomic --model_type amil --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode pathomic --model_type amil --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_luad --mode pathomic --model_type amil --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode pathomic --model_type amil --fusion bilinear --apply_sigfeats
```

### Cluster MI-FCN
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_blca --mode cluster --model_type mi_fcn --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_brca --mode cluster --model_type mi_fcn --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode cluster --model_type mi_fcn --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_luad --mode cluster --model_type mi_fcn --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode cluster --model_type mi_fcn --apply_sigfeats

CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_blca --mode clusteromic --model_type mi_fcn --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_brca --mode clusteromic --model_type mi_fcn --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode clusteromic --model_type mi_fcn --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_luad --mode clusteromic --model_type mi_fcn --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode clusteromic --model_type mi_fcn --fusion concat --apply_sigfeats

CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_blca --mode clusteromic --model_type mi_fcn --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_brca --mode clusteromic --model_type mi_fcn --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode clusteromic --model_type mi_fcn --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_luad --mode clusteromic --model_type mi_fcn --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode clusteromic --model_type mi_fcn --fusion bilinear --apply_sigfeats
```

### MCAT
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_blca --mode coattn --model_type mcat --fusion concat --apply_sig
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_brca --mode coattn --model_type mcat --fusion concat --apply_sig
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode coattn --model_type mcat --fusion concat --apply_sig
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode coattn --model_type mcat --fusion concat --apply_sig
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_luad --mode coattn --model_type mcat --fusion concat --apply_sig
```