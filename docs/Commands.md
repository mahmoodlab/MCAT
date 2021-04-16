Commands for running experiments.
===========
# Training
### Deep Sets
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca_100 --mode path --model_type deepset --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca_100 --mode path --model_type deepset --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg_100 --mode path --model_type deepset --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad_100 --mode path --model_type deepset --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec_100 --mode path --model_type deepset --apply_sigfeats

CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca_100 --mode pathomic --model_type deepset --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca_100 --mode pathomic --model_type deepset --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg_100 --mode pathomic --model_type deepset --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad_100 --mode pathomic --model_type deepset --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec_100 --mode pathomic --model_type deepset --fusion concat --apply_sigfeats

CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca_100 --mode pathomic --model_type deepset --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca_100 --mode pathomic --model_type deepset --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg_100 --mode pathomic --model_type deepset --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad_100 --mode pathomic --model_type deepset --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec_100 --mode pathomic --model_type deepset --fusion bilinear --apply_sigfeats
```

### Attention MIL
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_blca_100 --mode path --model_type amil --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_brca_100 --mode path --model_type amil --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg_100 --mode path --model_type amil --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_luad_100 --mode path --model_type amil --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ucec_100 --mode path --model_type amil --apply_sigfeats

CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_blca_100 --mode pathomic --model_type amil --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_brca_100 --mode pathomic --model_type amil --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg_100 --mode pathomic --model_type amil --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_luad_100 --mode pathomic --model_type amil --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ucec_100 --mode pathomic --model_type amil --fusion concat --apply_sigfeats

CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_blca_100 --mode pathomic --model_type amil --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_brca_100 --mode pathomic --model_type amil --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg_100 --mode pathomic --model_type amil --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_luad_100 --mode pathomic --model_type amil --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ucec_100 --mode pathomic --model_type amil --fusion bilinear --apply_sigfeats
```

### Cluster MI-FCN
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_blca_100 --mode cluster --model_type mifcn --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_brca_100 --mode cluster --model_type mifcn --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg_100 --mode cluster --model_type mifcn --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_luad_100 --mode cluster --model_type mifcn --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ucec_100 --mode cluster --model_type mifcn --apply_sigfeats

CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_blca_100 --mode clusteromic --model_type mifcn --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_brca_100 --mode clusteromic --model_type mifcn --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg_100 --mode clusteromic --model_type mifcn --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_luad_100 --mode clusteromic --model_type mifcn --fusion concat --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ucec_100 --mode clusteromic --model_type mifcn --fusion concat --apply_sigfeats

CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_blca_100 --mode clusteromic --model_type mifcn --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_brca_100 --mode clusteromic --model_type mifcn --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg_100 --mode clusteromic --model_type mifcn --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_luad_100 --mode clusteromic --model_type mifcn --fusion bilinear --apply_sigfeats
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ucec_100 --mode clusteromic --model_type mifcn --fusion bilinear --apply_sigfeats
```

### MCAT
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_blca_100 --mode coattn --model_type mcat --fusion concat --apply_sig
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_brca_100 --mode coattn --model_type mcat --fusion concat --apply_sig
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg_100 --mode coattn --model_type mcat --fusion concat --apply_sig
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_ucec_100 --mode coattn --model_type mcat --fusion concat --apply_sig
CUDA_VISIBLE_DEVICES=1 python main.py --which_splits 5foldcv --split_dir tcga_luad_100 --mode coattn --model_type mcat --fusion concat --apply_sig
```