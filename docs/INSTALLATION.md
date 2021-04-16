For instructions on installing anaconda on your machine (download the distribution that comes with python 3):
https://www.anaconda.com/distribution/

After setting up anaconda, first install openslide:
```shell
sudo apt-get install openslide-tools
```

Next, use the environment configuration file located in **docs/mcat.yaml** to create a conda environment:
```shell
conda env create -n mcat -f docs/requirements.yaml
```

Activate the environment:
```shell
conda activate MCAT
```
