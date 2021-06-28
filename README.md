# TCIR-Benchmark

This repository is the reproduce version by Boyo Chen of paper:
### Benchmarking Tropical Cyclone Rapid Intensification with Satellite Images and Attention-based Deep Models

## Requirements

To install requirements:

0. install pipenv (if you don't have it installed yet)
```setup
pip install pipenv
```
1. use pipenv to install dependencies:
```
pipenv install
```
2. install tensorflow **in the** pipenv shell
(choose compatible tensorflow version according to your cuda/cudnn version)
```
pipenv run pip install tensorflow
pipenv run pip install tensorflow_addons
```

## Training

To run the experiments, run this command:

```train
pipenv run python main.py <experiment_path>

<experiment_path>:

# ordinary ConvLSTM
experiments/baseline.yml
```

***Notice that on the very first execution, it will download and extract the dataset before saving it into a folder "TCSA_data/".
This demands approximately 20GB space on disk***

### Some usful aguments

#### To limit GPU usage
Add *GPU_limit* argument, for example:
```args
pipenv run python train main.py <experiment_path> --GPU_limit 3000
```

#### To set CUDA_VISIBLE_DEVICE
Add *-d* argument, for example:
```args
pipenv run python train main.py <experiment_path> -d 0
```

## Evaluation

All the experiments are evaluated automaticly by tensorboard and recorded in the folder "logs".
To check the result:

```eval
pipenv run tensorboard --logdir logs

# If you're running this on somewhat like a workstation, you could bind port like this:
pipenv run tensorboard --logdir logs --port=1234 --bind_all
```
