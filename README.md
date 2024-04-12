# flox-synthetic-tests
This repository presents the source code for the synthetic tests for the following paper:
> John Smith, et al. "<title goes here>." Some Venue (YYYY).

## General Setup

### Python
First, you must setup your Python environment (either `venv` or `conda`). These tests were run with Python 3.11.8 specifically.
Next, you must install the dependencies from the in your environment using `requirements.txt` (pip) or `env.yml` (conda).

### Data Download
For any of the model training in these tests, we use the Fashion MNIST benchmark dataset. To use this dataset for these tests,
you must first download the data onto your machine in a directory of your choosing (just be sure to take note of where you save it).
This can be done by running the provided Python script:
```sh
$ python download_data.py --root .
```
This will download the dataset using `torchvision.datasets`. 

## Experiments
In the `experiments/` directory, we provide the code for executing and plotting the results for each of the synthetic tests from the paper.


### Hierarchical Topology Test (Artifact 2)
...

### Asynchronous Comparison Test (Artifact 3)
...

### Remote Execution Test (Artifact 4)
...

## Other
If you are looking for **Scaling Tests (Artifact 1)** click [here](https://github.com/h-flox/flox-scaling-tests).
