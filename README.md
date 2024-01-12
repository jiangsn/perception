# CNNs Prefer Large Training Data Span; Coupling Span and Diversity Improves Robustness


## Setup
1. Clone this repository using `git clone` and change your working directory `cd perception`.
2. Install necessary packages using `conda env create -f environment.yml`, and activate it using `conda activate CP`. We strongly recommand you create a seperate environment to avoid any potential conflicts. If you don't have anaconda installed already, you can download it [here](https://www.anaconda.com/download/).

## Run jobs
Execute `python exp/<EXPERIMENT>.py <BARTYPE> <METHOD> <RUNINDEX> <LEVEL> <MODEL>` to run the experiment ONCE.

### Arguments
- EXPERIMENT: cnn_ratio for ratio sampling, cnn_height for height sampling.
- BARTYPE: Five tyoes of drawing the bar charts. Can be 1, 2, 3, 4, or 5.
- METHOD: The data sampling method, can be IID, COV, ADV, or OOD. You can add _"\_FEATURE"_ suffix to force the sampling method to include the minimum and maximum samples. 
- RUNINDEX: A non-negative integer that can be intepreted as a random seed. This is used to keep the test set the same across the sampling methods.
- LEVEL: Downsampling level, can be 2, 4, 8, or 16. The number of training samples will be reduced to the one half, one quater, one eighth, or one sixteenth of the original.
- MODEL: VGG or ResNet. Choose to whether use VGG19 or ResNet50.

You can find detailed explainations to these arguments in our paper or you can check the code directly.


# Results
The training/validation/test samples, test images and labels, and the model training history of each experiment run are stored in a Python dict and dumpped as a pickle file. You can mearge the results of multiple runs into a single table for fast and structured processing using `jupyter/formatting.ipynb`.

Formatted results of our studies are available 


## Environment
Use `conda env create -f environment.yml` to create environment.

Envrironment name is 'CP', if there exist an environment with same name please change `name:` in `environment.yml`.

And use `conda avtivate CP` to activate environment.

## Run Jobs
Please run `study2_height.py` Use `python study2_height.py TYPE SPLIT SEED GPU_ID` (run one job).

We have 300 jobs in total for this sub-study, please consider using batch script (see **Batch Script** below)

### Arguments
`TYPE`: \['type1', 'type2', 'type3', 'type4', 'type5'\]

`SPLIT`: \['All', 'HalfRandom', 'HalfMin', 'HalfMax', 'HalfAdversarial', 'HalfCoverage'\], six different sampling methods

`SEED`: In order to make different sampling methods comparable, we let each run share the same random seed amoung different sampling methods. E.g. `SEED` of type2, third run is 23, regardless the `SPLIT`.

`GPU_ID`: An optional parameter that specify which GPU is used for running this job.

### Batch Script
Please refer to the comments in `scheduler.py`

Basic idea: scheduler will continue querying if there is a CNN perception job running on each GPU. If a GPU is available it will start a thread to run a job on that GPU.

### Output Path
Please change the prefix in `study2_height.py` (line 52)

## Results
Although we saved weight of each run, we don't need them for now. So if you're running out of storage please feel free to remove it.

And if it is possible, please upload results (".p" files) to this repo.
