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


## Results
The training/validation/test samples, test images and labels, and the model training history of each experiment run are stored in a Python dict and dumpped as a pickle file. You can mearge the results of multiple runs into a single table for fast and structured processing using `jupyter/formatting.ipynb`.

Formatted results of our studies are available in `results/formatted_data/`

- `*.feather`: Predictions and ground truths store in [feather formate](https://arrow.apache.org/docs/python/feather.html). 
- `*.history.feather`: Training histories.
- `*.p`: Configuration metadata in [pickle format](https://docs.python.org/3/library/pickle.html).


## Figures
Check the scipt `jupyter/draw_figures.ipynb`, it generages all figures we used in our paper.

## Statistic Analysis
