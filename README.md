## Structure of the repo:

- **data**
    
    Datasets and all data-related stuff is here 
    (could be symbolic link to the real data location)

- **experiments** 

    Folder contains 
    * [cfgs] param (config) files for experiments;
    * [logs] all intermediate results from different experiments (param files, checkpoints, logs, ...);
    * [scripts] scripts to run specific experiments; 
                
- **notebooks**

    Place for jupyter notebooks.
    Notebooks could contain some analysis
    (dataset analysis, evalution results), demo, some ongoing work

- **material**

    Data, which include additional images, results, model weights, ...

    * [results] useful results are stored here;
    * [images] images for demo, readme, ...;
    * [weights] weights of the models (e.g. pretrained backbones ) 
    
- **src**

    Codebase.

## Requirements

Tested with

* python 3.6
* pytorch 1.1+
* tensorboard from (tf-nightly-2.0-preview)
        
## How to use

* load the data

(cifar10) go to the folder 'data' and run [YOU NEED ~ 300 MB]
> ./get_cifar10_dataset.sh

(dtd) go to the folder 'data' and run
> ./get_cifar10_dataset.sh
>
* 'train'

Run from the $PROJECT$ directory:

(cifar)
> python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml

(dtd)
> python src/main_train.py --param_file experiments/cfgs/params_dtd.yaml




