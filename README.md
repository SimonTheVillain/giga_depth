# GigaDepth
Simon Schreiberhuber, Jean-Baptiste Weibel, Timothy Patten, Markus Vincze
Paper (TODO: LINK), Dataset (TODO: LINK)

## Requirements
Our algorithm relies on custom CUDA kernels and thus requires an **NVIDIA GPU**. 
Tensor cores are not utilized, as they are not as beneficial to our regression tree as they are for the CNN backbone.
We tested this work on a RTX 3090 and a RTX 2070 and expect it to work on older as well as newer cards.

The libraries used are:

`numpy, OpenCV, cuda (11), pytorch (1.8), ninja, tensorboard, scipy, matplotlib`

## Dataset(s)
Download the datasets at this link (TODO)
## Trained Model
Download the pretrained models at this link (TODO)

## Usage
### Training
We offer a few configuration files for training in our `configs` folder
- `lcn_4jitter_1280classes.yaml` using Local Contrast Normalization for the input, 
jittering the training data by 4 pixel vertically and splitting into 1280 classes per line. 
- `lcn_4jitter_1920classes.yaml` same but with 1920 classes
- `lcn_4jitter_1280classes_8gb.yaml` here the backbone, MLP tree and regressor are reduced such that a 8GB GPU 
can handle it.

Training then is started by giving the config file and the folder to the dataset
```
python3 train.py --config_file=configs/lcn_4jitter_1280classes_8gb.yaml --dataset_path=synthetic_sequences_train
```
The path to the batch size can be overwritten to fit the GPU memory size by `-b=1`. Specifically a 24GB GPU as the 
RTX 3090 should be able to handle a batch size of at least 4 if not 8 and train within 12 hours.

### Run
To run the script on a dataset, either synthetic or real call and output color coded images in the output folder.
```
python3 run.py --model_path= --input_path= --output_path
```




## Shortcomings
Aside of the sortcomings mentioned in the paper (e.g. the requirement for full supervision, no domain adaptation), 
we hereby admit to a few more issues.
- The network has to be retrained for each sensor as long rectification cannot compensate for all production tolerances.
- The code quality of this repo is very poor. If you miss functionality for evaluation, 
take a look at the other branches. Other baseline methods can be found as forks on my github account.
- The CUDA implementation of the MLP tree uses one kernel call per MLP layer. 
One could combine these calls into one big kernel that keeps the intermediate results in shared memory/L1 cache 
of the GPU cores. This would vastly speed up computation similar to [tini-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
- The capture of the pattern is flawed as the wall we used to project the pattern might not
be perfectly flat and the factory sensor calibration being flawed.
##  Citation

If you find the code, data, or the models useful, please cite this paper:
```
 @InProceedings{Schreiberhuber_2022_CVPR,
    author    = {Schreiberhuber, Weibel, Patten, Vincze},
    title     = {GigaDepth: Learning Depth from Structured Light with Branching Neural Networks},
    booktitle = {Proceedings of the European Converence on Computer Vision (ECCV)},
    month     = {Oktober},
    year      = {2022},
    pages     = {}
}
```
## Acknowledgements 
The research leading to these results has received funding from EC Horizon 2020 for Research and Innovation under grant agreement No. 101017089, TraceBot and the Austrian Science Foundation (FWF) under grant agreement No. I3969-N30, InDex.


## Licence
``` 
 [Creative Commons Attribution Non-commercial No Derivatives](http://creativecommons.org/licenses/by-nc-nd/3.0/)
```