# GigaDepth
Simon Schreiberhuber, Jean-Baptiste Weibel, Timothy Patten, Markus Vincze
[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930209.pdf) [Supplementary](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930209-supp.pdf), [Dataset/Trained Models](https://doi.org/10.48436/q76vf-y9t57)

## Requirements
Our algorithm relies on custom CUDA kernels and thus requires an **NVIDIA GPU**. 
Tensor cores are not utilized, as they are not as beneficial to our regression tree as they are for the CNN backbone.
We tested this work on a Ubuntu 20.04 machine with a RTX 3090 and a RTX 2070 and expect it to work on older as well as newer cards.

The libraries used are:
```
numpy, OpenCV, cuda (11), pytorch (1.8), ninja, tensorboard, scipy, matplotlib
```
Feel free to contact [Simon Schreiberhuber](simon.schreiberhuber@gmx.net) if you find this list not to be complete.

## Dataset(s)
Download the datasets at the [link](https://doi.org/10.48436/q76vf-y9t57) from above.
There are two versions of the training set. One where the input images are compressed as (lossy) `jpg` files, 
one with `png` encoding. Both variants contain the following folders:
- **synthetic_train**: Sequences of 4 frames each rendered with Unity3D. Randomly selected objects with randomly 
sampled size and textures are randomly placed around a camera to generate an extremely diverse dataset.
- **synthetic_test**: Single frames rendered with Unity3D. Scenes and objects are 100% different from the training data.
- **patterns_capture**: Several images captured by the Occipital Structure Core
- **captured_train**: Captured data that can be used for training of methods as "DepthInSpace", "ActiveStereoNet", 
"ConnectingTheDots" or "HyperDepth". 241 of the 967 sequences also capture frames with deactivated dot-projector. 
In "ConnectingTheDots" terminology this would be equivalent the "ambient" images.
- **captured_test_planes**: Captured ceiling. We initially had an evaluation based on planes fitted on this data, 
but as some baseline algorithms performed so abysmally, we excluded it.
- **captured_test**: 11 Sequences captured by the Structure Core with baseline pointcluds captured by a
Photoneo MotionCam-3D M in scanning mode.
- - **captured_test_unused**: Three additional scenes that we didn't include in our testing. 
They were simply captured after performing our evaluation, we did not exclude them to tweak our results.


We believe that the `jpg` variant of the dataset with 190GB is sufficient for training and won't lead to a reduced performance due 
to compression artifacts. If you still feel that the compression is creating some form of bias, we suggest to load the 
bigger `png` dataset at around 500GB.
## Trained Model
Download the pretrained models at this [link](https://doi.org/10.48436/q76vf-y9t57) from above.
We provide two models:
- `jitter4_1280classes.pt` trained on synthetic data according to `lcn_4jitter_1280classes.yaml`
- `jitter4_1920classes.pt` trained on synthetic data according to `lcn_4jitter_1920classes.yaml`

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
python3 train.py --config_file=configs/lcn_4jitter_1280classes_8gb.yaml --dataset_path=synthetic_train
```
The path to the batch size can be overwritten to fit the GPU memory size by `-b=1`. Specifically a 24GB GPU as the 
RTX 3090 should be able to handle a batch size of at least 4 if not 8 and train within 12 hours.

### Run
To run the script on a dataset, either synthetic or real call and output color coded images in the output folder.
```
python3 run.py --model_path=trained_models/jitter4_1280classes.pt --input_path=.../synthetic_test --output_path=.../synthetic_test_results
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
