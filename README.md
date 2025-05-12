# OpenCap Core
This code takes two or more videos and estimates 3D marker positions and human movement kinematics (joint angles) in an OpenSim format. Kinetics (forces) can then be calculated using these outputs using the [opencap-processing](https://github.com/stanfordnmbl/opencap-processing) repository. Learn more about data collection at [opencap.ai](https://opencap.ai). There are three possible ways to use this code:
1) Collect data and have it automatically processed using our web application ([app.opencap.ai](https://app.opencap.ai)) using iOS devices. We are running the pipeline in the cloud; this service is freely available for academic research use. Visit [opencap.ai/get-started](https://opencap.ai/get-started) to start collecting data. See an example session [here](https://app.opencap.ai/session/7272a71a-e70a-4794-a253-39e11cb7542c).
2) Run this pipeline locally using videos recorded using [app.opencap.ai](https://app.opencap.ai). Results can be viewed locally, and they will also be updated in the cloud database, so they can be visualized at [app.opencap.ai](https://app.opencap.ai). This is useful for customizing the pipeline, for reprocessing data using high accuracy pose estimation settings, or for debugging.
3) Run this pipeline locally using videos collected near-synchronously from another source (e.g., videos collected synchronously with marker-based motion capture). Easy-to-use utilities for this pipeline are under development and will be released soon.

<p align="center">
  <img src="media/cut_fastAndSlow.gif">
</p>


## Publication
More information is available in our [preprint](https://www.biorxiv.org/content/10.1101/2022.07.07.499061v1): <br> <br> 
Uhlrich SD*, Falisse A*, Kidzinski L*, Ko M, Chaudhari AS, Hicks JL, Delp SL, 2022. OpenCap: 3D human movement dynamics from smartphone videos. _biorxiv_. https://doi.org/10.1101/2022.07.07.499061. *contributed equally <br> <br> 
Archived code base accompanying the paper: [https://doi.org/10.5281/zenodo.7419967](https://doi.org/10.5281/zenodo.7419967).

## Running the pipeline locally
### Hardware and OS requirements:
These instructions are for Windows 10. The pipeline also runs on Ubuntu. Minimum GPU requirements: CUDA-enabled GPU with at least 4GB memory. Not all of the OpenPose settings will run on small GPUs. To run the OpenPose settings we use in the cloud pipeline, you need a GPU with 8GB of memory. To run the high resolution settings, you need a GPU with at least 24GB memory. For local postprocessing, we use NVIDIA GeForce RTX 3090s (24GB).

### Installation
1. Install [Anaconda](https://www.anaconda.com/).
2. Fork and clone the repository to your machine.
3. Open the Anaconda command prompt and create a conda environment: `conda create -n opencap python=3.9 pip spyder`.
4. Activate the environment: `conda activate opencap`.
5. Install OpenSim: `conda install -c opensim-org opensim=4.4=py39np120`. Visit this [webpage](https://simtk-confluence.stanford.edu:8443/display/OpenSim/Conda+Package) for more details about the OpenSim conda package. 
6. Install Visual Studio Community 2022 from [here](https://visualstudio.microsoft.com/vs/community/). During installation, select "Desktop development with C++". 
7. For GPU support of tensorflow, install the [NVIDIA driver](https://www.nvidia.com/download/index.aspx?lang=en-us) for your GPU. Then in the anaconda prompt, install CUDA and cudnn: `conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`. [More information about setting up GPU support for tensorflow](https://www.tensorflow.org/install/pip).
8. Install other dependencies. Make sure you have navigated to the local directory where the repository is cloned, then: `python -m pip install -r requirements.txt`.
9. Copy ffmpeg and openpose builds found in this [Google Drive dependencies folder](https://drive.google.com/drive/folders/17ihUjaKsc8vwzOuzKWIMndNz_Z7Odm4N?usp=sharing) to the C drive: put them into `C:\ffmpeg` and `C:\openpose` such that the binary folders are `C:\ffmpeg\bin` and `C:\openpose\bin`. The up-to-date versions can also be used ([OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases), [ffmpeg](https://www.gyan.dev/ffmpeg/builds/)), but we recommend the provided versions from the Google Drive folder, since they have been tested with the pipeline.
10. Add ffmpeg to the PATH Environment Variable: press Windows key, type environment variables, click Environment Variables. In System variables, click Path, and add `C:\ffmpeg\bin`. 

### Running the pipeline using data collected at app.opencap.ai

1) Authenticate and save an environment variable by running `Examples/createAuthenticationEnvFile.py`. You can proceed without this, but you will be required to log in every time you run a script.
2) Copy your session identifier from [app.opencap.ai](https://app.opencap.ai) into `Examples/reprocessSession.py`, choose your pose estimation settings, and run it. The session id is the 36-character string at the end of the session url. For example, the session identifier for https://app.opencap.ai/session/7272a71a-e70a-4794-a253-39e11cb7542c is `'7272a71a-e70a-4794-a253-39e11cb7542c'`. If you reprocess a session that you recorded, results will be written back to the database, and if you choose, they will be saved locally in `./Data/<session_id>`.
3) To compute kinetics we recommend starting with `example_kinetics.py` in the [opencap-processing](https://github.com/stanfordnmbl/opencap-processing) repository. Data from many sessions can also be downloaded in batch using `batchDownload.py` in the opencap-processing repository or the `Examples/batchDownloadData.py` script in this repository.
4) If you are using HRNet of mmpose, you need to install the following version:
```bash
openmim                   0.3.9 
mmcv                      1.7.0                    
mmcv-full                 1.7.0                    
mmdet                     2.28.2                   
mmengine                  0.10.7                   
mmpose                    0.29.0
torch                     1.13.1+cu117
torchvision               0.14.1+cu117       
```
5) You also may need to download the detection and keypoints models from:
```bash
https://huggingface.co/spaces/fffiloni/mmpose-estimation/blob/dd314fcab24437b8d2847cda867fcd46036ec91f/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_256x192_dark-469327ef_20200922.pth
```
#### Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR  | Hand AP | Hand AR | Whole AP | Whole AR | ckpt | log |
| :---- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----: | :-----: | :------: |:-------: |:------: | :------: |
| [pose_hrnet_w32_dark](/configs/wholebody/darkpose/coco-wholebody/hrnet_w32_coco_wholebody_256x192_dark.py)  | 256x192 | 0.694 | 0.764 | 0.565 | 0.674 | 0.736 | 0.808 | 0.503 | 0.602 | 0.582 | 0.671 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_256x192_dark-469327ef_20200922.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_wholebody_256x192_dark_20200922.log.json) |
| [pose_hrnet_w48_dark+](/configs/wholebody/darkpose/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py)  | 384x288 | 0.742 | 0.807 | 0.705 | 0.804 | 0.840 | 0.892 | 0.602 | 0.694 | 0.661 | 0.743 | [ckpt](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth) | [log](https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark_20200918.log.json) |

Note: `+` means the model is first pre-trained on original COCO dataset, and then fine-tuned on COCO-WholeBody dataset. We find this will lead to better performance.

6) You may need to change model_ckpt_person and model_ckpt_pose paths in utilsDetector.py. 

### Reproducing results from the paper 
1) Data used in the OpenCap publication are available on [SimTK](https://simtk.org/projects/opencap). This dataset includes raw data (e.g., videos, motion capture, ground reaction forces, electromyography), and processed data (e.g., scaled OpenSim models, inverse kinematics, inverse dynamics, and dynamic simulation results).
2) The scripts to process and plot the results are found in the `ReproducePaperResults` directory (see README.md in this folder for more details).
