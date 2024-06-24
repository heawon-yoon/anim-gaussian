
# Animatable Gaussian Avatar

Modeling animatable human avatars from only Few pictures.
colmap dataset is generally reconstructed in real video scenes.
Making data sets in the realm of virtual people is difficult.
This project mainly introduces how to do avatar related 3D reconstruction. 
Useing smpl models and extended lbs for animation

## High quality
It only takes a few minutes to recreate high-quality 3D models
<div>
  <img src="assets/img.png" alt="Image 1" style="display: inline; margin-right: 10px;">
</div>


## Animatable
We show avatars animated by challenging motions from [AMASS](https://amass.is.tue.mpg.de/) dataset.
<div>
  <img src="assets/anim001.gif" alt="Image 1" width="300" height="300" style="display: inline; margin-right: 10px;">
  <img src="assets/anim002.gif" alt="Image 2" width="300" height="300" style="display: inline;">
<img src="assets/canon_001.gif" alt="Image 2" width="300" height="300" style="display: inline;">
</div>


## Installation

```
git clone --recursive https://github.com/heawon-yoon/anim-gaussian.git

cd anim-gaussian

conda create -n anim python=3.8 -y

conda activate anim

#torch
conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

#gaussian
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

#pytorch3d
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html

#requirements
pip install -r requirements.txt
pip install git+https://github.com/mattloper/chumpy.git
```
  If the gaussian module fails to be installed in ubuntu. Take a look at this. This is a some problem I have met
  1. we could not find ninja or g++<br/>
        sudo apt-get update<br/>
        sudo apt install build-essential<br/>
        sudo apt-get install ninja-build

  2. No such file or directory: ‘:/usr/local/cuda-11.8/bin/nvcc.<br/>
     Execute the command directly on the current command line<br/>
        export CUDA_HOME=/usr/local/cuda<br/>
        install again<br/>
        pip install submodules/diff-gaussian-rasterization<br/>
        pip install submodules/simple-knn
   
     window OS and other problem Please refer to this project about gaussians [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)<br/>

  3. pytorch3d problem Please refer to this project about  [pytorch3d](https://github.com/facebookresearch/pytorch3d.git)
# Preparing the datasets and models

## Datasets
- Download the SMPL neutral body model
    - Register to [SMPL](https://smpl.is.tue.mpg.de/index.html) website.
    - Download v1.1.0 file from the [download](https://smpl.is.tue.mpg.de/download.php) page.
    - Extract the files and rename `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` to `SMPL_NEUTRAL.pkl`.
    - Put the files into `./data/smpl/` folder with the following structure:

        ```
        data/smpl/
        ├── SMPL_NEUTRAL.pkl
        ```

- Download  dataset and pretrained models:
     - google link ([download](https://drive.google.com/file/d/1LLfmUnaWQxvge5y-4X51IbKa-ZqdWLIQ/view?usp=sharing))
     - baidu link ([download](https://pan.baidu.com/s/14rvfQQaYHWpoved1cpsd9w?pwd=2tqp))


- Download AMASS dataset for novel animation rendering:
  - AMASS dataset is used for rendering novel poses.
  - We used SFU mocap(SMPL+H G) and MPI_mosh (SMPL+H G) subsets, please download from [AMASS](https://amass.is.tue.mpg.de/download.php).
  - Put the downloaded mocap data in to `./data/` folder.

After following the above steps, you should obtain a folder structure similar to this:

```
data/
├── smpl
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   ├── SMPL_NEUTRAL.pkl
├── humans
│   ├── blender
│   ├── mask
│   ├── cameras.json
│   ├── point_cloud.ply
├── MPI_mosh
│   ├── 00008
│   ├── 00031
│   ├── ...
│   └── 50027
└── SFU
    ├── 0005
    ├── 0007
    ├── ...
    └── 0018
```


# Evaluation and Animation


```
python render_around.py
```

This command will generate 360 degree rotation video and animation video in output folder



# Training

```
python train.py -s data/humans -m ./output/0001

#useing tensorboard to check training result
tensorboard --logdir=./output/0001
```
Open this URL in your browser 
tensorboard_url : http://localhost:6006/




# Custom Dataset

  I used the vroid software to generate the avatar model.
  To generate your own dataset, refer to the blender.py file.
  Open blender app and paste the above code from the script menu.<br/>
  video link : [bilibili](https://www.bilibili.com/video/BV14MgXeyEvn/),[youtube](https://youtu.be/_6nPq05nwOw?si=i-NMwbXrVtkjcCNe)
  1. blender.py will generate Multi-view images and cameras.json file
  
  2. generate mask files Please refer to [SAM](https://github.com/facebookresearch/segment-anything.git),
  sam.py is just simple example code. Please modify the code according to your needs


## Acknowledgements

This project is built on source codes shared by [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [ML-HUGS](https://github.com/apple/ml-hugs.git).

# License
The model is licensed under the [Apache 2.0 license](LICENSE).