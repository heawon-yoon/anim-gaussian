
# Animatable Gaussian Avatar

Modeling animatable human avatars from only Few pictures.
gaussian is generally reconstructed in real video scenes.
Making data sets in the realm of virtual people is difficult.
This project mainly introduces how to do avatar related 3D reconstruction. Useing smpl models for animation

## High quality


1. Audio file UVR voice separation

2. Vocal noise reduction and enhancement

3. AutoSlice voice intelligent AI slicing by segment

4. ASR automatic vocal recognition and lyrics conversion

5. After generating lyrics, you can manually modify the lyrics

6. MFA lyrics audio automatic forced alignment

7. Data preprocessing

8. Start infer

9. Recombine vocals and accompaniment to generate final audio

## Installation

```
git clone https://github.com/hunkunai/music.git

cd music


conda create -n music python=3.8 -y

conda activate music


pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


pip install -r requirements.txt

conda install -c conda-forge montreal-forced-aligner==2.0.6 --yes

pip install paddlepaddle==2.4.2

pip install setuptools-scm

pip install pytest-runner

pip install paddlespeech==1.4.1


```


## Download Models

1.Download checkpoint 

final path like checkpoint/nsf_hifigan,checkpoint/my_experiment
```
  cd checkpoint/
  wget https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip
  unzip nsf_hifigan_20221211.zip
  wget https://github.com/hunkunai/music/releases/download/music/my_experiment.zip
  unzip my_experiment.zip

```

2.Download uvr model 


final path like assets/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth

```
  mkdir assets/uvr5_weights
  cd assets/uvr5_weights/

  wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth

```

3.Download mfa model 

final path like assets/uvr5_weights/mfa-opencpop-extension.zip
not to unzip that!

```

  cd assets/
  wget https://huggingface.co/datasets/fox7005/tool/resolve/main/mfa-opencpop-extension.zip

```


## User Guide

demos on [bilibili](https://www.bilibili.com/video/BV1wG41197K4/)   [bilibili](https://www.bilibili.com/video/BV1bN41137UA/?vd_source=5afbd824d0483e6ab60779ed3faa4535)


##### If you encounter a "module not found" error during startup or runtime, please reinstall the module according to the version specified in the requirements.txt file.

Step 1:<br/>
    start gradio ui

```
python app.py

```

Step 2:<br/>
    upload audio file and click button
    <div>
      <img alt="" src="https://github.com/hunkunai/music/blob/main/WechatIMG543.jpeg" width="600" height="400" />
    <div/>



Step 3:<br/>
    click infer button
    <div>
      <img alt="" src="https://github.com/hunkunai/music/blob/main/WechatIMG544.jpeg" width="600" height="400" />
    <div/>







