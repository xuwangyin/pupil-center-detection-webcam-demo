# pupil-center-detection-webcam-demo

To run the demo, first install the required packages:
```
conda create -n detection-demo python=3.7
conda activate detection-demo
conda install -c conda-forge opencv
conda install -c conda-forge dlib
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
Then run the main script:
`python main.py`
