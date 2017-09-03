# Tacotron_pytorch
Pytorch implementation of Tacotron: A Fully End-to-End Text-To-Speech Synthesis Model

https://arxiv.org/abs/1703.10135

## Requirements
  * pytorch
  * librosa
  * py-webrtcvad

## Data
Please register to use Blizzard Challenge data set. (http://www.cstr.ed.ac.uk/projects/blizzard/)

In the code, the option 'blizzard' is for the Blizzard Challenge data of 2013.

The option 'etri' is for Korean TTS dataset published by 'ETRI' and you need to buy license to use it.

You need to download and unzip the data from the website.

Then, set paths in the codes (train.py, preprocess.py, generate.py) accordingly. (find 'dir_' and change the following lines)

## How to run
  * Please refer the code to see what options/hyperparameters are available
1. Prepare data and preprocess the data (ex. blizzard) by running: preprocess.py --data 'blizzard'
(You may want to trim silences in audio files before preprocessing. Please use trimmer.py)
2. Run 'train.py' with arguments.
3. After training, run 'generate.py' with arguments to get generated audio file.


## Comment
Contributions and comments are always welcome.
I refered https://github.com/keithito/tacotron for the preprocessing code. Thank you.
