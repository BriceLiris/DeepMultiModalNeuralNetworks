# DeepMultiModalNeuralNetworks

Here is the code allowing us to get results regarding different architectures and configurations of a Multimodal Neural Network.

# Installation
Once Python installed, please run the following 3 commands in order to install the correct versions to reproduce our results
```
pip install h5py
pip install keras 2.1.6
pip intall tensorflow==1.8.0
```

# Data
You will find in the `data` folder, AVLetters data that we extracted (and transformed in numpy arrays) and organized in a h5 file. The original dataset can be downloaded from [here](http://www.ee.surrey.ac.uk/Projects/LILiR/datasets/avletters1/index.html).

You can find in this file a key corresponding to each dataset person's name. Each key will let you access independently visual data, audio data and their labels (one hot format and plain format).

# Running the models
This repository contains data that we used and a single file, called `Xflow.py`, in which the model described in the paper is implemented. The code comes from XFlow's authors, but we added some comments to help you to build each model. If you want to perform a scripted architecture exploration, we advise you to create several python files instead of commenting / uncommenting each part. If that's the case, don't forget to edit the `__init__.py` in `model`.

The current state of `run_model.py` will repeat 10 Leave One Out Cross Validation on AVLetters data. This script allows to perform exactly the same procedure as explained in our paper. On top of that it will also save all data along training in a file called `results_XFlow.json`. Make sure to change its name if you run several models.
