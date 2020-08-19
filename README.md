# Public-AI-Model
This code has been made available for public use. Send us an email if you need an updated version of our work - okon@vmedkit.com

## Preparation Stage
props.py in folder "audio functions" extracts the number of audio channels, sample rate and bit-depth
explore.py in folder "audio functions" standardizes the dataset before training
mfcc.py in folder "audio functions" extracts the Mel-Frequency Cepstral Coefficients (MFCC) features from our audio files

convert.py in folder "data preparation" splits the dataset into training and testing sets. The testing set size will be 20% and we will set a random state.

## Training Stage
We will use a sequential model, starting with a simple model architecture, consisting of four Conv2D convolution layers, with our final output layer being a dense layer. Our output layer will have 10 nodes (num_labels) which matches the number of possible classifications.
For compiling our model, we will use the following three parameters
=> Loss - Categorical Cross Entropy
=> Metrics - Accuracy
=> Optimizer - Adam

