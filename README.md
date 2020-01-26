# TTS-Final-Task
Text to speech final task

```
By Juan Ormaechea Belloso
```

## Index

- 1. Introduction
- 2. My Git projects
   - 2.1 Most important files of the project
- 3. Explaining the code
   - 3.1 Downloading necessary files part.
   - 3.2 Installing TTS requirements part.
   - 3.3 Setting up different Jason files part.
   - 3.3 Training part.
   - 3.4 Initialize models part
- 4. Tacotron and tacotron
- 5. Conclusion



## 1. Introduction

First, I’m going to star my job with one introduction telling what we are going to
see in my project.

So, starting with that I’m going to say that my job is trying to synthesis a
Spanish audio using a speech synthesis from a text that we wrote before. That
process is called text to speech, but before to start talking about TTS I need to
explain what I’m speaking about.

On the one hand a Speech synthesis is the artificial production of
human speech. On the other hand, a text-to-speech (TTS) system converts
normal language text into speech.

Synthesized speech can be created by concatenating pieces of recorded
speech that are stored in a database. Systems differ in the size of the stored
speech units; a system that stores phones provides the largest output range but
may lack clarity.

The quality of a speech synthesizer is judged by its similarity to the human
voice and by its ability to be understood clearly. An intelligible text-to-speech
program allows people with visual impairments or reading disabilities to listen to
written words on a home computer.

A text-to-speech system is composed of two parts: a front-end and a back-end.
The front-end has two major tasks. First, it converts raw text containing symbols
like numbers and abbreviations into the equivalent of written-out words. This
process is often called text normalization, pre-processing, or tokenization. The
front-end then assigns phonetic transcriptions to each word, and divides and
marks the text into prosodic units, like phrases, clauses, and sentences.

The process of assigning phonetic transcriptions to words is called text-to-
phoneme or grapheme-to-phoneme conversion. Phonetic transcriptions and
prosody information together make up the symbolic linguistic representation that
is output by the front-end. Finally, The back-end then converts the symbolic
linguistic representation into sound.


## 2. My Git projects

My project is 100% based on another existing git project called Mozilla TTS.
This one is a deep learning based text2speech solution. It favors simplicity over
complex and large models and yet, it aims to achieve state of the art results.

Currently, it is plotted on Tacotron and Tacotron2.

Tacotron based model is smaller and targets faster training/inference whereas
Tacotron2 based model is almost 3 times larger but achieves better results by
using a neural vocoder (WaveRNN).

Mozilla TTS is able to give on par or better performance compared to other
open-sourced text2speech solutions. It also supports various languages
(English, German, Chinese etc.), with very little change.

![USJ](https://drive.google.com/uc?export=view&id=1HWI0n2dt2shCmTx8_47kmVpP_mJgIR6g)



### 2.1 Most important files of the project

The project is composed by a lot of complex and different kind of python files
and Jason files. However, I’m going to explain how they are completed the most
important files.


If we navigate into utils → text, we can see something like this:

Here are the python files that contains symbols like numbers and abbreviations
that we need to converts into their equivalent written-out words as I say before,
remember that this process is called tokenization and when we arrived into this
files we are in the process of the front end of the TTS. Here we can see some of
that symbols that I mentioned before:

Then, tests → data → ljspeech folder contains the wavs files, I mean, the
recorded speech files that we need to get the synthetized audio. Remember
that we are going to use these files in the artificial human speech production
using speech synthesis as I say before.


Models folder contains the different tools that we are going to use for train our
recorded pieces of audio. In this case, tacotron and tacotron 2.

This part I think is the most important part for train our project. Later I’m going to
explain what they are and how they work.


## 3. Explaining the code

The project is developed in Google colab (a free cloud service) and with python
(an interpreted, high-level, general-purpose programming language).

We can divide the code in five parts:

```
1 - Downloading necessary files part.
```
```
2 - Installing TTS requirements part.
```
```
3 - Setting up different Jason files part.
```
```
4 - Training part.
```
```
5 - Initialize models part.
```
So, I’m going to explain, what do we do in all the different part and how do they
work.


### 3.1 Downloading necessary files part.

We are going to start the code downloading the necessary files from the
different gits URLs as we can see below:

As we can see, we download the Mozilla TTS project and WaveRNN project.
However, if we don’t want to download the Mozilla TTS project because we
developed our personal git project for this code, we can also put the URL there,
for example, I can put my personal git URL for this code:

https://github.com/jojoji54/TTS-Final-Task.git

However, I prefer to use the Mozilla TTS project instead of mine.

Also, we download WaveRNN project because we want to train a model using
also tacotron 2. We have to know that WaveRNN is a technique used to
synthesize neural audio much faster than other neural synthesizer by providing
rudimentary improvements.

This implementation is based on tensorflow and will require librosa to
manipulate raw audio.

Then, we must download the audio files that we want to use to train and get a
synthesize speech.


After download the audio files we create a train validation splits.

At this moment, the Test dataset provides the gold standard used to evaluate
the model. It is only used once a model is completely trained:

Then, we are going to download pretrained models for don’t waste our time or
training one, however we can train our pretrained models:

If you are thinking what those files are, I have to say that pretrained models’
files are a model created by someone else to solve a similar problem. Instead of


building a model from scratch to solve a similar problem, we use the model
trained on other problem as a starting point.


### 3.2 Installing TTS requirements part.

At this part we are going to install the Mozilla TTS project for use the config
code of the project:

Setup.py is the main python class of the project, from this file we can use every
file of the project as we can see on the resume where the program is installing
the code, we are installing the different parts of the project that we need for
work the code.


### 3.3 Setting up different Jason files part.

Then, we are going to stablish some parameter for the Jason file. These
parameters are just for this code and can will change depend our objectives,
also these parameters are very important for train a synthetize speech:

We are going to repeat this process also for WaveRNN Jason file:


### 3.3 Training part.

After we configure our different Jason files, we can start training our data as we
can see in the image below:

The code extracts the parameters of the Jason’s files, then it takes the recorded
speech files from the database and creates a Python NumPy Array files. This
kind of files are a grid of values, all the same type, and are indexed by a tuple of
nonnegative integers. The number of dimensions is the rank of the array;
the shape of an array is a tuple of integers giving the size of the array along
each dimension.

However, we are training a synthesize audios with two different models,
tacotron and tacotron 2, so we are going to train the audios with two different
tools for use with tacotron and tacotron 2, the first one is for tacotron, and this
one is for tacotron 2 using the Jason file of waveRNN:


After the program finished with the training, we can see that the code creates
something called: Pre- trained models. One for TTS and another one for
waveRNN.

These models are very important because they will give us the final solution,
however, we have to locate it first, later we can create the synthetize speech
using tacotron and tacotron 2:


### 3.4 Initialize models part

The most important part of the code, in this part of the code we do the process
that I explained in the introduction to get the synthesize audio. I mean, first the
code takes the pre trained models that we link before then differs in the size of
the stored speech units. And finally, the code provides the largest output range.

But this is just the first part, then as we have a phonetic transcription and we
can use It for make up the symbolic linguistic representation and converts the
symbolic linguistic representation into sound as we can see at least.


## 4. Tacotron and tacotron

As I said before, if we want to train a model, we need to use tacotron, tacotron 2
or both. If we use both we can get better result. However, if we have to chose
one of them is preferring to select tacotron 2 because is like the second version
of tacotron, but this one has better result.

At tis moment I’m going to explain how they work and why are so important.

So, staring with tacotron I can say that is an end-to-end generative text-to-
speech model that synthesizes speech directly from text and audio pairs.
Tacotron achieves a 3.82 mean opinion score on US English. Tacotron
generates speech at frame-level and is, therefore, faster than sample-level
autoregressive methods.

The model is trained on audio and text pairs, which makes it very adaptable to
new datasets. Tacotron has a seq2seq model that includes an encoder, an
attention-based decoder, and a post-processing net. As seen in the architecture
diagram below, the model takes characters as input and outputs a raw
spectrogram. This spectrogram is then converted to waveforms.

The figure below shows what the CBHG module looks like.


A character sequence is fed to the encoder, which extracts sequential
representations of text. Each character is represented as a one-hot vector and
embedded into a continuous vector. Non-linear transformations are then added,
followed by a dropout layer to reduce overfitting. This, in essence, reduces the
mispronunciation of words.

The decode used is a tanh content-based attention decoder. The waveforms are
then generated using the Griffin-Lim algorithm. The hyper-parameters used for
this model are shown below.


Finally, Tacotron 2 is an AI-powered speech synthesis system that can convert
text to speech and is a neural network architecture synthesizes speech directly
from text. It functions based on the combination of convolutional neural network
(CNN) and recurrent neural network (RNN).

the system consists of two components:

1. A recurrent sequence-to-sequence feature prediction network with
    attention which predicts a sequence of mel spectrogram frames from an
    input character sequence
2. A modified version of WaveNet which generates time-domain waveform
    samples conditioned on the predicted mel spectrogram frames

The differences between Tacotron and tacotron 2 are easy to see:

Tacotron 2 is said to be an amalgamation of the best features of
Google’s WaveNet, a deep generative model of raw audio
waveforms, and Tacotron, its earlier speech recognition project. The sequence-
to-sequence model that generates mel spectrograms has been borrowed
from Tacotron, while the generative model synthesizing time domain waveforms
from the generated spectrograms has been borrowed from WaveNet.

Tacotron is considered to be superior to many existing text-to-speech programs.
In the vocodingprocess, Tacotron uses the Griffin-Lim algorithm for phase
estimation, followed by an inverse short-time Fourier transform. However,
Griffin-Lim produces lower audio fidelity and characteristic artifacts when
compared to approaches like WaveNet.


## 5. Conclusion

A text to speech system is a very complex system that not any one can do it. It
creates from a lot of very difficult structure connections.

So, I create a text to speech system using Python code for synthetize an audio
from a Spanish audio that I stored before. Then I trained the different files that I
stored, and the system gave me some files that I’m going to organize in a matrix
that I’m going to use for create the synthetize audio. For this, I used a pre
trained model for help me with the training.

I create my task based on another one because of that anagrams and complex
structure of the system. However, when I had been creating the code, I learnt a
lot of things about it; How Tacotron and Tacotron 2 works, the python code that
I use for the code... and a lot more. I also know how works tacotron 2 using
their different neural networks that it has, recurrent neural networks for
example. Know I can create a different text to speech system based on what I
did.

Anyway, according of how much time I needed for complete this project , I need
more than a week. I also have to say that I could not have finished the project if
Rafa didn’t help me for finish it.

