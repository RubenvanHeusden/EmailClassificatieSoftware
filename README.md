# EmailClassificatieSoftware

This repository contains the code and documentation for a project conducted at the City of Amsterdam regarding
the automatic classification of Emails. It contains implementations for several often-used and state-of-the-art
models in the field of Text Classification, as well as A Multitask model called the Multigate Mixture-of-Experts 
model that was tested during the research period. 

## Contents

The project contains several state-of-the-art model for Text Classification, as well as some useful utilities
to streamline the training process, such as methods to create train and test sets with roughly equal label distributions

## Installation

Currently the package has only been tested on Windows 10 but it should also work on Linux and Mac based systems,
possible requiring manual installation of some dependencies found in the requirements.txt
It is highly recommended to install the package in a virtual environment to avoid any conflicts with existing packages.


## Installing dependencies via pip
When install via pip, open a Command Prompt(on Windows)/ terminal and navigate to the root folder of the 
EmailClassificatieSoftware package. Make sure that you have the python package manager(pip) installed

Now to install the dependencies, run the following command:

```
pip install -r requirements.txt
```
This will install all the required dependencies for the package.

## Installing dependencies via Anaconda 
When using Anaconda, open the Anaconda Prompt and navigate to the root folder of the EmailClassificatieSoftware
package. Now to install the required dependencies run the following command:

```
conda env create -f environment.yml
```

This will create a new Conda Virtual Environment with the name 'emailclassificatiesoftware' and all of the 
dependencies required to run the package installed.

## Quick start
For tips and pointers on how to get started with this package, please refer to the 
'getting started' section of the docs

## Working with word embeddings

The CNN, Bidirectional LSTM and Multigate Mixture of Experts models both work with pretrained word embeddings as input. 
In the case of the Dutch Language, the word embeddings that are used in this research are the CLIPS word embeddings
that can be found on (https://github.com/clips/dutchembeddings)

The training scripts for these models will automatically download these word embeddings when they are not 
found in the 'word_vectors' folder in the module, and will use those word embeddings from then onwards.

