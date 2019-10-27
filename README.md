[![Build Status](https://travis-ci.com/wilmol/bert-nlp-java.svg?branch=master)](https://travis-ci.com/wilmol/bert-nlp-java)
[![codecov](https://codecov.io/gh/wilmol/bert-nlp-java/branch/master/graph/badge.svg)](https://codecov.io/gh/wilmol/bert-nlp-java)
[![GitHub license](https://img.shields.io/github/license/wilmol/bert-nlp-java.svg)](https://github.com/wilmol/bert-nlp-java/blob/master/LICENSE)

# BERT NLP Java

## Requirements
* Java 8

Additionally, if you want to train models:
* Python 3.6
* A CUDA GPU (i.e. NVIDIA), with the various software installed
  * Just follow https://www.tensorflow.org/install/gpu
  * Alternatively, access to a Google Cloud TPU 

## What's included
* Java code to read and process a BERT TensorFlow model
* Gradle tasks and Java code to automate the training process
* Interactive examples

## Supported NLP tasks
* [Text classification](#text-classification)

**WIP:**
* Question answering
* Named entity recognition (NER)

## Text classification
* AKA document classification
* This assigns a label to a passage of text according to its content

### Run example main class
* Just run the following gradle task:

### Training a model
* Organise your training data as follows:
  * 