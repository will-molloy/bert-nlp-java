[![Build Status](https://travis-ci.com/wilmol/bert-nlp-java.svg?branch=master)](https://travis-ci.com/wilmol/bert-nlp-java)
[![codecov](https://codecov.io/gh/wilmol/bert-nlp-java/branch/master/graph/badge.svg)](https://codecov.io/gh/wilmol/bert-nlp-java)
[![GitHub license](https://img.shields.io/github/license/wilmol/bert-nlp-java.svg)](https://github.com/wilmol/bert-nlp-java/blob/master/LICENSE)

# BERT NLP Java

## What's included
* Java code to read and process a BERT TensorFlow model
* Gradle tasks to automate the training process

## Requirements
* Java 8

Additionally, if you want to train models:
* Python 3.6
* A CUDA GPU (i.e. NVIDIA), with the various software installed
  * Just follow https://www.tensorflow.org/install/gpu
  * Alternatively, access to a Google Cloud TPU 

## Supported NLP tasks
* [Document categorization](bert-nlp-java/src/test/java/com/wilmol/bert/DocumentCategorizerTest.java)

**WIP:**
* Question answering
* Named entity recognition (NER)
