# Text Similarity with Contextualised Embeddings

This repository contains a library that I use for my **Natural language processing** projects.

All the code in the library is based on [Pytorch](https://pytorch.org/mobile/home/).

<a href="https://pytorch.org/"><img src="https://miro.medium.com/max/691/0*xXUYOs5MWWenxoNz" width="200"/></a>
<br>

Most of the models in the library are built upon pretrained models from the [sentence-transformers](https://github.com/UKPLab/sentence-transformers) library,

<a href="https://fastapi.tiangolo.com/"><img src="https://www.sbert.net/_static/logo.png" width="200"/></a>
<br>

which offers a wide variety of options for very performant sentence embeddings models, which is in turn based on the popular [transformers](https://huggingface.co/) library by Huggingface

<a href="https://fastapi.tiangolo.com/"><img src="https://repository-images.githubusercontent.com/155220641/a16c4880-a501-11ea-9e8f-646cf611702e" width="200"/></a>

## ✨ Contents

1. Scripts to train and test word-level and sentence-level embeddings models on various NLP tasks
2. Wrappers around Huggingface pretrained model to perform experiments on text similarity tasks
3. A semantic search pipeline built on top of performing sentence embedding models and approximate nearest neighbours algorithms
4. A model compression pipeline that includes functions to distill, prune, quantize and convert models to optimized formats such as Onnx, Tensorflow Lite and Torchscript to use in edge devices
5. Scripts to train models on a variety of text similarity and sequence classification tasks

## Work in Progress
- [ ] [Sense-aware embeddings](https://github.com/cr1m5onk1ng/text_similarity/tree/master/src/pipeline/word_sense) creation exploiting WordNet relations and contextualised embeddings
- [ ] PySpark integration for faster text preprocessing for larger datasets

## Author

**Mirco Cardinale**
[Personal website](https://mirco-cardinale-portfolio.herokuapp.com/)

## 🔖 LICENCE

[Apache-2.0](https://github.com/cr1m5onk1ng/text_similarity/LICENSE)
