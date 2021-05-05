# On-device semantic search engine for small corpuses

WORK IN PROGRESS

This is an attempt to build a small semantic search engine based on contextualized embeddings (BERT-base models) and Nearest Neighbors search that runs on mobile devices.
The project is still in its early stages, since it's quite a complex thing to do and I'm a noob.

## Project description

### Why

Since the [app I'm building](https://github.com/cr1m5onk1ng/nala_android_app/tree/dev) allows the user to save sentences locally for review, if the sentences corpus
grows quite significantly (thousands of sentences), it's quite handy to have a local search engine that does the bothersome job of finding similar/dissimilar sentences in your corpus for you. So that's why.

### The Model

The idea is to compress a sentence encoder model and convert it to torchscript via the tools provided by PyTorch. The model is a sentence-transformers based model, which provides high quality multilingual sentence embeddings.
The model is distilled and fine-pruned using a language-specific corpus collected from the web (news websites, forums and Wikipedia) with a projection layer that reduces the dimension of the output embeddings, allowing to store sentence representions without making the user run out of memory.

### The Search Algorithm

Since there are no libraries specifically built to implement NN algorithms on Android (quite understandably), I'm experimenting with using [this library](https://github.com/stepstone-tech/hnswlib-jna) based on a JNA version of the popular [hnswlib](https://github.com/nmslib/hnswlib) library and hope it doesn't perform too poorly on mobile hardware.
