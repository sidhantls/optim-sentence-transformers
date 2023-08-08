# Optimized Sentence Transformers 

This package simplifies Sentence Transformers model deployment via [ONNX](https://onnxruntime.ai/) and [optimum](https://huggingface.co/docs/optimum/). Up to 40% faster inference of [SentenceTransformer](https://www.sbert.net/) models with minimal code changes. 

<p align="center">
  <img src="https://github.com/sidhantls/optimized-st-ckpt/blob/main/imgs/workflow.PNG" width="700" height="140"/>
</p>

This provides functionality to optimize models using optimum and perform inference using the same [sentence transformer](https://www.sbert.net/) `model.encode` method. 


## Installation
Requires Python 3.8+ 

### Install with pip
`pip install optim_sentence_transformers`

### Install from source
```
git clone github.com/sidhantls/optim-sentence-transformers;
cd optim-sentence-transformers;
pip install -e .;
```

## Stats

Model name: sentence-transformers/all-distilroberta-v1
<p align="center">
  <img src="https://github.com/sidhantls/optimized-st-ckpt/blob/main/imgs/latency_percent_difference.png" width="700" height="400" />
</p>



## Usage 

See `benchmark_optimized.ipynb`

Supported Optimizations: "onnx" and "graph_optim" ((graph optimization)[https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization])

```
from sentence_transformers import SentenceTransformer
from optim_sentence_transformers import SentenceTransformerOptim, optimize_model

model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')

# train if required and save
model.save('trained_model') 

model_name_or_path = 'trained_model'
# model_name_or_path = 'sentence-transformers/all-distilroberta-v1' # to optimize default model 

# optimize model
save_dir = 'onnx'
optimize_model(model_name_or_path = model_name_or_path,
             pooling_model=None,
             save_dir='onnx',
             optimize_mode='onnx'                                 
             )
             
# load optimized model 
optim_model = SentenceTransformerOptim(save_dir)
optim_model.encode(['text'], normalize_embeddings=True)
``` 

In some cases model.encode in sentence-transformers will always return normalized vectors due to normalization layer during init. Here, if vectors are required to be normalized, set normalize_embeddings=True. 

## Contributions 
Contributions are welcome. New feature contributions also are welcome and idea is to keep it easy to use, support most use-cases with few number of nobs.

## References 
* [Hugginface optimum](https://huggingface.co/docs/optimum/) 
* [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) 
* [onnxruntime](onnxruntime.ai)
* [philschmid's blog post](https://www.philschmid.de/optimize-sentence-transformers) about Onnx and model quantization  

