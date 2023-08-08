from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
import os
from os import path
import shutil
import json
from pathlib import Path

from sentence_transformers.models import Pooling
import logging
from tqdm.autonotebook import trange

import numpy as np
from numpy import ndarray
from torch import nn, Tensor
import torch

from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.configuration import OptimizationConfig, AutoQuantizationConfig

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class SentenceTransformerOptim:
    """
    Loads a Quantized SentenceTransformer model that can be used to encode sentences into vectors

    """

    def __init__(self, onnx_dir, pooling_model=None):
        file_name = "model.onnx"
        self.model = ORTModelForFeatureExtraction.from_pretrained(onnx_dir, file_name=file_name)

        # get pooling model
        pool_path = path.join(onnx_dir, 'pooling_config.json')
        if not pooling_model and path.exists(pool_path):
            with open(pool_path) as f:
                self.pooling_config = json.load(f)
            logger.warning(
                f'Using found pooling config: {self.pooling_config}\nIf normalized embeddings are required, set normalize_embeddings=True in model.encode')

            self.pooling_model = Pooling(**pooling_config)
        else:
            logger.warning(
                "No pooling model found. Creating a new one with MEAN pooling.\nIf normalized embeddings are required, set normalize_embeddings=True in model.encode")
            self.pooling_model = Pooling(self.model.config.hidden_size, 'mean')

        self.tokenizer = AutoTokenizer.from_pretrained(onnx_dir)

    def encode(self, sentences: Union[str, List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = 'cpu',
               normalize_embeddings: bool = False) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Based on: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py

        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """

        if not device == 'cpu':
            raise ValueError(f'Only supported device is cpu. Got device: {device}')

        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != 'sentence_embedding':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            features = self.tokenizer(sentences_batch, max_length=None, truncation=True, return_tensors='pt')

            with torch.no_grad():
                model_output = self.model(**features)

                out_features = {}

                out_features['token_embeddings'] = model_output[0]
                out_features['attention_mask'] = features['attention_mask']
                self.pooling_model(out_features)

                if output_value == 'token_embeddings':
                    embeddings = []
                    for token_emb, attention in zip(token_embeddings, features['attention_mask']):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0:last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features['sentence_embedding'])):
                        row = {name: out_features[name][sent_idx] for name in out_features}
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes the texts
        """
        return self.tokenizer.tokenize(texts)

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings


def _load_auto_model(model_name_or_path, pooling_model=None):
    """
    Creates a simple Transformer + Mean Pooling model and returns the modules
    """
    model = ORTModelForFeatureExtraction.from_pretrained(model_name_or_path, from_transformers=True)
    pooling_path = path.join(model_name_or_path, '1_Pooling/config.json')

    if path.exists(pooling_path):
        with open(pooling_path) as f:
            pooling_config = json.load(f)
    else:
        pooling_config = None

    if isinstance(pooling_model, Pooling):
        pass
    elif isinstance(pooling_config, dict) and 'pooling_mode_mean_tokens' in pooling_config:
        print(f"Found Pooling config. If normalized embeddings required, use normalize_embeddings in model.encode")
        pooling_model = Pooling(**pooling_config)

    else:
        logger.warning(
            "No pooling model found. Creating a new one with MEAN pooling.\nIf normalized embeddings required, use normalize_embeddings in model.encode")
        pooling_model = Pooling(model.config.hidden_size, 'mean')

    return model, pooling_config


def optimize_model(model_name_or_path: Optional[str] = None,
                   pooling_model: Optional[str] = None,
                   save_dir: Optional[str] = '',
                   optimize_mode='onnx',
                   ):

    if not save_dir:
        save_dir = os.getcwd()

    os.makedirs(save_dir, exist_ok=True)

    onnx_model, pooling_config = _load_auto_model(model_name_or_path, pooling_model=pooling_model)

    if optimize_mode == 'onnx':
        print('Converting model to onnx..')
    else:
        print(f'Optimizing onnx model using {optimize_mode}..')

    saved_dir = _optimize(save_dir, onnx_model, pooling_config, optimize_mode)

    if isinstance(pooling_config, dict):
        with open(path.join(saved_dir, 'pool_config.json'), 'w') as f:
            json.dump(pooling_config, f)


def _optimize(save_dir, model, pooling_config, optimize_mode='onnx'):
    """
    Quantize sentence transformers model

    """

    supported_names = ['onnx', 'graph_optim']
    if optimize_mode not in supported_names:
        raise ValueError(f'Optimization {optimize_mode} not in supported quantization: {supported_names}')

    onnx_path = save_dir
    model.save_pretrained(onnx_path)

    if optimize_mode == 'onnx':
        print(f'Optimized model using {optimize_mode} saved at {onnx_path}')
        return onnx_path

    optimizer = ORTOptimizer.from_pretrained(model)

    # optimization_config=99 enables all available graph optimisations
    optimization_config = OptimizationConfig(optimization_level=99)

    shutil.rmtree(onnx_path)

    # apply the graph optimization configuration to the model
    graph_path = onnx_path
    optimizer.optimize(
        save_dir=graph_path,
        file_suffix='',
        optimization_config=optimization_config,
    )

    if optimize_mode == 'graph_optim':
        print(f'Optimized model using {optimize_mode} saved at {onnx_path}')
        return graph_path

    return
