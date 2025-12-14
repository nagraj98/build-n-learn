"""
We will be implementing attention from the attention is all you need paper.
"""

import math
import json
import os
import numpy as np
import scipy as sp

from attention.bpe import Tokenizer


class Embedder:
    """
    Docstring for Embedder
    """
    def __init__(self, vocabulary: list, embedding_size: int = 512):
        self.vocabulary = vocabulary
        self.embedding_matrix = np.random.rand(len(vocabulary), embedding_size)

    def get_embedding(self, token: str):
        """
        Docstring for get_embedding
        
        :param token: input token to get embedding for
        :return: embedding vector for the token
        """

        if token in self.vocabulary:
            token_index = self.vocabulary.index(token)
            return self.embedding_matrix[token_index]
        else:
            raise ValueError(f"Token {token} not in vocabulary")
    
    def get_token_from_embedding(self, embedding: np.array):
        """
        Docstring for get_token_from_embedding
        
        :param embedding: input embedding vector
        :return: token corresponding to the embedding
        """

        # Find the closest embedding in the embedding matrix
        distances = np.linalg.norm(self.embedding_matrix - embedding, axis=1)
        closest_index = np.argmin(distances)
        return self.vocabulary[closest_index]


class AttentionHead:
    """
    A single Attention Head with its own query, key and value weights
    """

    def __init__(self, dmodel, dq, dk, dv):
        self.dmodel = dmodel
        self.dk = dk
        self.dq = dq
        self.dv = dv

        # Initialise the weight matrices for Query, Key and Value
        self.wq = np.random.rand(dmodel, dq)
        self.wk = np.random.rand(dmodel, dk)
        self.wv = np.random.rand(dmodel, dv)

        self.query = None
        self.key = None
        self.value = None
        self.attention = None
        

    def apply_self_attention(self, embeddings_matrix: np.array):
        """
        Docstring for apply_self_attention
        
        :param embeddings_matrix: matrix of the input embbeddings, of size (num_tokens, dmodel) or x*dmodel
        """

        # transform the input into Query, Key and Value
        query = np.matmul(embeddings_matrix, self.wq) # so this gives us x*dq or x*64
        key = np.matmul(embeddings_matrix, self.wk) # so this gives us x*dk or x*64
        value = np.matmul(embeddings_matrix, self.wv)

        # apply attention formula : softmax((Q.Kt)/√dk).V
        q_dot_ktranspose = np.matmul(query, np.transpose(key))/math.sqrt(self.dk)
        qkt_softmax = sp.special.softmax(q_dot_ktranspose, axis=1)
        self.attention = np.matmul(qkt_softmax, value) # x*(dmodel/h)


class LayerNorm:
    """
    Docstring for LayerNorm
    """
    def __init__(self, dmodel, eps=1e-5):
        self.gamma = np.ones(dmodel)
        self.beta = np.zeros(dmodel)
        self.eps = eps

    def forward(self, x):
        # x shape: (num_tokens, dmodel)
        mean = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


def main():

    OUTPUT_DIRECTORY = "./attention/vocab/"
    dmodel = 64
    heads_count = 8
    dq = dk = dv = int(dmodel/heads_count)

    attention_heads = [AttentionHead(dmodel, dq, dk, dv) for i in range(heads_count)]

    # splitting the input into tokens using the tokenizer
    vocabulary_map_jsonsafe = json.load(open(os.path.join(OUTPUT_DIRECTORY, "vocabulary_map.json"), 'r', encoding='utf-8'))
    vocabulary_map = {
        tuple(k.split("\u241F")): v for k, v in  vocabulary_map_jsonsafe.items()
    }
    tokenizer = Tokenizer(vocabulary_map=vocabulary_map)

    # converting the tokens into embeddings using the embedder
    vocabulary = open(os.path.join(OUTPUT_DIRECTORY, "vocabulary.txt"), 'r', encoding='utf-8').read().splitlines()
    embedder = Embedder(vocabulary=vocabulary, embedding_size=dmodel)

    layer_norm1 = LayerNorm(dmodel)


    # Start the processing
    input_chunk = ["The cat sat on the mat."]

    tokenizer.encode_input(input_chunk)
    embeddings = [embedder.get_embedding(token) for token in tokenizer.encoded_tokens]
    embeddings_matrix =  np.array(embeddings)

    for head in attention_heads:
        head.apply_self_attention(embeddings_matrix)

    Z_concat = np.concatenate([head.attention for head in attention_heads], axis=1)  # x*dmodel

    # final projection
    w0 = np.random.rand(dmodel, dmodel)
    Z_out = np.matmul(Z_concat, w0)  # x*dmodel

    # add the projection output to the original embeddings (residual connection)
    final_embeddings = Z_out + embeddings_matrix  # x*dmodel

    attention_out = layer_norm1.forward(final_embeddings)

    # TODO : add feed forward network after this --- IGNORE ---

    # TODO : add the final linear layer for predictions, logits = X_out W_vocab + b
    # where W_vocab is of size dmodel*vocab_size

    # TODO : apply softmax to get probabilities over the vocabulary
    # TODO : decoder attention with masking - this is different from the encoder attention implemented here

    # TODO : loss function (categorical cross-entropy) and backpropagation to train the model


if __name__ == "__main__":
    main()