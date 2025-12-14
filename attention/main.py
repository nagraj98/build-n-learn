"""
We will be implementing attention from the attention is all you need paper.
"""

import math
import numpy as np
import scipy as sp


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
        

    def apply_self_attention(self, input_X):
        """
        Docstring for apply_self_attention
        
        :param input_X: matrix of the input embbeddings
        """

        # transform the input into Query, Key and Value
        query = np.matmul(input_X, self.wq) # so this gives us x*dq or x*64
        key = np.matmul(input_X, self.wk) # so this gives us x*dk or x*64
        value = np.matmul(input_X, self.wv)

        # apply attention formula : softmax((Q.Kt)/√dk).V
        q_dot_ktranspose = np.matmul(query, np.transpose(key))/math.sqrt(self.dk)
        qkt_softmax = sp.special.softmax(q_dot_ktranspose, axis=1)
        self.attention = np.matmul(qkt_softmax, value)

        # TODO : Now using this attention matrix, what do we update ?

def main():

    
    dmodel = 64
    heads_count = 8
    dq = dk = dv = int(dmodel/heads_count)

    attention_heads = [AttentionHead(dmodel, dq, dk, dv) for i in range(heads_count)]

    input_chunk = ["Hello this is Harry calling from Hogwarts"]

    def tokenize(input_chunk: str):
        print("splitting the input into tokens")
        return input_chunk.split()

    def get_embedding(token: str):
        print("getting the embedding for the token")
        return len(token) * [0.01]  # dummy embedding

    def get_X(input_chunk: str):
        input_tokens = tokenize(input_chunk)
        input_embeddings = [get_embedding(token) for token in input_tokens]
        print(np.array(input_embeddings))
        return np.array(input_embeddings)

    X = get_X(input_chunk)
    # this matrix becomes 7*512

if __name__ == "__main__":
    main()