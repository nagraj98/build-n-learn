# We will be implementing attention from the attention is all you need paper.

import numpy as np
import scipy as sp
import math

dmodel = 64
h = 8
dq = dk = int(dmodel/h)
dv = int(dmodel/h)

Wq = np.random.rand(dmodel, dq)
Wk = np.random.rand(dmodel, dk)
Wv = np.random.rand(dmodel, dv)


input_tokens = ["Hello", "this", "is", "Harry", "calling", "from", "Hogwarts"]
# x = 7

# def get_embedding(token, dmodel):
#     embedding = token

#     # size of embeddings = embedding_size
#     # assume embedding size is 512
#     # d_model = 512
#     return 

# input_embeddings = [get_embedding(token) for token in input_tokens]

def get_X(input_tokens):
    return np.random.rand(len(input_tokens), dmodel)

X = get_X(input_tokens)
# this matrix becomes 7*512

# 8 attention heads, so one head can be of dmodel/8

Query = np.matmul(X, Wq) # so this gives us x*dq or x*64
Key = np.matmul(X, Wk) # so this gives us x*dk or x*64
Value = np.matmul(X, Wv)

def attention():
    QKt = np.matmul(Query, np.transpose(Key))/math.sqrt(dk)
    QKt_softmax = sp.special.softmax(QKt, axis=1)
    attn = np.matmul(QKt_softmax, Value)
    print(attn)
    # print(QKt)

# print(X)
# print(Wq)
attention()

# for these input embeddings, 
# we want a 7*7 matrix that tells the relation between all word pairs.

# my = [1,2,3,4,5,6,7,8,9,10]

# def softmax(nums) :
#     exp_nums = [math.exp(num) for num in nums]
#     return [num/sum(exp_nums) for num in exp_nums]

# print(softmax(my))
# print(sum(softmax(my)))



# # Difference between softmax and normalisation :
# normalisation just scales the inputs between 0 an d1
# softmax does couple of things in addition to plain normalisation :
#     1. exphasizing large values, diminishing small values
#     2. handling negative inputs