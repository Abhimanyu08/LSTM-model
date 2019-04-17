#Installing dependencies
import numpy as np
import pandas as pd
import re
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
import random

path = 'Downloads/poems/all.csv'
data = pd.read_csv(path)
data = np.array(data)

x = data[:,1]
print(x.shape)

#Converting all poems into set of words without punctuation marks
for i in range(x.shape[0]):
    x[i] = np.array(list(set(re.findall('[A-Za-z][a-z]+',x[i]))))
    
def readglovevec(file_name):
    with open(file_name,encoding = 'utf8') as f:
        words = set()
        word_to_vec_map = {}
        con = f.readlines()
        for line in con:
            line = line.strip().split()
            c_word = line[0]
            words.add(c_word)
            word_to_vec_map[c_word] = np.array(line[1:],dtype = np.float64)
    i = 1
    word_to_index = {}
    index_to_word = {}
    for w in sorted(words):
        word_to_index[w] = i
        index_to_word[i] = w
        i = i+1
    return word_to_index,index_to_word,word_to_vec_map

wi,iw,wv = readglovevec('Downloads/poems/glove.6B.50d.txt')

def poems_to_indices(x,word_to_indices,max_len):
    m = x.shape[0]
    X = np.zeros((m,max_len))
    
    for i in range(m):
        j = 0
        for w in x[i]:
            w = w.lower()
            if j>=max_len:
                break
            if w in wi.keys():
                X[i,j] = wi[w]
            j = j+1            
    return X
max_len = 110

X_indices = poems_to_indices(x,wi,max_len)

y = data[:,4]
for i in range(len(y)):
    if y[i] == 'Mythology & Folklore':
        y[i] = 0
    elif y[i] == 'Nature':
        y[i] = 1
    else:
        y[i] = 2
print(y.shape)

#Converting y into corresponding array of one _hot vectors
def conv_to_one_hot(y,categories):
    
    Y = np.zeros((y.shape[0],categories))
    for i in range(y.shape[0]):
        Y[i,y[i]] = 1
    return Y
    
categories = 3

Y = conv_to_one_hot(y,categories)
print(Y.shape)
#print(Y[0:200,:])

#Embedding layer which converts each word of the poem in its corresponding glove representation
def pretrained_embedding_layer(word_to_vec,word_to_index):
    vocab_len = len(word_to_index) + 1                  
    emb_dim = word_to_vec["apple"].shape[0]
    
    emb_matrix = np.zeros((vocab_len,emb_dim))
    
    for word,index in word_to_index.items():
        emb_matrix[index,:] = word_to_vec[word]
        
    embedding_layer = Embedding(vocab_len,emb_dim, trainable = False)
    
    embedding_layer.build((None,emb_dim))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer
    
def shuffle(X,Y):
    m = X.shape[0]
    ls = np.array(range(m))
    random.shuffle(ls)
    return X[ls,:],Y[ls,:]
    
X,Y = shuffle(X_indices,Y)
#print(X.shape,Y.shape)    

#Model
def model(input_shape, word_to_vec_map, word_to_index):
    poem_indices = Input(shape = input_shape, dtype = 'int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map,word_to_index)
    embeddings = embedding_layer(poem_indices)
    
    X = LSTM(128,return_sequences = True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128,return_sequences = True)(X)
    X = Dropout(0.5)(X)
    X = LSTM(128)(X)
    X = Dropout(0.5)(X)
    X = Dense(3,activation = 'softmax')(X)
    
    model = Model(inputs = poem_indices, outputs = X)
    
    ### END CODE HERE ###
    
    return model
  
model = model((max_len,),wv,wi)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs = 10, batch_size = 16, shuffle=True,validation_split = 0.1)
