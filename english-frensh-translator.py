import pdb
import pickle
import string
from flask import Flask, render_template, request, jsonify
from flask import Flask
from flask_restful import Resource, Api
from flask_cors import CORS
import time
import numpy as np
import scipy
import sklearn
import pandas as pd
from os import getcwd

en_embeddings_subset = pickle.load(open("en_embeddings.p", "rb"))
fr_embeddings_subset = pickle.load(open("fr_embeddings.p", "rb"))


def get_dict(file_name):
    """
    This function returns the english to french dictionary given a file where the each column corresponds to a word.
    Check out the files this function takes in your workspace.
    """
    my_file = pd.read_csv(file_name, delimiter=' ')
    etof = {}  # the english to french dictionary to be returned
    for i in range(len(my_file)):
        # indexing into the rows.
        en = my_file.loc[i][0]
        fr = my_file.loc[i][1]
        etof[en] = fr

    return etof


en_fr_train = get_dict('en-fr.train.txt')
en_fr_test = get_dict('en-fr.test.txt')



def get_matrices(en_fr, french_vecs, english_vecs):
    """
    Input:
        en_fr: English to French dictionary
        french_vecs: French words to their corresponding word embeddings.
        english_vecs: English words to their corresponding word embeddings.
    Output: 
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the French embeddings.
    """
    
    # list to append the english word 
    X_w = list()
    # list to append the French word 
    Y_w = list()
    
    
    # list to append the english word embeddings
    X_l = list()
    # list to append the French word embeddings
    Y_l = list()
    
    # get the english words from the dict
    english_words = list(en_fr_train.keys())
    # get the French words from the dict
    french_words = list(en_fr_train.values())
    
    # for each (english_word, its french_translation):
    for word in zip(english_words,french_words):
        
        # if the english word in our subset of english embeddings 
        # and if the french word in our subset of french embeddings 
        if word[0] in english_vecs.keys() and word[1] in french_vecs.keys():
            
            # append the embedding for the english word[i] in X_l
            X_l.append(english_vecs[word[0]])
            # append the embedding for the french word[i] in Y_l
            Y_l.append(french_vecs[word[1]])
            
            # list of found english words
            X_w.append(word[0])
            # list of found french words
            Y_w.append(word[1])
            
            
    # transform the list of english word embeddings to a numpy array
    X = np.vstack(X_l)
    # transform the list of french word embeddings to a numpy array
    Y = np.vstack(Y_l)
    
    return X_w, Y_w, X, Y



# get training dataset
f_eng_w, f_fr_w, X_train, Y_train = get_matrices(en_fr_train, fr_embeddings_subset, en_embeddings_subset)

f_eng_w_val, f_fr_w_val, X_val, Y_val = get_matrices(en_fr_test, fr_embeddings_subset, en_embeddings_subset)

def compute_loss(X, Y, R):
    
    # in this notbook X in in the shape of: (4932, 300)
    # in this notbook Y in in the shape of: (4932, 300)
    # in this notbook R in in the shape of: (300, 300)
    
    # m is the number of training examples
    m = X.shape[0]
    
    # X(4932, 300).R (300, 300) -> (4932, 300)
    # XR (4932, 300) - Y (4932, 300) -> (4932, 300)
    
    # diff between XR and Y
    diff = np.dot(X, R) - Y
    # elementwise squared
    squared_diff = diff ** 2
    # sum of squared diffs
    sum_squared_diff = np.sum(squared_diff)
    
    loss = sum_squared_diff / m
    
    return loss


def compute_gradient(X, Y, R):
    
    # m is the number of training examples
    m = X.shape[0]
    
    # shape(X) -> (4932, 300)
    # shape(XTranspose) -> (300, 4932)
    # shape(X.R-Y) -> (4932, 300)
    # XT (300, 4932) . (X.R-Y) (4932, 300) -> (300, 300)

    # gradient is X^T(XR - Y) * 2/m
    gradient = np.dot(X.transpose(),np.dot(X,R)-Y)*(2/m)
    
    return gradient

def train(X, Y, train_steps=100, learning_rate=0.0001, show_loss_after_n_iteration=25):
    
    # Shape(R) -> (300, 300)
    R = np.random.rand(X.shape[1], X.shape[1])
    
    # for each iteration:
    # compute the gradient and modify R based on that gradient and the learning_rate
    for iteration in range(train_steps):
        
        # compute the gradient
        gradient = compute_gradient(X, Y, R)
        
        # modify R
        R = R - (learning_rate * gradient)
        
        # compute the loss and modifying R
        loss = compute_loss(X_train, Y_train, R)
        
        # show loss after each 100 iteration
        if iteration % show_loss_after_n_iteration == 0:
            print("Loss After {} is: {}".format(iteration, loss))
        
    return R


def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''
    # you have to set this variable to the true label.
    cos = -10
    dot = np.dot(A, B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = dot / (norma * normb)

    return cos

def nearest_neighbor(v, candidates, k=1):
    
    similarity_l = []

    # for each candidate vector...
    for row in candidates:
        # get the cosine similarity
        cos_similarity = cosine_similarity(v,row)

        # append the similarity to the list
        similarity_l.append(cos_similarity)
        
    # sort the similarity list and get the indices of the sorted list
    sorted_ids = np.argsort(similarity_l)

    # get the indices of the k most similar candidate vectors
    k_idx = sorted_ids[-k:]

    
    return k_idx

def test_vocabulary(X, Y, R):
    '''
    Input:
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the French embeddings.
        R: the transform matrix which translates word embeddings from
        English to French word vector space.
    Output:
        accuracy: for the English to French capitals
    '''

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # The prediction is X times R
    pred = np.dot(X,R)

    # initialize the number correct to zero
    num_correct = 0

    # loop through each row in pred (each transformed embedding)
    for i in range(len(pred)):
        # get the index of the nearest neighbor of pred at row 'i'; also pass in the candidates in Y
        pred_idx = nearest_neighbor(pred[i],Y)

        # if the index of the nearest neighbor equals the row of i... \
        if pred_idx == i:
            # increment the number correct by 1.
            num_correct += 1

    # accuracy is the number correct divided by the number of rows in 'pred' (also number of rows in X)
    accuracy = num_correct / len(pred)

    ### END CODE HERE ###

    return accuracy

def predict (R, X):
     
    if X not in en_embeddings_subset:
        return "word not in my small embeddings"
    
    X = en_embeddings_subset[X]
    
    X.reshape((1, 300))
    
    prediction = np.dot(X, R)
    
    pred_idx = nearest_neighbor(prediction,Y_train)
    
    return f_fr_w[int(pred_idx)]

#R_train = train(X_train, Y_train, train_steps=10, learning_rate=0.8, show_loss_after_n_iteration=100)

R_train=np.loadtxt("R_train.txt")


app = Flask(__name__)
CORS(app)

api=Api(app)

class prediction1(Resource):
    def get(self,text):
        if text in f_eng_w:
           return  [["word ON my english words"],[predict(R_train,text)]]
        
        else :
            return  [["word Not on my english words"],[predict(R_train,text)]]
        
        
        
api.add_resource(prediction1,'/prediction1/<string:text>')

@app.route('/')
def home():
    return render_template('english-frensh-translator.html')

     
if __name__ == "__main__":
    app.run(debug=True)
    
'''if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)
'''