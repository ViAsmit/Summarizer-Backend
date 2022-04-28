from flask import Flask, jsonify, request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi


import numpy as np
import pandas as pd
import pickle
from statistics import mode
import nltk
from nltk import word_tokenize
from nltk.stem import LancasterStemmer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input,LSTM,Embedding,Dense,Concatenate,Attention
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup


# encoder inference
latent_dim=500
# load the model
model = models.load_model("../s2s")
num_in_words = 10344
num_tr_words = 4169
max_in_len = 73
max_tr_len = 17
in_tokenizer = Tokenizer()
# in_tokenizer.fit_on_texts(x_train)
tr_tokenizer = Tokenizer()
# tr_tokenizer.fit_on_texts(y_train)

contractions=pickle.load(open("./contractions.pkl","rb"))['contractions']
stop_words=set(stopwords.words('english'))

#construct encoder model from the output of 6 layer i.e.last LSTM layer
en_outputs,state_h_enc,state_c_enc = model.layers[6].output
en_states=[state_h_enc,state_c_enc]
#add input and state from the layer.
en_model = Model(model.input[0],[en_outputs]+en_states)

# decoder inference
#create Input object for hidden and cell state for decoder
#shape of layer with hidden or latent dimension
dec_state_input_h = Input(shape=(latent_dim,))
dec_state_input_c = Input(shape=(latent_dim,))

dec_hidden_state_input = Input(shape=(max_in_len,latent_dim))

# Get the embeddings and input layer from the model
decoder_inputs_t = model.input[1]  # input_2
dec_inputs =  tf.identity(decoder_inputs_t)
dec_emb_layer = model.layers[5]
dec_lstm = model.layers[7]
dec_embedding= dec_emb_layer(dec_inputs)

#add input and initialize LSTM layer with encoder LSTM states.
dec_outputs2, state_h2, state_c2 = dec_lstm(dec_embedding, initial_state=[dec_state_input_h,dec_state_input_c])

#Attention layer
attention = model.layers[8]
attn_out2 = attention([dec_outputs2,dec_hidden_state_input])

merge2 = Concatenate(axis=-1)([dec_outputs2, attn_out2])

#Dense layer
dec_dense = model.layers[10]
dec_outputs2 = dec_dense(merge2)

# Finally define the Model Class
dec_model = Model(
[dec_inputs] + [dec_hidden_state_input,dec_state_input_h,dec_state_input_c],
[dec_outputs2] + [state_h2, state_c2])

#create a dictionary with a key as index and value as words.
reverse_target_word_index = tr_tokenizer.index_word
reverse_source_word_index = in_tokenizer.index_word
target_word_index = tr_tokenizer.word_index
reverse_target_word_index[0]=' '

def clean(texts,src):    
    texts = BeautifulSoup(texts, "lxml").text
    words=word_tokenize(texts.lower())
    words= list(filter(lambda w:(w.isalpha() and len(w)>=3),words))
    words = [contractions[w] if w in contractions else w for w in words ]
    if src=="inputs":
        words= [stemm.stem(w) for w in words if w not in stop_words]
    else:
        words= [w for w in words if w not in stop_words]
    return words

def decode_sequence(input_seq):
    #get the encoder output and states by passing the input sequence
    en_out, en_h, en_c= en_model.predict(input_seq)

    #target sequence with inital word as 'sos'
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_index['sos']

    #if the iteration reaches the end of text than it will be stop the iteration
    stop_condition = False
    #append every predicted word in decoded sentence
    decoded_sentence = ""
    while not stop_condition: 
        #get predicted output, hidden and cell state.
        output_words, dec_h, dec_c= dec_model.predict([target_seq] + [en_out,en_h, en_c])
        
        #get the index and from the dictionary get the word for that index.
        word_index = np.argmax(output_words[0, -1, :])
        text_word = reverse_target_word_index[word_index]
        decoded_sentence += text_word +" "

        # Exit condition: either hit max length
        # or find a stop word or last word.
        if text_word == "eos" or len(decoded_sentence) > max_tr_len:
          stop_condition = True
        
        #update target sequence to the current word index.
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = word_index
        en_h, en_c = dec_h, dec_c
    
    #return the deocded sentence
    return decoded_sentence


# text_to_summarize = "National Commercial Bank (NCB), Saudi Arabia's largest lender by assets, agreed to buy rival Samba Financial Group for $15 billion in the biggest banking takeover this year.NCB will pay 28.45 riyals ($7.58) for each Samba share, according to a statement on Sunday, valuing it at about 55.7 billion riyals. NCB will offer 0.739 new shares for each Samba share, at the lower end of the 0.736-0.787 ratio the banks set when they signed an initial framework agreement in June.The offer is a 3.5% premium to Samba’s Oct. 8 closing price of 27.50 riyals and about 24% higher than the level the shares traded at before the talks were made public. Bloomberg News first reported the merger discussions.The new bank will have total assets of more than $220 billion, creating the Gulf region’s third-largest lender. The entity’s $46 billion market capitalization nearly matches that of Qatar National Bank QPSC, which is still the Middle East’s biggest lender with about $268 billion of assets."
summarizer = pipeline("summarization")

@app.route("/predict")
def index():
    video = request.args.get('video', 'SX_ykAJLi-k')
    print(video)
    transcript = YouTubeTranscriptApi.get_transcript(video)
    transcript = list(map(lambda x:x["text"], transcript))
    transcript = list(filter(lambda x:x[0]!="(",transcript))
    transcript = " ".join(transcript)
    transcript = transcript[:2000]
    summary = summarizer(transcript)
    return jsonify({
        "videoId": video,
        "text": transcript,
        "summary": summary
    }), 200

@app.route("/test")
def test():
    return "hello"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)