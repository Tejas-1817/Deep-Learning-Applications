#########################################################################################
# Required Python Packages
#########################################################################################
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

#########################################################################################
# Function Name : MarvellousSentimentAnalysis
# Description   : Perform basic Sentiment Analysis using Recurrent Neural Network (RNN)
# Input         : Small dataset of text sentences with labels (Positive = 1, Negative = 0)
# Output        : Train RNN model, predict sentiment of new sentences
# Author        : Tejas Khandu Vatane
# Date          : 29/09/2025
#########################################################################################
def MarvellousSentimentAnalysis():
    # 1) Tiny dataset (Positive = 1, Negative = 0)
    sentences = [
        "I Love movie",
        "This film was great",
        "What a fantastic experiance",
        "I really enjoyed it",
        "Absolutely wonderful acting",
        "I hate this movie",
        "this film was terible",
        "what a bad experiance",
        "I really disliked it",
        "Absolutely horrible acting"
    ]

    print("Input dataset : ")
    print(sentences)

    labels = [1,1,1,1,1,0,0,0,0,0]   # Sentiment labels: 1 = Positive, 0 = Negative

    #########################################################################################
    # Step 2 : Text Tokenization
    # Convert words into integer indices using Keras Tokenizer
    # Keep top 50 most frequent words
    #########################################################################################
    tokenizer = Tokenizer(num_words = 50)
    tokenizer.fit_on_texts(sentences)
    X = tokenizer.texts_to_sequences(sentences)
    print("Word Index : ",tokenizer.word_index)   # Show vocabulary mapping

    #########################################################################################
    # Step 3 : Padding Sequences
    # All input sequences must have same length for RNN
    # Pad shorter sequences with zeros
    #########################################################################################
    maxlen = 5
    X = pad_sequences(X, maxlen = maxlen)
    y = np.array(labels)

    #########################################################################################
    # Step 4 : Build Simple RNN Model
    # - Embedding Layer: Convert integer words into dense vectors
    # - SimpleRNN Layer: Capture sequential dependencies
    # - Dense Layer: Output probability (0 = Negative, 1 = Positive)
    #########################################################################################
    model = Sequential()
    model.add(Embedding(input_dim = 50, output_dim = 8, input_length = maxlen))  # Word embeddings
    model.add(SimpleRNN(8, activation = "tanh"))  # Recurrent layer
    model.add(Dense(1, activation = "sigmoid"))   # Binary classification output

    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

    #########################################################################################
    # Step 5 : Train Model
    # Train for 30 epochs on given dataset
    #########################################################################################
    model.fit(X, y, epochs = 30, verbose = 0)

    #########################################################################################
    # Step 6 : Test Model on New Examples
    # Provide new sentences and predict sentiment
    #########################################################################################
    test_sentences = ["I enjoyed this film", "I hated this film"]
    test_seq = tokenizer.texts_to_sequences(test_sentences)
    test_seq = pad_sequences(test_seq, maxlen = maxlen)

    pred = model.predict(test_seq)

    for s, p in zip(test_sentences, pred):
        print(f"Sentence : {s} -> Sentiment : ", "Positive" if p > 0.5 else "Negative")


#########################################################################################
# Function Name : main
# Description   : Entry point of application. Calls Sentiment Analysis function
# Input         : None
# Output        : Displays dataset, vocabulary, predictions of test sentences
# Author        : Tejas Khandu Vatane
# Date          : 29/09/2025
#########################################################################################
def main():
    MarvellousSentimentAnalysis()


#########################################################################################
# Application Starter
#########################################################################################
if __name__ == "__main__":
    main()
