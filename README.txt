Project Title:- Sentiment Analysis using Python (Recurrent Neural Network)

This project demonstrates Sentiment Analysis using Python and TensorFlow/Keras.
It trains a simple Recurrent Neural Network (RNN) on a small dataset of positive and negative sentences and predicts whether new sentences express a positive or negative sentiment.


Goal:
  Preprocess text data (tokenization and padding).
  Build and train an RNN model on labeled sentences.
  Predict sentiment (Positive/Negative) of new sentences.


Core Idea:
  Use Tokenizer to convert text into integer sequences.
  Apply Padding so all sequences have equal length.
  Build a Sequential model with:
    Embedding Layer → convert words into dense vectors.
    SimpleRNN Layer → capture sequential meaning of sentences.
    Dense Layer → predict sentiment (binary output: 0 = Negative, 1 = Positive).
    Train the model and test it with new inputs.


Output:
  Console prints showing:
    Input dataset
    Word index mapping (vocabulary)
    Predictions for test sentences


Description:
  In this project, Python and TensorFlow/Keras are used to build a basic NLP pipeline for sentiment analysis.



Workflow:
  A tiny dataset of 10 sentences (5 positive, 5 negative) is used.
  Sentences are tokenized into integers and padded to equal length.
  An RNN model is trained with embeddings and sequential processing.
  After training, the model can classify new unseen sentences.



This project helps you understand:
  How text tokenization works in NLP.
  How to use embeddings to represent words numerically.
  How RNNs capture sequence patterns in language.
  How to train and test deep learning models in Python.


Dependencies:
  Before running the project, install the required Python packages:
    numpy → For handling arrays and labels.
    tensorflow/keras → For tokenization, sequence padding, and building the RNN model.


Source Code:
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

def MarvellousSentimentAnalysis():
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

    labels = [1,1,1,1,1,0,0,0,0,0]

    tokenizer = Tokenizer(num_words = 50)
    tokenizer.fit_on_texts(sentences)
    X = tokenizer.texts_to_sequences(sentences)
    X = pad_sequences(X, maxlen = 5)
    y = np.array(labels)

    model = Sequential()
    model.add(Embedding(input_dim = 50, output_dim = 8, input_length = 5))
    model.add(SimpleRNN(8, activation = "tanh"))
    model.add(Dense(1, activation = "sigmoid"))
    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    model.fit(X, y, epochs = 30, verbose = 0)

    test_sentences = ["I enjoyed this film", "I hated this film"]
    test_seq = tokenizer.texts_to_sequences(test_sentences)
    test_seq = pad_sequences(test_seq, maxlen = 5)
    pred = model.predict(test_seq)

    for s, p in zip(test_sentences, pred):
        print(f"Sentence : {s} -> Sentiment : ", "Positive" if p > 0.5 else "Negative")

def main():
    MarvellousSentimentAnalysis()

if __name__ == "__main__":
    main()


Explanation of Project & Workflow:
  Tokenizer
  Converts words into integer indices, creating a vocabulary mapping.

  Pad Sequences
  Ensures all input sequences have the same length before feeding into RNN.

  Build Model
  Embedding → Dense word representation.
  SimpleRNN → Reads sequences and captures dependencies.
  Dense → Outputs sentiment probability.

  Training
  Model learns positive and negative sentence patterns.
  Prediction
  Given new sentences, the trained model outputs Positive or Negative sentiment.

Author

  Tejas Khandu Vatane
  Date: 29/09/2025
  Day: Monday