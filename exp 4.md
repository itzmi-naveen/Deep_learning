
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import numpy as np

print("TensorFlow version:", tf.__version__)

data = """Deep learning is amazing. 
Deep learning builds intelligent systems."""

corpus = data.lower().split("\n")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print("Total unique words:", total_words)

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        ngram_seq = token_list[:i+1]
        input_sequences.append(ngram_seq)


max_seq_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding="pre")

X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

print("Max sequence length:", max_seq_len)


model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_seq_len-1))
model.add(SimpleRNN(100))
model.add(Dense(total_words, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

history = model.fit(X, y, epochs=500, verbose=1)


def predict_next_word(seed_text, next_words=1):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding="pre")
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
        
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                seed_text += " " + word
                break
    return seed_text
seed_text = "Deep learning"
for i in range(7):
    new_text = predict_next_word(seed_text, 1)
    print(f"{seed_text!r} -> '{new_text.split()[-1]}'")
    seed_text = new_text
<img width="866" height="169" alt="image" src="https://github.com/user-attachments/assets/0edcbf3d-20fd-417f-b976-e238a346f409" />
