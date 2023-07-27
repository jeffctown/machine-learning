import tensorflow as tf
import numpy as np
import csv
from bs4 import BeautifulSoup
import string
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping


class TwitterSentamentAnalyzer:
    SENTIMENT_THRESHOLDS = (0.4, 0.7)

    def __init__(self,
                 validation_data_file='training-data/twitter_validation.csv',
                 training_data_file='training-data/twitter_training.csv',
                 max_sequence_length=10,
                 truncate='post',
                 oov_token='<OOV>',
                 max_vocab_size=20000,
                 max_epochs=100,
                 embedding_dim=6,
                 positive_label='Positive',
                 neutral_label='Neutral',
                 negative_label='Negative'):
        print(f'Initializing {self.__class__}')
        self.validation_data_file = validation_data_file
        self.training_data_file = training_data_file
        self.max_sequence_length = max_sequence_length
        self.truncate = truncate
        self.oov_token = oov_token
        self.max_vocab_size = max_vocab_size
        self.max_epochs = 100
        self.embedding_dim = embedding_dim
        self.positive_label = positive_label
        self.neutral_label = neutral_label
        self.negative_label = negative_label
        self.training_sentences = []
        self.training_labels = []
        self.validation_sentences = []
        self.validation_labels = []

    def process_training_data(self):
        (self.validation_sentences, self.validation_labels) = self.process_data(self.training_data_file)

    def process_validation_data(self):
        (self.training_sentences, self.training_labels) = self.process_data(self.validation_data_file)

    def prepare(self):
        nltk.download('stopwords')
        self.stop_words = stopwords.words("english")
        print(f'Loaded {len(self.stop_words)} stop words.')

    def process_data(self, filename, print_logs = True, print_sentences = False):
        sentences = []
        labels = []
        if print_logs:
            print(f'Processing data for file: {filename}')
        table = str.maketrans('', '', string.punctuation)
        with open(filename,  encoding='UTF-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                label = row[2]
                sentence = row[3].lower()
                sentence = sentence.replace(",", " , ")
                sentence = sentence.replace(".", " . ")
                sentence = sentence.replace("-", " - ")
                sentence = sentence.replace("/", " / ")
                soup = BeautifulSoup(sentence, features="html.parser")
                sentence = soup.get_text()

                words = sentence.split()
                filtered_sentence = ""
                filtered_words = []
                for word in words:
                    word = word.translate(table)
                    if word not in self.stop_words:
                        filtered_words.append(word)
                if len(filtered_words) > 0:
                    filtered_sentence = " ".join(filtered_words)
                    label = self.encode_label(label)
                    if label is not None:
                        sentences.append(filtered_sentence)
                        labels.append(label)
                        if print_sentences:
                            print(f'{label}: {filtered_sentence}')
        if print_logs:
            print(f'Done processing data for file: {filename}')
        return (sentences, labels)

    def encode_label(self, label):
        if label == self.positive_label:
            return 1.0
        elif label == self.neutral_label:
            return 0.0
        elif label == self.negative_label:
            return -1.0
        return None

    def tokenize(self):
        self.tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token=self.oov_token)
        self.tokenizer.fit_on_texts(self.training_sentences)

        training_sequences = self.tokenizer.texts_to_sequences(self.training_sentences)
        self.training_padded = pad_sequences(training_sequences, maxlen=self.max_sequence_length, padding=self.truncate, truncating=self.truncate)

        validation_sequences = self.tokenizer.texts_to_sequences(self.validation_sentences)
        self.validation_padded = pad_sequences(validation_sequences, maxlen=self.max_sequence_length, padding=self.truncate, truncating=self.truncate)

    def train_model(self):
        self.training_padded = np.array(self.training_padded)
        self.training_labels = np.array(self.training_labels)
        self.validation_padded = np.array(self.validation_padded)
        self.validation_labels = np.array(self.validation_labels)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.max_vocab_size, self.embedding_dim),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        self.model.summary()

        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
            EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)
        ]
        self.history = self.model.fit(self.training_padded, self.training_labels, epochs=self.max_epochs, validation_data=(self.validation_padded, self.validation_labels), verbose=2, callbacks=callbacks)

    def train(self):
        self.prepare()
        self.process_training_data()
        self.process_validation_data()
        self.tokenize()
        self.train_model()

    def predict(self, sentence, include_neutral=True):
        x_test = pad_sequences(self.tokenizer.texts_to_sequences([sentence]), maxlen=self.max_sequence_length)
        score = self.model.predict([x_test])[0]
        label = self.decode_sentiment(score, include_neutral=include_neutral)
        print(f'"label": {label}, "score": {float(score)}')

    def decode_sentiment(self, score, include_neutral=True):
        if include_neutral:
            label = self.negative_label
            if score <= self.SENTIMENT_THRESHOLDS[0]:
                label = self.negative_label
            elif score >= self.SENTIMENT_THRESHOLDS[1]:
                label = self.negative_label

            return label
        else:
            return self.negative_label if score < 0.5 else self.negative_label

    def plot_training_history(self):
        print(f'{self.history.history.keys()}')
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'b', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()

def main():
    analyzer = TwitterSentamentAnalyzer()
    analyzer.train()
    analyzer.predict("I hate the rain")
    analyzer.predict("I love sunny days")
    analyzer.predict("Turkey is a meat")
    analyzer.plot_training_history()

if __name__ == "__main__":
    main()