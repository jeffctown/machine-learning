import tensorflow as tf
import collections.abc
#tensorflow_datasets needs the following alias to be done manually.
collections.Iterable = collections.abc.Iterable
import tensorflow_datasets as tfds
from tensorflow import keras
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup


#
# test_data = [
#     'Today is a snowy day',
#     'Will it be rainy tomorrow?'
# ]

# padding
# padded = pad_sequences(sequences, padding='post', maxlen=6, truncating='post')
# for sequence in padded:
#     print(sequence)


imdb_sentences = []
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split='train'))
for item in train_data:
    imdb_sentences.append(str(item['text'].decode('UTF-8')))

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
             "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
             "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should",
             "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
             "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were",
             "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why",
             "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
             "yourselves"]

def clean(sentences):
    table = str.maketrans('', '', string.punctuation)
    filtered_sentences = []
    for sentence in sentences:
        # use beautiful soup to remove HTML elements (like <br>)
        soup = BeautifulSoup(sentence, features="html.parser")
        sentence = soup.get_text()
        sentence = sentence.replace(',', ' , ')
        sentence = sentence.replace('.', ' . ')
        sentence = sentence.replace('-', ' - ')
        sentence = sentence.replace('/', ' / ')
        words = sentence.split()
        filtered_sentence = ""
        for word in words:
            # lower case
            word = word.lower()
            # remove punctuation from words
            word = word.translate(table)
            # remove all stopwords
            if word not in stopwords:
                filtered_sentence = filtered_sentence + word + " "
        filtered_sentences.append(filtered_sentence)
    return filtered_sentences


tokenizer = Tokenizer(num_words=25000)
tokenizer.fit_on_texts(imdb_sentences)
word_index = tokenizer.word_index

imdb_sentences = clean(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)
print(f'words in index: {len(word_index)}')

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?'
]
sentences = clean(sentences)
my_sequences = tokenizer.texts_to_sequences(sentences)
print(my_sequences)

reverse_word_index = dict([(value, key) for (key, value) in tokenizer.word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i, '?') for i in my_sequences[0]])
print(decoded_review)

# using subwords
(train_data, test_data), info = tfds.load('imdb_reviews/subwords8k',
                                          split=(tfds.Split.TRAIN, tfds.Split.TEST),
                                          as_supervised=True,
                                          with_info=True)

encoder = info.features['text'].encoder
print(f'Vocabulary size: {encoder.vocab_size}')
