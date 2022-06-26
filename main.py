import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten, Activation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os
from itertools import chain


def read_text(file_name):
    # text stream
    read_file = open(file_name, 'r', encoding='utf-8')
    text = read_file.read()
    text = text.replace("\n", " ")

    return text


writers = ["Беляев", "Булгаков", "Васильев", "Гоголь", "Гончаров", "Горький", "Грибоедов",
           "Достоевский", "Каверин", "Катаев", "Куприн", "Лермонтов", "Лесков", "Носов",
           "Пастернак", "Пушкин", "Толстой", "Тургенев", "Чехов", "Шолохов"]

num_classes = len(writers)

texts_list = []

for j in os.listdir('./assets/'):
    print(j)
    texts_list.append(read_text('./assets/' + j))

    print(j, 'added to dataset')

texts_len = [len(text) for text in texts_list]
train_len_shares = [(current_text_len - round(current_text_len / 5)) for current_text_len in texts_len]
t_num = 0
print('text datasets sizes')

for text_len in texts_len:
    t_num += 1
    print(f'text length №{t_num}: {text_len}')

t_num = 0
for train_len_share in train_len_shares:
    t_num += 1
    print(f'text test length №{t_num}: {train_len_share}')

train_data = []
test_data = []

for i in range(len(texts_list)):
    train_len = int(len(texts_list[i]) * 0.8)

    train_data = list(chain(train_data, ([texts_list[i][:train_len]])))
    test_data = list(chain(test_data, ([texts_list[i][train_len:]])))


def log_data_info(data, data_label):
    print(f'number of elements {data_label}: {len(data)}')
    print(f'sample {data_label} data type: {type(data)}')
    print(f'type of first element from {data_label}: {type(data[0])}')
    print(f'length of first element from {data_label} (tokens): {len(data[0])}')
    print(f'excerpt from the {data_label}: {data[0][:26]}')


log_data_info(train_data, 'train data')
log_data_info(test_data, 'test data')

# max words number
maxWordsCount = 10000

# tokenizer  model
tokenizer = Tokenizer(num_words=maxWordsCount,  # Максимальное кол-во слов
                      filters='!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff',  # Фильтры исходного текста
                      lower=True, split=' ',  # Все буквы к нижнему регистру, разделение слов пробелом
                      oov_token='unknown',  # Один лейбл для всех незнакомых слов
                      char_level=False)  # Без выравнивания символов

# initialize words vocabulary based on its frequency
tokenizer.fit_on_texts(train_data)
tokenizer.fit_on_texts(test_data)

train_sequence = tokenizer.texts_to_sequences(train_data)
test_sequence = tokenizer.texts_to_sequences(test_data)


def log_sequence_info(sequence, sequence_label):
    print(f'number of all {sequence_label} indices:  {len(sequence[0])}')
    print(f'first 15 indices from {sequence_label}: {sequence[0][:15]}')


log_sequence_info(train_sequence, 'train sequence')
log_sequence_info(test_sequence, 'test sequence')


def split_sequence(sequence,
                   win_size,
                   hop):

    return [sequence[i:i + win_size] for i in range(0, len(sequence) - win_size + 1, hop)]


def vectorize_sequence(seq_list,  # array of files with array of word indexes
                       win_size,
                       hop):

    # number of class files
    class_count = len(seq_list)

    x, y = [], []

    for cls in range(class_count):
        # words sample
        vectors = split_sequence(seq_list[cls], win_size, hop)
        x += vectors

        # OHE output format
        y += [utils.to_categorical(cls, class_count)] * len(vectors)

    return np.array(x), np.array(y)


sample_len = 1000

step = 500

x_train, y_train = vectorize_sequence(train_sequence, sample_len, step)
x_test, y_test = vectorize_sequence(test_sequence, sample_len, step)

# data dimension
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# final samples prepare
x_train_BoW = tokenizer.sequences_to_matrix(x_train.tolist())
x_test_BoW = tokenizer.sequences_to_matrix(x_test.tolist())

modelBoW = Sequential()  # Bag of Words
modelBoW.add(BatchNormalization(input_dim=maxWordsCount))
modelBoW.add(Dense(80, activation="relu"))
modelBoW.add(Dropout(0.6))
modelBoW.add(Dense(20, activation="relu"))
modelBoW.add(Dense(20, activation='sigmoid'))

modelBoW.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
modelBoW.summary()

# Bag of Words
history = modelBoW.fit(x_train_BoW,
                       y_train,
                       epochs=20,
                       batch_size=64,
                       validation_data=(x_test_BoW, y_test))


plt.plot(history.history['accuracy'],
         label='correct response rate on train sample')
plt.plot(history.history['val_accuracy'],
         label='correct response rate on test sample')
plt.xlabel('Epoch')
plt.ylabel('Correct response rate')
plt.legend()
plt.show()
