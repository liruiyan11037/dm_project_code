import re
import os
import glob
import nltk
from nltk.stem.lancaster import LancasterStemmer
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt


# %%
def get_file_names(root_path: str, dir_name: str, file_name: str, file_type: str) -> list:
    file_list = glob.glob(os.path.join(root_path, dir_name) + '/*' + file_name + '*' + '.' + file_type)
    return file_list


def read_stop_words(file, mode='r', separator='|', encoding='utf-8') -> set:
    result = set()
    if mode == 'rb':
        with open(file, mode=mode) as f:
            for line in f:
                line = line.decode(encoding, 'ignore')
                temp_set = list(map(lambda x: x.strip(' ').lower().replace('\n', ''), line.split(separator)))
                result.update(temp_set)
    else:
        with open(file, encoding=encoding, mode=mode) as f:  #
            for line in f:
                temp_set = list(map(lambda x: x.strip(' ').lower().replace('\n', ''), line.split(separator)))
                result.update(temp_set)
    return result


def read_stop_words_list(file_list, mode='r', separator='|', encoding='utf-8'):
    result = set()
    for f in file_list:
        print(f)
        result.update(read_stop_words(f, mode, separator, encoding))
    return result


class DataProcessor:
    words_dict = dict()
    stopwords = None

    def __init__(self, train_data: pd.Series, train_label: pd.Series, test_data: pd.Series):
        self.data = train_data
        self.label = train_label
        self.test_data = test_data
        self.score = None
        self.test_score = None

    @staticmethod
    def decompose_sentiment_file(sentiment_file_path, extract_file_name):
        data_xls = pd.ExcelFile(sentiment_file_path)
        for name in data_xls.sheet_names:
            df = pd.read_excel(sentiment_file_path, sheet_name=name)
            DataProcessor.words_dict[name] = set(map(lambda x: str(x).lower().strip(), df.iloc[0:, 0].to_list()))
        DataProcessor.words_dict.pop(extract_file_name)

    @staticmethod
    def tokenize(data):
        data = data.map(lambda x: re.sub(r'\d+', '', x))
        data = data.map(lambda x: x.lower())
        data = data.map(nltk.word_tokenize)
        return data.map(lambda x: [w for w in x if w not in DataProcessor.stopwords])
        # stemmer = LancasterStemmer()
        # self.data = self.data.map(lambda x: [stemmer.stem(w) for w in x])

    def tokenize_train(self):
        self.data = DataProcessor.tokenize(self.data)

    def tokenize_test(self):
        self.test_data = DataProcessor.tokenize(self.test_data)

    def expand_lexicon(self):
        positive_set = set()
        negative_set = set()
        for X, y in zip(self.data, self.label):
            if y == 1:
                positive_set.update(X)
            else:
                negative_set.update(X)
        common_set = positive_set.intersection(negative_set)
        positive_set = positive_set - common_set
        negative_set = negative_set - common_set
        for val in DataProcessor.words_dict.values():
            positive_set = positive_set - val
            negative_set = negative_set - val
        DataProcessor.words_dict['Positive'].update(positive_set)
        DataProcessor.words_dict['Negative'].update(negative_set)

    @staticmethod
    def mark_semantic_score(word_list) -> float:
        """
        Definition of the sentiment lexicon: All negative words and adverbs of degree between two emotional words
        and the phrase formed by the latter of these two emotional words.
        e.g. negation_words + degree_word + sentiment_word or degree_word + negation_words + sentiment_word.
        The final sentiment score should = -1^n * (2 or 0.5) * sentiment score (-1 or 1).
        Where n is the number of negation words
        :param word_list: list of words
        :return: float
        """
        total_sentiment = 0.0
        negation_count = 0
        degree_count = 1
        for word in word_list:
            if word in DataProcessor.words_dict['Negation']:
                negation_count += 1
                continue
            if word in DataProcessor.words_dict['StrongModal']:
                degree_count *= 2
                continue
            if word in DataProcessor.words_dict['WeakModal']:
                degree_count *= 0.5
                continue
            if word in DataProcessor.words_dict['Positive']:
                total_sentiment += (-1) ** negation_count * degree_count * 1
                negation_count = 0
                degree_count = 1
                continue
            if word in DataProcessor.words_dict['Negative']:
                total_sentiment += (-1) ** negation_count * degree_count * -1
                negation_count = 0
                degree_count = 1

        return total_sentiment

    def mark_semantic_sentiment(self):
        self.score = self.data.map(DataProcessor.mark_semantic_score)
        self.test_score = self.test_data.map(DataProcessor.mark_semantic_score)


def judge_label(f_x: pd.Series, threshold: float) -> pd.Series:
    return f_x.map(lambda x: -1 if x <= threshold else 1)


def accuracy(f_x: pd.Series, y_true: pd.Series, threshold: float) -> float:
    judged_f_x = judge_label(f_x, threshold)
    return round(sum(judged_f_x == y_true) / len(f_x), 4)


def search_threshold(f_x: pd.Series, y_true: pd.Series) -> list:
    accuracy_list = []
    threshold_list = []
    best_threshold = 0
    best_accuracy = 0
    for threshold in np.array(sorted(f_x)) - 0.5:
        temp_accuracy = accuracy(f_x, y_true, threshold)
        accuracy_list.append(temp_accuracy)
        threshold_list.append(threshold)
        if best_accuracy <= temp_accuracy:
            best_threshold = threshold
            best_accuracy = temp_accuracy
    print(f'The best threshold is {best_threshold} with accuracy of {best_accuracy}')
    return [threshold_list, accuracy_list, best_threshold, best_accuracy]


def draw_accuracy_plot(threshold_list, accuracy_list, title: str):
    fig = plt.figure(1, figsize=(6, 6))
    ax = fig.add_subplot(111)
    plt.title(title)
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.plot(threshold_list, accuracy_list, 'o-', color="blue")
    plt.show()


if __name__ == '__main__':
    root_path = r'G:\Master\master_course\Data Mining\group_project\dm_project_code'
    commas = list(string.punctuation + r'-?-+=_!(){}[]""<>,./' + 'abcdefghijklmnopqrstuvwxyz')
    negations = ['no', 'not', 'none', 'neither', 'nobody']
    DataProcessor.stopwords = set(nltk.corpus.stopwords.words('english'))
    DataProcessor.stopwords.update(commas)
    Stopwords_paths = get_file_names(root_path, 'dictionary', 'StopWords', 'txt')
    DataProcessor.stopwords.update(read_stop_words_list(Stopwords_paths))
    train_path = get_file_names(root_path, 'data', 'train-sample', 'xlsx')[0]
    sentiment_path = get_file_names(root_path, 'dictionary', 'Lough', 'xlsx')[0]
    train = pd.read_excel(train_path, sheet_name="Sheet1")
    test_path = get_file_names(root_path, 'data', 'test-sample', 'xlsx')[0]
    test = pd.read_excel(test_path, sheet_name="Sheet1")
    test_X = test['title']
    train_X = train['title']
    train_y = train['sentiment']
    task1_processor = DataProcessor(train_X, train_y, test_X)
    DataProcessor.decompose_sentiment_file(sentiment_path, 'Documentation')
    DataProcessor.words_dict['Negation'] = set(negations)
    task1_processor.tokenize_train()
    task1_processor.tokenize_test()
    task1_processor.expand_lexicon()
    task1_processor.mark_semantic_sentiment()
    print(task1_processor.data)
    print(task1_processor.score)
    train_threshold_list, train_accuracy_list, train_threshold, train_accuracy = search_threshold(task1_processor.score,
                                                                                                  task1_processor.label)
    draw_accuracy_plot(train_threshold_list, train_accuracy_list, 'train sample accuracy plot')
    test_label = judge_label(task1_processor.test_score, train_threshold)
    test_submit_path = get_file_names(root_path, 'data', 'test-submit-sample', 'xlsx')[0]
    test_submit = pd.read_excel(test_submit_path, sheet_name="Sheet1")
    test_submit['predicted_sentiment'] = test_label
    print(test_label)
    test_submit.to_excel(test_submit_path, sheet_name='Sheet1')
