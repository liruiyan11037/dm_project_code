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
    file_names = glob.glob(os.path.join(root_path, dir_name) + '/*' + file_name + '*' + '.' + file_type)
    return file_names

def get_file_path(root_path: str, dir_name: str, file_name: str, file_type: str) -> str:
    file_path = os.path.join(root_path, dir_name) + '\\' + file_name + '.' + file_type
    return file_path

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
        # print(f)
        result.update(read_stop_words(f, mode, separator, encoding))
    return result

def getTopK(dict_, k):
    keys = list(dict_.keys())
    for key_i in range(len(keys)):
        for key_j in range(key_i+1, len(keys)):
            if dict_[keys[key_i]] < dict_[keys[key_j]]:
                temp = keys[key_i]
                keys[key_i] = keys[key_j]
                keys[key_j] = temp

    result = dict()
    for i in range(k):
        if len(keys) > i:
            result[keys[i]] = dict_[keys[i]]
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
        # stemmer = LancasterStemmer()
        # data = data.map(lambda x: [stemmer.stem(w) for w in x])
        return data.map(lambda x: [w for w in x if w not in DataProcessor.stopwords])
        
    def filtered(data):  # eliminate the word that are too short (very likely to be meaningless)
        return data.map(lambda x: [w for w in x if len(w) >= 3])

    def tokenize_train(self):
        self.data = DataProcessor.tokenize(self.data)
        self.data = DataProcessor.filtered(self.data)

    def tokenize_test(self):
        self.test_data = DataProcessor.tokenize(self.test_data)
        self.test_data = DataProcessor.filtered(self.test_data)

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

    def expand_lexicon_top_k(self, k = 20):
        positive_dict = dict()
        negative_dict = dict()
        common_list = list()
        for X, y in zip(self.data, self.label):
            for word in X:
                if y == 1:
                    positive_dict[word] = positive_dict.get(word,0) + 1
                else:
                    negative_dict[word] = negative_dict.get(word,0) + 1
        for pos_word in positive_dict.keys():
            if pos_word in negative_dict.keys():
                common_list.append(pos_word)
        for word in common_list:
            del positive_dict[word]
            del negative_dict[word]
        positive_dict_top_k = getTopK(positive_dict, k)
        negative_dict_top_k = getTopK(negative_dict, k)
        positive_set = set(positive_dict_top_k)
        negative_set = set(negative_dict_top_k)
        new_common_set = positive_set.intersection(negative_set)
        positive_set = positive_set - new_common_set
        negative_set = negative_set - new_common_set
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

    def mark_semantic_score2(word_list) -> float:
        """
        Another method to calculate sentiment score based on lexicon
        """
        total_sentiment = 0.0
        for i in range(len(word_list)):
            if word_list[i] in DataProcessor.words_dict['Negative']:
                if i>0 and word_list[i-1] in DataProcessor.words_dict['Negation']:
                    total_sentiment += 1
                elif i>0 and word_list[i-1] in DataProcessor.words_dict['StrongModal']:
                    total_sentiment -= 2
                else: total_sentiment -= 1
            elif word_list[i] in DataProcessor.words_dict['Positive']:
                if i>0 and word_list[i-1] in DataProcessor.words_dict['Negation']:
                    total_sentiment -= 1
                elif i>0 and word_list[i-1] in  DataProcessor.words_dict['StrongModal']:
                    total_sentiment += 2
                elif i>0 and word_list[i-1] in DataProcessor.words_dict['Negative']:
                    total_sentiment -= 1
                elif i<len(word_list)-1 and word_list[i+1] in DataProcessor.words_dict['Negative']:
                    total_sentiment -= 1
                else: total_sentiment += 1
            elif word_list[i] in DataProcessor.words_dict['Negation']:
                total_sentiment -= 0.5
        return total_sentiment


    def mark_semantic_sentiment(self):        
        self.score = self.data.map(DataProcessor.mark_semantic_score)
        self.test_score = self.test_data.map(DataProcessor.mark_semantic_score)
        
        # self.score = self.data.map(DataProcessor.mark_semantic_score2)
        # self.test_score = self.test_data.map(DataProcessor.mark_semantic_score2)


def judge_label(f_x: pd.Series, threshold: float) -> pd.Series:
    return f_x.map(lambda x: -1 if x <= threshold else 1)

def calculate_recall(f_x: pd.Series, y_true: pd.Series, threshold: float) -> float:
    y_predicted = judge_label(f_x, threshold)
    return sum((y_predicted == 1) & (y_true == 1)) / sum(y_true == 1)

def calculate_precision(f_x: pd.Series, y_true: pd.Series, threshold: float) -> float:
    y_predicted = judge_label(f_x, threshold)
    return sum((y_predicted == 1) & (y_true == 1)) / sum(y_predicted == 1)
    
def calculate_F1(f_x: pd.Series, y_true: pd.Series, threshold: float) -> float:
    recall = calculate_recall(f_x, y_true, threshold)
    precision = calculate_precision(f_x, y_true, threshold)
    return 2*recall*precision / (recall+precision)

def calculate_accuracy(f_x: pd.Series, y_true: pd.Series, threshold: float) -> float:
    y_predicted = judge_label(f_x, threshold)
    return sum(y_predicted == y_true) / len(f_x)

def search_threshold(f_x: pd.Series, y_true: pd.Series) -> list:
    f1_score_list = []
    threshold_list = []
    best_threshold = 0
    best_f1_score = 0
    for threshold in np.array(sorted(f_x)) - 0.5:
        temp_f1_score = calculate_F1(f_x, y_true, threshold)
        f1_score_list.append(temp_f1_score)
        threshold_list.append(threshold)
        if best_f1_score <= temp_f1_score:
            best_threshold = threshold
            best_f1_score = temp_f1_score
    print(f'The best threshold is {best_threshold} with F1-score on the training set to be {best_f1_score}.')
    return [threshold_list, f1_score_list, best_threshold, best_f1_score]


def draw_f1_score_plot(threshold_list, f1_score_list, title: str):
    fig = plt.figure(1, figsize=(6, 6))
    plt.title(title)
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.plot(threshold_list, f1_score_list, 'o-', color="blue")
    plt.show()

def result_to_dataframe(predicted_label: pd.Series) -> pd.DataFrame:
    dict_predicted = {'rid':predicted_label.index,'predicted_label':predicted_label.values}
    df_predicted = pd.DataFrame(dict_predicted)
    return df_predicted

if __name__ == '__main__':
    root_path = r'C:\PythonClass\term 2\MDS5724 Data Mining\group project\dm_project_code'
    commas = list(string.punctuation + r'`-?-+=_!(){}[]""<>,./' + 'abcdefghijklmnopqrstuvwxyz')
    commas.extend(["``","'s"])
    negations = ['no', 'not', 'none', 'neither', 'nobody']
    DataProcessor.stopwords = set(nltk.corpus.stopwords.words('english'))
    DataProcessor.stopwords.update(commas)
    Stopwords_paths = get_file_names(root_path, 'dictionary', 'StopWords', 'txt')
    DataProcessor.stopwords.update(read_stop_words_list(Stopwords_paths))
    
    train_path = get_file_path(root_path, 'data', 'train', 'xlsx')
    sentiment_path = get_file_names(root_path, 'dictionary', 'Lough', 'xlsx')[0]
    test_path = get_file_path(root_path, 'data', 'test', 'xlsx')
    
    train = pd.read_excel(train_path, sheet_name="train", engine='openpyxl')
    train = train.sample(frac=1, random_state = 0).reset_index(drop=True)
    train_size = 0.7*len(train)
    train_X = train.loc[:train_size,'text']
    train_y_true = train.loc[:train_size:,'label']
    validation_X = train.loc[train_size:,'text']
    validation_y_true = train.loc[train_size:,'label']
    test = pd.read_excel(test_path, sheet_name="test_full", engine='openpyxl')
    test_X = test['text']
    
    task1_processor = DataProcessor(train_X, train_y_true, validation_X)
    DataProcessor.decompose_sentiment_file(sentiment_path, 'Documentation')
    DataProcessor.words_dict['Negation'] = set(negations)
    task1_processor.tokenize_train()
    task1_processor.tokenize_test()
    task1_processor.expand_lexicon_top_k(100)
    task1_processor.mark_semantic_sentiment()
    
    print('-----------------------------------------Model 1-----------------------------------------\n')
    train_threshold_list, train_f1_score_list, train_threshold, train_f1_score = search_threshold(task1_processor.score,
                                                                                                  task1_processor.label)
    draw_f1_score_plot(train_threshold_list, train_f1_score_list, 'train set F1-score plot')
    task1_F1_score_validation = calculate_F1(task1_processor.test_score, validation_y_true, train_threshold)
    print('F1-score on the validation set: {}. \n'.format(task1_F1_score_validation))

    # test_y_predicted = judge_label(task1_processor.test_score, train_threshold)
    # task1_test_result = result_to_dataframe(test_y_predicted)
    # task1_test_result_path = get_file_path(root_path, 'data', 'test_to-submit-model-1', 'xlsx')
    # task1_test_result.to_excel(task1_test_result_path, index=False)
    print('-----------------------------------------------------------------------------------------\n')
