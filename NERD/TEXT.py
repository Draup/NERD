#!/usr/bin/env python
# coding: utf-8

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.stats import entropy
import pickle
import os

from flask import Flask
from flask import request
from jinja2 import Template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, FeatureUnion
import re

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from nltk import pos_tag, word_tokenize
import unicodedata


def get_nth_token(text, n):
    """
    Splits text into tokens and returns the nth token
    Args:
        text: text string
        n: 0 indexed token to return

    Returns:

    """
    toks = re.findall('[\w+\(\),:;\[\]]+', text)
    if len(toks) > n:
        return toks[n]
    else:
        return ''


def cleanup_string(text):
    """
    Basic text Sanitization
    Args:
        text: text to cleanup

    Returns:

    """
    toret = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    toret = toret.strip()
    toret = re.sub('[\r\n\t]+', ' ', toret)
    # toks = re.findall('[\w+\(\),:;\[\]]+', toret)
    # toret = ' '.join(toks)
    toret = re.sub('[^\w+\(\),:;\[\]\-.|& \t\n/]+', ' ', toret)

    return toret


def get_pos_string(text, text_len=100):
    if len(text) < text_len:
        tags = pos_tag(word_tokenize(text))
        tags = [t[1] for t in tags]
        return ' '.join(tags)
    else:
        return ''


def is_alpha_and_numeric(string):
    toret = ''
    if string.isdigit():
        toret = 'DIGIT'
    elif string.isalpha():
        if string.isupper():
            toret = 'ALPHA_UPPER'
        elif string.islower():
            toret = 'ALPHA_LOWER'
        else:
            toret = 'ALPHA'
    elif len(string) > 0:
        toks = [string[0], string[-1]]
        alphanum = 0
        for tok in toks:
            if tok.isdigit():
                alphanum += 1
            elif tok.isalpha():
                alphanum -= 1
        if alphanum == 0:
            toret = 'ALPHA_NUM'
    else:
        toret = 'EMPTY'

    return toret


class ColumnsSelector(TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.cols]


class DropColumns(TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(columns=self.cols)
        return X


class ToDense(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.todense()


class DefaultTextFeaturizer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = pd.DataFrame(data={'text': X})
        data.text = data.text.apply(lambda x: cleanup_string(x))
        data["pos_string"] = data.text.apply(lambda x: get_pos_string(x))
        data['text_feature_text_length'] = data['text'].apply(lambda x: len(x))
        data['text_feature_capitals'] = data['text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
        data['text_feature_digits'] = data['text'].apply(lambda comment: sum(1 for c in comment if c.isdigit()))
        data['text_feature_caps_vs_length'] = data.apply(
            lambda row: row['text_feature_capitals'] / (row['text_feature_text_length'] + 0.001), axis=1)
        data['text_feature_num_symbols'] = data['text'].apply(lambda comment: len(re.findall('\W', comment)))
        data['text_feature_num_words'] = data['text'].apply(lambda comment: len(comment.split()))
        data['text_feature_num_unique_words'] = data['text'].apply(lambda comment: len(set(w for w in comment.split())))
        data['text_feature_words_vs_unique'] = data['text_feature_num_unique_words'] / (
                data['text_feature_num_words'] + 0.001)

        data['text_feature_first_token'] = data['text'].apply(lambda x: is_alpha_and_numeric(get_nth_token(x, 0)))
        data['text_feature_second_token'] = data['text'].apply(lambda x: is_alpha_and_numeric(get_nth_token(x, 1)))
        data['text_feature_third_token'] = data['text'].apply(lambda x: is_alpha_and_numeric(get_nth_token(x, 2)))

        data['text_feature_title_word_count'] = data['text'].apply(lambda x: sum(1 for c in x.split() if c.istitle()))
        data['text_feature_title_word_total_word_ratio'] = data['text_feature_title_word_count'] / (
                data['text_feature_num_words'] + 0.001)
        data['text_feature_numeric_tokens'] = data['text'].apply(lambda x: sum(1 for c in x.split() if c.isdigit()))
        data['text_feature_capital_tokens'] = data['text'].apply(lambda x: sum(1 for c in x.split() if c.isupper()))

        return data.drop(columns=['text'])


class MultiLabelEncoder(TransformerMixin):
    def __init__(self, inplace=False):
        self.inplace = inplace

    def fit(self, X, y=None):
        self.encoder = {}
        self.cols = [c for c in X.columns if X[c].dtype.name == 'object']
        for col in self.cols:
            col_enc = {}
            count = 1
            unique = list(X[col].unique())
            for u in unique:
                col_enc[u] = count
                count += 1
            self.encoder[col] = col_enc

        return self

    def transform(self, X):
        if self.inplace:
            temp = X
        else:
            temp = X.copy()

        for col in self.cols:
            temp[col] = temp[col].apply(lambda x: self.encoder[col].get(x, 0))

        return temp


class BaseTextClassifier:
    """
    A utility class for Text Classification
    """

    def __init__(self, unlabelled, labelled=None, feature_transformer=None, data_directory=''):
        """
        Initialize with a DataFrame(['text']) and/or DataFrame(['text', 'class'])
        Args:
            unlabelled: DataFrame(['text'])
            labelled: DataFrame(['text', 'class'])
            feature_transformer: Sklearn transformer to calculate extra features
            data_directory: Default data directory
        """
        self.unlabelled = pd.DataFrame(data={'text': unlabelled})
        self.labelled = labelled

        if self.labelled is not None:
            self.all_data = pd.concat([self.unlabelled, self.labelled])
        else:
            self.all_data = self.unlabelled
            self.all_data['class'] = np.nan

        # extra feature functions
        if feature_transformer is None:
            self.feature_transformer = DefaultTextFeaturizer()
        else:
            self.feature_transformer = feature_transformer

        self._refresh_text_feature_data()

        self.model = None
        self.data_directory = os.path.join(data_directory, 'Text_Classification_Data')
        os.makedirs(self.data_directory, exist_ok=True)

    def _refresh_text_feature_data(self):
        feature_data = self.feature_transformer.fit_transform(self.all_data['text'])
        self.feature_columns = list(feature_data.columns)
        for col in feature_data.columns:
            self.all_data[col] = feature_data[col]

    def get_new_random_example(self):
        """
        Returns a random example to be tagged. Used to bootstrap the model.
        Returns:

        """
        unl = self.all_data[self.all_data['class'].isna()].index
        self.current_example_index = np.random.choice(unl)
        self.current_example = self.all_data.iloc[self.current_example_index]
        return self.current_example['text']

    def query_new_example(self, mode='entropy'):
        """
        Returns a new example based on the chosen active learning strategy.
        Args:
            mode: Active Learning Strategy
                - max (Default)
                - mean
        Returns:

        """
        if mode == 'entropy':
            unlab = self.all_data[self.all_data['class'].isna()]
            unlabelled_idx = unlab.index
            proba = self.model.predict_proba(unlab)
            proba_idx = np.argmax(entropy(proba.T))
            actual_idx = unlabelled_idx[proba_idx]

            self.current_example_index = actual_idx
            self.current_example = self.all_data.iloc[self.current_example_index]
            return self.current_example['text']

    def update_model(self):
        """
        Updates the model with the currently labelled dataset
        Returns:

        """

        if self.model is None:
            self.model = Pipeline([
                ('fu', FeatureUnion([
                    ('text_vectorizer',
                     make_pipeline(ColumnsSelector('text'), CountVectorizer(ngram_range=(1, 2)), ToDense())),
                    ('text_featurizer', make_pipeline(ColumnsSelector(self.feature_columns), MultiLabelEncoder()))
                ])),
                ('clf', RandomForestClassifier())
            ])

        lab = self.all_data[self.all_data['class'].notna()]
        self.model.fit(lab, lab['class'])

    def save_example(self, data):
        """
        Saves the current example with the user tagged data
        Args:
            data: User tagged data. [list of tags].

        Returns:

        """
        self.all_data.loc[self.current_example_index, 'class'] = data

    def save_data(self, filepath=None):
        """
        Saves the labelled data to a file
        Args:
            filepath: file to save the data in a pickle format.

        Returns:

        """
        if filepath is None:
            filepath = os.path.join(self.data_directory, 'text_classification_data.csv')
        self.all_data[['text', 'class']].to_csv(filepath, index=False)

    def load_data(self, filepath=None):
        """
        Loads labelled data from file.|
        Args:
            filepath: file containing pickeled labelled dataset

        Returns:

        """
        if filepath is None:
            filepath = os.path.join(self.data_directory, 'text_classification_data.csv')
        self.labelled = pd.read_csv(filepath)
        self.all_data = pd.concat([self.all_data, self.labelled])
        self.all_data.reset_index(inplace=True)
        self._refresh_text_feature_data()

    def add_unlabelled_examples(self, examples):
        """
        Append more unlabelled data to dataset
        :param examples: List of strings
        :return:
        """
        new_examples = pd.DataFrame(data={'text': examples})
        self.all_data = pd.concat([self.all_data, new_examples])
        self.all_data.reset_index(inplace=True)
        self._refresh_text_feature_data()


list_of_colors = "#e6194B, #3cb44b, #ffe119, #4363d8, #f58231, #911eb4, #42d4f4, #f032e6, #bfef45, #fabebe, #469990, #e6beff, #9A6324, #fffac8, #800000, #aaffc3, #808000, #ffd8b1, #000075, #a9a9a9"
list_of_colors = list_of_colors.split(', ')


def render_app_template(unique_tags_data):
    """
    Tag data in the form
    [
        (tag_id, readable_tag_name)
    ]
    :param unique_tags_data:
    :return: html template to render
    """

    if len(unique_tags_data) > len(list_of_colors):
        return "Too many tags. Add more colors to list_of_colors"

    trainer_path = os.path.join(os.path.dirname(__file__), 'html_templates',
                                'text_classifier.html')
    with open(trainer_path) as templ:
        template = Template(templ.read())

    css_classes = []
    for index, item in enumerate(unique_tags_data):
        css_classes.append((item[0], list_of_colors[index]))

    return template.render(css_classes=css_classes, id_color_map=css_classes, tag_controls=unique_tags_data)


def get_app(tagger, tags):
    app = Flask(__name__)

    @app.route("/")
    def base_app():
        return render_app_template(tags)

    @app.route('/load_example')
    def load_example():
        if tagger.model is None:
            example = tagger.get_new_random_example()
        else:
            example = tagger.query_new_example(mode='entropy')

        # print(f'Returning example ::: {example[:100]}')

        return example

    @app.route('/update_model')
    def update_model():
        tagger.update_model()
        return "Model Updated Successfully"

    @app.route('/save_example', methods=['POST'])
    def save_example():
        form_data = request.form
        tag = form_data['tag']
        tagger.save_example(tag)
        return 'Success'

    @app.route('/save_data')
    def save_tagged_data():
        print("save_tagged_data")
        tagger.save_data()
        return 'Data Saved'

    return app


class TextClassifier:
    def __init__(self, dataset, unique_tags, data_directory=''):
        """
        need unique tag, tag tiles
        EX:
        tags= [
            ("CT", "Course Title"),
            ("CC", "Course Code"),
            ("PREQ", "Pre-requisites"),
            ("PROF", "Professor"),
            ("SE", "Season"),
            ("CR", "Credits")
        ]
        :param unique_tags:
        """
        self.unique_tags = unique_tags
        self.tagger = BaseTextClassifier(dataset, data_directory=data_directory)
        self.app = get_app(self.tagger, self.unique_tags)
        self.utmapping = {t[0]: t[1] for t in self.unique_tags}

    def start_server(self, port=None):
        """
        Start the ner tagging server
        :param port: Port number to bind the server to.
        :return:
        """
        if port:
            self.app.run(port)
        else:
            self.app.run(port=5050)

    def add_unlabelled_examples(self, examples):
        """
        Append unlabelled examples to dataset
        :param examples: list of strings
        :return:
        """
        self.tagger.add_unlabelled_examples(examples)

    def save_labelled_examples(self, filepath):
        """
        Save labelled examples to file
        :param filepath: destination filename
        :return:
        """
        self.tagger.save_data(filepath)

    def load_labelled_examples(self, filepath):
        """
        Load labelled examples to the dataset
        :param filepath: source filename
        :return:
        """
        self.tagger.load_data(filepath)

    def save_model(self, model_filename):
        """
        Save classifier model to file
        :param model_filename: destination filename
        :return:
        """
        with open(model_filename, 'wb') as out:
            pickle.dump(self.tagger.model, out)

    def load_model(self, model_filename):
        """
        Load classifier model from file
        :param model_filename: source filename
        :return:
        """
        with open(model_filename, 'rb') as inp:
            self.tagger.model = pickle.load(inp)

    def update_model(self):
        """
        Updates the model
        :return:
        """
        self.tagger.update_model()


if __name__ == '__main__':
    # Unique Tags / Classes
    # some text
    tags = [
        ("Address", "Address"),
        ("Other", "Non Address"),
    ]
