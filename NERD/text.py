#!/usr/bin/env python
# coding: utf-8

import pprint
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.stats import entropy
import pickle
import os
import json
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
from sklearn.preprocessing import MultiLabelBinarizer


def get_nth_token(text, n):
    toks = re.findall('[\w+\(\),:;\[\]]+', text)
    if len(toks) > n:
        return toks[n]
    else:
        return ''


def cleanup_string(text):
    toret = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('ascii')
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
        data['text_feature_capitals'] = data['text'].apply(
            lambda comment: sum(1 for c in comment if c.isupper()))
        data['text_feature_digits'] = data['text'].apply(
            lambda comment: sum(1 for c in comment if c.isdigit()))
        data['text_feature_caps_vs_length'] = data.apply(
            lambda row: row['text_feature_capitals'] / (row['text_feature_text_length'] + 0.001), axis=1)
        data['text_feature_num_symbols'] = data['text'].apply(
            lambda comment: len(re.findall('\W', comment)))
        data['text_feature_num_words'] = data['text'].apply(
            lambda comment: len(comment.split()))
        data['text_feature_num_unique_words'] = data['text'].apply(
            lambda comment: len(set(w for w in comment.split())))
        data['text_feature_words_vs_unique'] = data['text_feature_num_unique_words'] / (
            data['text_feature_num_words'] + 0.001)

        data['text_feature_first_token'] = data['text'].apply(
            lambda x: is_alpha_and_numeric(get_nth_token(x, 0)))
        data['text_feature_second_token'] = data['text'].apply(
            lambda x: is_alpha_and_numeric(get_nth_token(x, 1)))
        data['text_feature_third_token'] = data['text'].apply(
            lambda x: is_alpha_and_numeric(get_nth_token(x, 2)))

        data['text_feature_title_word_count'] = data['text'].apply(
            lambda x: sum(1 for c in x.split() if c.istitle()))
        data['text_feature_title_word_total_word_ratio'] = data['text_feature_title_word_count'] / (
            data['text_feature_num_words'] + 0.001)
        data['text_feature_numeric_tokens'] = data['text'].apply(
            lambda x: sum(1 for c in x.split() if c.isdigit()))
        data['text_feature_capital_tokens'] = data['text'].apply(
            lambda x: sum(1 for c in x.split() if c.isupper()))

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

    def __init__(self,
                 all_data,
                 text_col='text',
                 label_col='class',
                 multilabel=False,
                 feature_transformer=None,
                 display_function=None,
                 data_directory=''):
        """
        Initialize with a DataFrame(['text']) and/or DataFrame(['text', 'class'])
        :param unlabelled: DataFrame(['text'])
        :param labelled: DataFrame(['text', 'class'])
        """
        if type(all_data) == list:
            self.all_data = pd.DataFrame(data={
                'text': all_data
            })
        else:
            self.all_data = all_data

        self.label_col = label_col
        self.text_col = text_col
        self.multilabel = multilabel
        if self.label_col not in self.all_data:
            self.all_data[self.label_col] = np.nan

        # extra feature functions
        if feature_transformer == 'default':
            self.feature_transformer = DefaultTextFeaturizer()
        elif feature_transformer is not None:
            self.feature_transformer = feature_transformer
        else:
            self.feature_transformer = None

        # self._refresh_text_feature_data()

        self.model = None
        self.display_function = display_function
        self.data_directory = os.path.join(
            data_directory, 'Text_Classification_Data')
        os.makedirs(self.data_directory, exist_ok=True)

    # def _refresh_text_feature_data(self):
    #
    #     feature_data = self.feature_transformer.fit_transform(self.all_data['text'])
    #     self.feature_columns = list(feature_data.columns)
    #     for col in feature_data.columns:
    #         self.all_data[col] = feature_data[col]

    def default_display_function(self, example):
        data = {
            'text': example[self.text_col],
        }
        dump = json.dumps(data, indent=2)

        return dump

    def generate_view(self, example):
        if self.display_function:
            return self.display_function(example)
        else:
            return self.default_display_function(example)

    def get_new_random_example(self):
        """
        Returns a random example to be tagged. Used to bootstrap the model.
        :return:
        """
        unl = self.all_data[self.all_data[self.label_col].isna()].index
        current_example_index = np.random.choice(unl)
        current_example = self.all_data.loc[current_example_index]

        toret = {
            'example_index': int(current_example_index),
            'view': self.generate_view(current_example)
        }
        if self.model:
            preds = self.model.predict(
                self.all_data.loc[current_example_index: current_example_index+1])
            if self.multilabel:
                preds = list(
                    self.multilabel_binarizer.inverse_transform(preds)[0])
                toret['predictions'] = preds
            else:
                preds = [preds[0]]
                toret['predictions'] = preds

        return toret

    def query_new_example(self, mode='entropy'):
        """
        Returns a new example based on the chosen active learning strategy.
        :param mode: Active Learning Strategy
            - max (Default)
            - mean
        :return:
        """
        unlab = self.all_data[self.all_data[self.label_col].isna()]
        if len(unlab) > 1000:
            unlab = unlab.sample(1000)
        if mode == 'entropy':
            unlabelled_idx = unlab.index
            probs = self.model.predict_proba(unlab)

            if self.multilabel:
                ent = np.array([entropy(p.T) for p in probs])
                mean_proba = ent.mean(axis=0)
                proba_idx = np.argmax(mean_proba)
            else:
                ent = entropy(probs.T)
                proba_idx = np.argmax(ent)

            actual_idx = unlabelled_idx[proba_idx]
            current_example_index = actual_idx
            current_example = self.all_data.loc[current_example_index]
            toret = {
                'example_index': int(current_example_index),
                'view': self.generate_view(current_example)
            }
            if self.model:
                preds = self.model.predict(
                    self.all_data.loc[current_example_index: current_example_index+1])
                if self.multilabel:
                    preds = list(
                        self.multilabel_binarizer.inverse_transform(preds)[0])
                    toret['predictions'] = preds
                else:
                    preds = [preds[0]]
                    toret['predictions'] = preds
            return toret

    def update_model(self):
        """
        Updates the model with the currently labelled dataset
        :return:
        """

        lab = self.all_data[self.all_data[self.label_col].notna()]
        if len(lab) == 0:
            return None

        if self.model is None:
            if self.feature_transformer is None:
                self.model = Pipeline([
                    ('vect', make_pipeline(ColumnsSelector(self.text_col),
                     CountVectorizer(ngram_range=(1, 2)))),
                    ('clf', RandomForestClassifier())
                ])
            else:

                self.model = Pipeline([
                    ('fu', FeatureUnion([
                        ('text_vectorizer',
                         make_pipeline(ColumnsSelector(self.text_col), CountVectorizer(ngram_range=(1, 2)), ToDense())),
                        ('text_featurizer', make_pipeline(ColumnsSelector(
                            self.feature_columns), MultiLabelEncoder()))
                    ])),
                    ('clf', RandomForestClassifier())
                ])

        if self.multilabel:
            self.multilabel_binarizer = MultiLabelBinarizer()
            labels = self.multilabel_binarizer.fit_transform(
                lab[self.label_col].apply(lambda x: x.split('; ')))
        else:
            labels = lab[self.label_col]

        self.model.fit(lab, labels)

    def save_example(self, example_index, data):
        """
        Saves the current example with the user tagged data
        :param data: User tagged data. [list of tags]
        :return:
        """

        print(data)
        self.all_data.loc[example_index, self.label_col] = '; '.join(data)

    def save_data(self, filepath=None):
        """
        Saves the labelled data to a file
        :param filepath: file to save the data in a pickle format.
        :return:
        """
        if filepath is None:
            filepath = os.path.join(
                self.data_directory, 'text_classification_data.csv')
        self.all_data[[self.text_col, self.label_col]
                      ].dropna().to_csv(filepath, index=False)

    def load_data(self, filepath=None):
        """
        Loads labelled data from file.
        :param filepath: file containing pickeled labelled dataset
        :return:
        """
        if filepath is None:
            filepath = os.path.join(
                self.data_directory, 'text_classification_data.csv')
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


def render_app_template(unique_tags_data, ui_data):
    """
    Tag data in the form
    [
        (tag_id, readable_tag_name)
    ]
    :param unique_tags_data:
    :return: html template to render
    """

    trainer_path = os.path.join(os.path.dirname(__file__), 'html_templates',
                                'text_classifier.html')
    with open(trainer_path) as templ:
        template = Template(templ.read())

    # css_classes = []
    # for index, item in enumerate(unique_tags_data):
    #     css_classes.append((item[0], list_of_colors[index]))

    return template.render(tag_controls=unique_tags_data, ui_data=json.dumps(ui_data))


def get_app(tagger, tags, ui_data):
    app = Flask(__name__)

    @app.route("/")
    def base_app():
        return render_app_template(tags, ui_data)

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
        print(form_data)
        payload = form_data['payload']
        form_data = json.loads(payload)
        tag = form_data['tag']
        example_index = int(form_data['example_index'])
        tagger.save_example(example_index, tag)
        return 'Success'

    @app.route('/save_data')
    def save_tagged_data():
        print("save_tagged_data")
        tagger.save_data()
        return 'Data Saved'

    return app


class TextClassifier:
    def __init__(self,
                 all_data,
                 unique_tags,
                 label_col='class',
                 text_col='text',
                 multilabel=False,
                 feature_transformer=None,
                 display_function=None,
                 data_directory=''):
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
        self.tagger = BaseTextClassifier(
            all_data,
            text_col=text_col,
            label_col=label_col,
            multilabel=multilabel,
            feature_transformer=feature_transformer,
            display_function=display_function,
            data_directory=data_directory)

        self.ui_data = {
            'multilabel': multilabel
        }
        self.app = get_app(self.tagger, self.unique_tags, self.ui_data)
        self.utmapping = {t[0]: t[1] for t in self.unique_tags}

    def start_server(self, host=None, port=None):
        """
        Start the ner tagging server
        :param port: Port number to bind the server to.
        :return:
        """
        self.app.run(host=host, port=port)

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

    tags = [
        ("Address", "Address"),
        ("Other", "Non Address"),
    ]
