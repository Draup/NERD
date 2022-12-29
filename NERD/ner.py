#!/usr/bin/env python
# coding: utf-8


import json
from sklearn_crfsuite import CRF
import numpy as np
from scipy.stats import entropy
from nltk import word_tokenize, pos_tag
import random
import pickle
import os
from bs4 import BeautifulSoup
from bs4 import Tag
from collections import Counter

from flask import Flask
from flask import request
from jinja2 import Template
import pandas as pd


def get_sentences(text):
    sents = text.split('\n')
    sents = [s.strip() for s in sents if len(s.strip()) > 0]
    return sents


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


def word2features(sent, i):
    """
    Calculate features for each word in the sentence
    :param sent: List of words in the sentence
    :param i: i'th word in the sentence
    :return:
    """
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.is_alphanum': is_alpha_and_numeric(word),
        'postag': postag,
    }

    if i > 0:
        word = sent[i - 1][0]
        postag = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word.lower(),
            '-1:word[-3:]': word[-3:],
            '-1:word[-2:]': word[-2:],
            '-1:word.istitle()': word.istitle(),
            '-1:word.isupper()': word.isupper(),
            '-1:postag': postag,
            '-1:word.is_alphanum': is_alpha_and_numeric(word)
        })
    else:
        features['BOS'] = True

    if i > 1:
        word = sent[i - 2][0]
        postag = sent[i - 2][1]
        features.update({
            '-2:word.lower()': word.lower(),
            '-2:word[-3:]': word[-3:],
            '-2:word[-2:]': word[-2:],
            '-2:word.istitle()': word.istitle(),
            '-2:word.isupper()': word.isupper(),
            '-2:postag': postag,
            '-2:word.is_alphanum': is_alpha_and_numeric(word)
        })

    if i < len(sent) - 1:
        word = sent[i + 1][0]
        postag = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word.lower(),
            '+1:word[-3:]': word[-3:],
            '+1:word[-2:]': word[-2:],
            '+1:word.istitle()': word.istitle(),
            '+1:word.isupper()': word.isupper(),
            '+1:postag': postag,
            '+1:word.is_alphanum': is_alpha_and_numeric(word)
        })
    else:
        features['EOS'] = True

    if i < len(sent) - 2:
        word = sent[i + 2][0]
        postag = sent[i + 2][1]
        features.update({
            '+2:word.lower()': word.lower(),
            '+2:word[-3:]': word[-3:],
            '+2:word[-2:]': word[-2:],
            '+2:word.istitle()': word.istitle(),
            '+2:word.isupper()': word.isupper(),
            '+2:postag': postag,
            '+2:word.is_alphanum': is_alpha_and_numeric(word)
        })

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def add_prediction_to_postagged_data(postagged, prediction):
    toret = []
    for i in range(len(postagged)):
        toret.append((postagged[i][0], postagged[i][1], prediction[i]))
    return toret


def get_prediction_uncertainity(pred, mode='max'):
    if len(pred) == 0:
        return 0
    un = []
    for tok in pred:
        probabilities = list(tok.values())
        ent = entropy(probabilities)
        un.append(ent)
    if mode == 'max':
        return max(un)
    elif mode == 'mean':
        return sum(un) / len(un)
    
    
def find_entities_in_text(text, model, utmapping):
    
    text = get_pos_tagged_example(text)
    features = sent2features(text)
    prediction = model.predict_single(features)
    lst = zip([t[0] for t in text], prediction)
    curr_ent = 'O'
    ent_toks = []
    entities = []
    for item in lst:
        text = item[0]
        tag = item[1]
        if tag.startswith('B-'):
            if len(ent_toks) > 0:
                entities.append({
                    'value': ' '.join(ent_toks),
                    'entity': utmapping[curr_ent],
                })
                ent_toks = []
            curr_ent = tag[2:]
            ent_toks.append(text)
        elif tag.startswith('I-'):
            if curr_ent == 'O':
                continue
            ent_toks.append(text)
        elif tag.startswith('L-'):
            if curr_ent == 'O':
                continue
            ent_toks.append(text)
            entities.append({
                'value': ' '.join(ent_toks),
                'entity': utmapping[curr_ent],
            })
            ent_toks = []
        elif tag.startswith('U-'):
            curr_ent = tag[2:]
            ent_toks = []
            entities.append({
                'value': text,
                'entity': utmapping[curr_ent],
            })
        elif tag.startswith('O'):
            if len(ent_toks) > 0:
                entities.append({
                    'value': ' '.join(ent_toks),
                    'entity': utmapping[curr_ent],
                })
            ent_toks = []
            curr_ent = 'O'

    if len(ent_toks) > 0:
        entities.append({
            'value': ' '.join(ent_toks),
            'entity': utmapping[curr_ent],
        })
    return entities

class BaseNerTagger:
    """
    A utility class for NER Tagging.
    """

    def __init__(self, unlabelled, labelled=None, data_directory=''):
        """
        Initialize with a list of unlabelled strings and/or list of tagged tuples.
        :param unlabelled: list of strings
        :param labelled: list of {list of tuples [(token, pos_tag, tag), ...]}
        """
        if unlabelled is None:
            unlabelled = []
        else:
            unlabelled = [
                {'raw': get_pos_tagged_example(text)} for text in unlabelled]
        if labelled is None:
            labelled = []

        self.dataset = []
        for ex in unlabelled:
            self.dataset.append({
                'status': 'UNLABELLED',
                'data': ex
            })

        for ex in labelled:
            self.dataset.append({
                'status': 'LABELLED',
                'data': ex
            })

        self.model = None

        self.data_directory = os.path.join(data_directory, 'NER_Data')
        os.makedirs(self.data_directory, exist_ok=True)

    def get_unlabelled_indices(self):
        return [index for index, ex in enumerate(self.dataset) if ex['status'] == 'UNLABELLED']

    def get_new_random_example(self):
        """
        Returns a random example to be tagged. Used to bootstrap the model.
        :return:
        """
        unlabelled_set = self.get_unlabelled_indices()
        current_example_index = random.choice(unlabelled_set)
        current_example = self.dataset[current_example_index]['data']
        toret = current_example['raw']
        return {
            'example_id': current_example_index,
            'example': toret
        }

    def get_new_random_predicted_example(self):
        """
        Returns a random example tagged by the currently tagged model.
        :return:
        """
        unlabelled_set = self.get_unlabelled_indices()
        current_example_index = random.choice(unlabelled_set)
        current_example = self.dataset[current_example_index]['data']
        raw = current_example['raw']
        features = sent2features(raw)
        preds = self.model.predict_single(features)
        toret = add_prediction_to_postagged_data(raw, preds)
        return {
            'example_id': current_example_index,
            'example': toret
        }

    def query_new_example(self, mode='max'):
        """
        Returns a new example based on the chosen active learning strategy.
        :param mode: Active Learning Strategy
            - max (Default)
            - mean
        :return:
        """
        unlabelled_set = self.get_unlabelled_indices()
        sample = random.choices(unlabelled_set, k=250)
        X = []
        for s in sample:
            example = self.dataset[s]['data']
            if 'features' not in example:
                example['features'] = sent2features(example['raw'])
            X.append(example['features'])
        preds = self.model.predict_marginals(X)
        uncertainities = [get_prediction_uncertainity(
            pred, mode) for pred in preds]
        index = np.argmax(uncertainities)
        current_example_index = sample[index]
        current_example = self.dataset[current_example_index]['data']
        raw = current_example['raw']
        features = current_example['features']
        preds = self.model.predict_single(features)
        toret = add_prediction_to_postagged_data(raw, preds)
        return {
            'example_id': current_example_index,
            'example': toret
        }

    def update_model(self):
        """
        Updates the model with the currently labelled dataset
        :return:
        """
        if self.model is None:
            self.model = CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=100,
                all_possible_transitions=True
            )

        labelled = [item['data']
                    for item in self.dataset if item['status'] == 'LABELLED']
        X = [item['features'] for item in labelled]
        Y = [sent2labels(item['raw']) for item in labelled]
        self.model.fit(X, Y)

    def save_example(self, example_id, data):
        """
        Saves the current example with the user tagged data
        :param data: User tagged data. [list of tags]
        :return:
        """

        current_example = self.dataset[example_id]['data']

        if len(data) != len(current_example['raw']):
            return False
        else:
            toret = []
            for index in range(len(data)):
                toret.append(
                    (current_example['raw'][index][0], current_example['raw'][index][1], data[index][1]))

            example = current_example
            example['raw'] = toret
            example['features'] = sent2features(toret)
            self.dataset[example_id]['status'] = 'LABELLED'

    def save_data(self, filepath=None):
        """
        Saves the labelled data to a file
        :param filepath: file to save the data in a pickle format.
        :return:
        """
        if filepath is None:
            filepath = os.path.join(
                self.data_directory, 'ner_tagged_data.pickle')
        with open(filepath, 'wb') as out:
            pickle.dump(self.labelled, out)

    def load_data(self, filepath=None):
        """
        Loads labelled data from file.
        :param filepath: file containing pickeled labelled dataset
        :return:
        """
        with open(filepath, 'rb') as inp:
            self.labelled = pickle.load(inp)
            for lab in self.labelled:
                lab['features'] = sent2features(lab['raw'])

    def add_unlabelled_examples(self, examples):
        """
        Append more unlabelled data to dataset
        :param examples: List of strings
        :return:
        """
        new_examples = [
            {'raw': get_pos_tagged_example(text)} for text in examples]
        self.unlabelled.extend(new_examples)


def is_a_tag(span):
    if 'data-tag' in span.attrs:
        return True
    return False


def get_bilou_tags_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    toret = []

    tag_items = soup.find_all(['span', 'br'], attrs={'data-tag': True})
    #     return tag_items
    tag_ids = [item.attrs['data-tag-id'] for item in tag_items]
    counter = Counter(tag_ids)

    items = soup.find_all(['span', 'br'])
    max_items = len(items)
    index = 0
    while index < max_items:
        item = items[index]
        if is_a_tag(item):
            tag_id = item.attrs['data-tag-id']
            tag = item.attrs['data-tag']
            size = counter[tag_id]
            if size == 1:
                toret.append((item.text, f'U-{tag}'))
                index += 1
            elif size == 2:
                toret.append((item.text, f'B-{tag}'))
                toret.append((items[index + 1].text, f'L-{tag}'))
                index += 2
            else:
                toret.append((item.text, f'B-{tag}'))
                for i in range(size - 2):
                    toret.append((items[index + i + 1].text, f'I-{tag}'))
                toret.append((items[index + size - 1].text, f'L-{tag}'))
                index += size
        else:
            toret.append((item.text, 'O'))
            index += 1

    return toret


def generate_html_from_example(ex):
    spans = []
    if type(ex) == type({}):
        ex = ex['raw']
    for item in ex:
        if item[0] == '\n':
            tag = Tag(name='br', can_be_empty_element=True)
        else:
            tag = Tag(name='span')
            tag.insert(0, item[0])

        spans.append(tag)

    if len(ex[0]) == 3:
        tagidcounter = 0
        last_tag = ''
        for i in range(len(ex)):
            tag = ex[i][2]
            if tag[0] in ['B', 'I']:
                tag = tag[2:]
                spans[i].attrs['data-tag-id'] = tagidcounter
                spans[i].attrs['data-tag'] = tag
                spans[i].attrs['class'] = tag

            elif tag[0] in ['L', 'U']:
                tag = tag[2:]
                spans[i].attrs['data-tag-id'] = tagidcounter
                spans[i].attrs['data-tag'] = tag
                spans[i].attrs['class'] = tag
                tagidcounter += 1

    soup = BeautifulSoup()
    soup.extend(spans)
    return str(soup)


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

    trainer_path = os.path.join(os.path.dirname(
        __file__), 'html_templates', 'ner_trainer.html.j2')
    with open(trainer_path) as templ:
        template = Template(templ.read())

    css_classes = []
    for index, item in enumerate(unique_tags_data):
        css_classes.append((item[0], list_of_colors[index]))

    return template.render(css_classes=css_classes, id_color_map=css_classes, tag_controls=unique_tags_data)


def get_app(ntagger, tags):
    app = Flask(__name__)

    @app.route("/")
    def base_app():
        return render_app_template(tags)

    @app.route('/load_example')
    def load_example():
        if ntagger.model is None:
            example = ntagger.get_new_random_example()
        else:
            example = ntagger.query_new_example(mode='max')

        return {
            'example_id': example['example_id'],
            'example_html': generate_html_from_example(example['example'])
        }

    @app.route('/update_model')
    def update_model():
        ntagger.update_model()
        return "Model Updated Successfully"

    @app.route('/save_example', methods=['POST'])
    def save_example():
        form_data = request.form
        html = form_data['html']
        example_id = int(form_data['example_id'])
        user_tags = get_bilou_tags_from_html(html)
        ntagger.save_example(example_id, user_tags)
        return 'Success'

    @app.route('/save_data')
    def save_tagged_data():
        print("save_tagged_data")
        ntagger.save_data()
        return 'Data Saved'

    return app


def get_pos_tagged_example(text):
    sents = get_sentences(text)
    tokens = []
    for index, sent in enumerate(sents):
        if index > 0 and index < len(sents) - 1:
            tokens.append('\n')

        tokens.extend(word_tokenize(sent))
    # tokens = word_tokenize(text)
    toret = pos_tag(tokens)
    return toret


class NerTagger:
    def __init__(self,
                 dataset,
                 unique_tags,
                 data_directory='',
                 multiuser=False
                 ):
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
        self.ntagger = BaseNerTagger(dataset, data_directory=data_directory)
        self.app = get_app(self.ntagger, self.unique_tags)
        self.utmapping = {t[0]: t[1] for t in self.unique_tags}

    def start_server(self, host=None, port=None):
        """
        Start the ner tagging server
        :param port: Port number to bind the server to.
        :return:
        """
        self.app.run(host, port)

    def add_unlabelled_examples(self, examples):
        """
        Append unlabelled examples to dataset
        :param examples: list of strings
        :return:
        """
        self.ntagger.add_unlabelled_examples(examples)

    def save_labelled_examples(self, filepath):
        """
        Save labelled examples to file
        :param filepath: destination filename
        :return:
        """
        self.ntagger.save_data(filepath)

    def load_labelled_examples(self, filepath):
        """
        Load labelled examples to the dataset
        :param filepath: source filename
        :return:
        """
        self.ntagger.load_data(filepath)

    def save_model(self, model_filename):
        """
        Save ner model to file
        :param model_filename: destination filename
        :return:
        """
        with open(model_filename, 'wb') as out:
            pickle.dump(self.ntagger.model, out)

    def load_model(self, model_filename):
        """
        Load ner model from file
        :param model_filename: source filename
        :return:
        """
        with open(model_filename, 'rb') as inp:
            self.ntagger.model = pickle.load(inp)

    def update_model(self):
        """
        Updates the model
        :return:
        """
        self.ntagger.update_model()

    def find_entities_in_text(self, text):
        return find_entities_in_text(text, self.ntagger.model, self.utmapping)
    


if __name__ == '__main__':
    # Unique Tags / Classes

    tags = [
        ("CT", "Course Title"),
        ("CC", "Course Code"),
        ("PREQ", "Pre-requisites"),
        ("PROF", "Professor"),
        ("SE", "Season"),
        ("CR", "Credits")
    ]
