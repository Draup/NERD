import numpy as np
from scipy.stats import entropy
import pickle
import os
from flask import Flask
from flask import request
from flask import session
from jinja2 import Template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import random
import json

from sklearn.pipeline import Pipeline
from .nerdlogin import LoginRequired

class BaseDataFrameClassifier:
    """
    A utility class for Text Classification
    """
    SAMPLE_THRESH = 1000

    def __init__(self,
                 data,
                 featurization_pipeline=None,
                 label_col='class',
                 user_col="NERD_USER",
                 multilabel=False,
                 data_directory='',
                 display_function=None,
                 unique_tags=None):
        """
        Initialize with a DataFrame(['text']) and/or DataFrame(['text', 'class'])
        Args:
            unlabelled: DataFrame(['text'])
            labelled: DataFrame(['text', 'class'])
            feature_transformer: Sklearn transformer to calculate extra features
            data_directory: Default data directory
        """
        self.unique_tags = unique_tags
        self.multilabel = multilabel
        self.display_function = display_function
        self.all_data = data
        self.label_col = label_col
        self.user_col = user_col
        self.info_cols = [self.label_col, self.user_col]
        for col in self.info_cols:
            if col not in self.all_data.columns:
                self.all_data[col] = np.nan

        self.featurization_pipeline = featurization_pipeline
        self.model = None
        self.data_directory = os.path.join(data_directory, 'DF_Classification_Data')
        os.makedirs(self.data_directory, exist_ok=True)

    def generate_view(self, example):
        example = json.loads(example.to_json())
        if self.display_function:
            return self.display_function(example)
        else:
            return json.dumps(example, indent=2)

    def get_new_random_example(self):
        """
        Returns a random example to be tagged. Used to bootstrap the model.
        Returns:
        """
        unl = self.all_data[self.all_data[self.label_col].isna()].index
        current_example_index = np.random.choice(unl)
        current_example = self.all_data.loc[current_example_index]

        toret = {
            'example_index': int(current_example_index),
            'view': self.generate_view(current_example)
        }
        if self.model:
            preds = self.model.predict(self.all_data.loc[current_example_index: current_example_index + 1])
            if self.multilabel:
                preds = list(self.multilabel_binarizer.inverse_transform(preds)[0])
                toret['predictions'] = preds
            else:
                preds = [preds[0]]
                toret['predictions'] = preds

        return toret

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
            THRESH = BaseDataFrameClassifier.SAMPLE_THRESH
            unlab = self.all_data[self.all_data[self.label_col].isna()]

            if len(unlab) > THRESH:
                unlab = unlab.sample(THRESH)
            unlabelled_idx = unlab.index
            proba = self.model.predict_proba(unlab)
            proba_idx = np.argmax(entropy(proba.T))
            actual_idx = unlabelled_idx[proba_idx]

            current_example_index = actual_idx
            current_example = self.all_data.loc[current_example_index]


        elif mode == 'top2diff':
            THRESH = BaseDataFrameClassifier.SAMPLE_THRESH
            unlab = self.all_data[self.all_data[self.label_col].isna()]

            #             unlab = self.all_data
            if len(unlab) > THRESH:
                unlab = unlab.sample(THRESH)
            unlabelled_idx = unlab.index
            proba = self.model.predict_proba(unlab)
            proba = np.sort(proba)
            proba_idx = np.argmin(proba[:, -1] - proba[:, -2])
            actual_idx = unlabelled_idx[proba_idx]

            current_example_index = actual_idx
            current_example = self.all_data.loc[current_example_index]

        toret = {
            'example_index': int(current_example_index),
            'view': self.generate_view(current_example)
        }
        if self.model:
            preds = self.model.predict(self.all_data.loc[current_example_index: current_example_index + 1])
            if self.multilabel:
                preds = list(self.multilabel_binarizer.inverse_transform(preds)[0])
                toret['predictions'] = preds
            else:
                preds = [preds[0]]
                toret['predictions'] = preds
        return toret

    def update_model(self):
        """
        Updates the model with the currently labelled dataset
        Returns:
        """

        if self.model is None:
            self.model = Pipeline([
                ('fu', self.featurization_pipeline),
                ('clf', RandomForestClassifier(n_estimators=50))
            ])

        lab = self.all_data[self.all_data[self.label_col].notna()]
        self.model.fit(lab, lab[self.label_col])

    def save_example(self, example_index, data, username=None):
        """
        Saves the current example with the user tagged data
        :param data: User tagged data. [list of tags]
        :return:
        """

        print(data)
        self.all_data.loc[example_index, self.label_col] = '; '.join(data)
        self.all_data.loc[example_index, self.user_col] = username

    def save_data(self, filepath=None):
        """
        Saves the labelled data to a file
        Args:
            filepath: file to save the data in a pickle format.
        Returns:
        """
        if filepath is None:
            filepath = os.path.join(self.data_directory, 'df_classification_data.pickle')
        
        filetype = filepath.split('.')[-1]
        to_save = self.all_data[self.all_data[self.label_col].notna()]
        if filetype == 'pickle':
            to_save.to_pickle(filepath)
        elif filetype == 'csv':
            to_save.to_csv(filepath)
        elif filetype == 'json':
            to_save.to_json(filepath, orient='records', lines=True)


class DataFrameClassifier:
    def __init__(self,
                 dataset, unique_tags,
                 label_col='class',
                 multilabel=False,
                 featurization_pipeline=None,
                 display_function=None,
                 data_directory='',
                 usermanagement=False
                 ):
        """
        Text Classifier from dataset and unique tags
        Args:
            dataset: list of strings
            unique_tags: list of tuples [(identifier, Readable Name)..]
            data_directory: Default data directory
        """
        self.unique_tags = unique_tags
        self.usermanagement = usermanagement
        self.utmapping = {t[0]: t[1] for t in self.unique_tags}
        self.tagger = BaseDataFrameClassifier(dataset,
                                              label_col=label_col,
                                              featurization_pipeline=featurization_pipeline,
                                              data_directory=data_directory,
                                              multilabel=multilabel,
                                              display_function=display_function,
                                              unique_tags=self.utmapping)
        self.ui_data = {
            'multilabel': multilabel
        }
        self.app = get_app(self, self.unique_tags, self.ui_data)
        
        if usermanagement:
            if data_directory == '':
                usersjson = './nerdusers.json'
            else:
                usersjson = f'./{data_directory}/nerdusers.json'
            
            print(usersjson)
            self.um = LoginRequired(self.app, usersjson=usersjson)

    def start_server(self, host=None, port=None):
        """
        Start text classification server at the given port.
        Args:
            port: Port number to bind the server to.
        Returns:
        """
        self.app.run(host=host, port=port)

    def add_unlabelled_examples(self, examples):
        """
        Append unlabelled examples to dataset
        Args:
            examples: list of strings
        Returns:
        """
        self.tagger.add_unlabelled_examples(examples)

    def save_labelled_examples(self, filepath):
        """
        Save labelled examples to file
        Args:
            filepath: destination filename
        Returns:
        """
        self.tagger.save_data(filepath)

    def load_labelled_examples(self, filepath):
        """
        Load labelled examples to the dataset
        Args:
            filepath: source filename
        Returns:
        """
        self.tagger.load_data(filepath)

    def save_model(self, model_filename):
        """
        Save classifier model to file
        Args:
            model_filename: destination filename
        Returns:
        """
        with open(model_filename, 'wb') as out:
            pickle.dump(self.tagger.model, out)

    def load_model(self, model_filename):
        """
        Load classifier model from file
        Args:
            model_filename: source filename
        Returns:
        """
        with open(model_filename, 'rb') as inp:
            self.tagger.model = pickle.load(inp)

    def update_model(self):
        """
        Updates the model
        Returns:
        """
        self.tagger.update_model()


def render_app_template(unique_tags_data, ui_data):
    """
    Tag data in the form
    [
        (tag_id, readable_tag_name)
    ]
    Args:
        unique_tags_data: list of tag tuples
    Returns: html to render
    """

    trainer_path = os.path.join(os.path.dirname(__file__), 'html_templates',
                                'text_classifier.html')
    with open(trainer_path) as templ:
        template = Template(templ.read())

    # css_classes = []
    # for index, item in enumerate(unique_tags_data):
    #     css_classes.append((item[0], list_of_colors[index]))

    return template.render(tag_controls=unique_tags_data, ui_data=json.dumps(ui_data))


def get_app(dfc, tags, ui_data):
    tagger = dfc.tagger
    app = Flask(__name__)
    print(dfc.usermanagement)

    @app.route("/")
    def base_app():
        return render_app_template(tags, ui_data)

    @app.route('/load_example')
    def load_example():
        if tagger.model is None:
            example = tagger.get_new_random_example()
        else:
            randint = random.random()
            if randint < 0.33:
                example = tagger.query_new_example(mode='entropy')
            elif randint < 0.67:
                example = tagger.query_new_example(mode='top2diff')
            else:
                example = tagger.get_new_random_example()

        # print(f'Returning example ::: {example[:100]}')

        return example

    @app.route('/update_model')
    def update_model():
        tagger.update_model()
        return "Model Updated Successfully"

    @app.route('/save_example', methods=['GET'])
    def save_example():
        print(request)
        data = request.args.get('payload')
        form_data = json.loads(data)
        print(form_data)
        tag = form_data['tag']
        example_index = int(form_data['example_index'])
        username = session.get('username', None)
        tagger.save_example(example_index, tag, username=username)
        return 'Success'

    @app.route('/save_data')
    def save_tagged_data():
        print("save_tagged_data")
        tagger.save_data()
        return 'Data Saved'

    return app
