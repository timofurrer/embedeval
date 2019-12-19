"""
embedeval
~~~~~~~~~

NLP Embedding Evaluation Tool

:copyright: (c) 2019 by David Staub <david_staub@hotmail.com)
:license: MIT, see LICENSE for more details.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from embedeval.logger import get_component_logger  # noqa
from embedeval.task import Task, TaskReport  # noqa

logger = get_component_logger("offense_detection")


def recall_metric(y_true, y_pred):
    """Calculate the Recall from a keras batch"""
    from keras import backend as K

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_metric(y_true, y_pred):
    """Calculate the Precision from a keras batch"""
    from keras import backend as K

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_metric(y_true, y_pred):
    """Calculate the F1 score from a keras batch"""
    from keras import backend as K

    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class OffenseDetectionTask(Task):  # type: ignore
    """Represents an Offense detection Task"""

    NAME = "de-offense-detection"

    DATASET_PATH = Path(__file__).parent / "data" / "germeval-2018"
    TRAIN_DATASET_PATH = DATASET_PATH / "train.txt"
    TEST_DATASET_PATH = DATASET_PATH / "test.txt"

    TEXT_COLUMN_NAME = "tweet"
    SENTIMENT_COLUMN_NAME = "label"

    def __init__(self):
        super().__init__()

        self.train_dataset = self._load_dataset(self.TRAIN_DATASET_PATH)
        self.test_dataset = self._load_dataset(self.TEST_DATASET_PATH)
        self.dataset = pd.concat([self.train_dataset, self.test_dataset])
        self.corpus = self.dataset[self.TEXT_COLUMN_NAME]
        self.tokenizer = self._create_tokenizer(self.corpus)

        self.corpus_sentence_length = self._calculate_sentence_length(self.corpus)

        self.train_corpus = self._prepare_dataset_corpus(
            self.train_dataset[self.TEXT_COLUMN_NAME],
            self.tokenizer,
            self.corpus_sentence_length,
        )
        self.test_corpus = self._prepare_dataset_corpus(
            self.test_dataset[self.TEXT_COLUMN_NAME],
            self.tokenizer,
            self.corpus_sentence_length,
        )

    def _load_dataset(self, dataset_path):
        logger.debug("Loading dataset %s", str(dataset_path))
        df = pd.read_csv(
            dataset_path,
            sep="\t",
            names=[self.TEXT_COLUMN_NAME, self.SENTIMENT_COLUMN_NAME],
            usecols=[0, 1],
        )
        df["label"] = df["label"].map({"OFFENSE": 1, "OTHER": 0})

        # remove words starting with an @ from all tweets
        df["tweet"] = df["tweet"].apply(lambda x: re.sub(r"@[A-Za-z0-9_]{3,}", "", x))
        logger.debug("Loaded dataset %s", str(dataset_path))
        return df

    def _create_tokenizer(self, corpus):
        # NOTE(TF): lazy import keras, because it takes forever to import
        #           and we don't want to do that just to import this task module.
        from keras.preprocessing.text import Tokenizer  # noqa

        word_tokenizer = Tokenizer()
        word_tokenizer.fit_on_texts(corpus)
        return word_tokenizer

    def _calculate_sentence_length(self, corpus):
        from nltk.tokenize import word_tokenize  # noqa

        # NOTE(TF): lazy import keras, because it takes forever to import
        #           and we don't want to do that just to import this task module.

        logger.debug("Calculating corpus sentence length")
        word_count = lambda sentence: len(word_tokenize(sentence))  # noqa
        longest_sentence = max(corpus, key=word_count)
        sentence_length = len(word_tokenize(longest_sentence))
        logger.debug("Calculated corpus sentence length of %d", sentence_length)

        return sentence_length

    def _prepare_dataset_corpus(self, corpus, tokenizer, sentence_length):
        # NOTE(TF): lazy import keras, because it takes forever to import
        #           and we don't want to do that just to import this task module.
        from keras.preprocessing.sequence import pad_sequences  # noqa

        text = tokenizer.texts_to_sequences(corpus)
        padded_text = pad_sequences(text, sentence_length, padding="post")
        return padded_text

    def evaluate(self, embedding) -> TaskReport:
        # define the minimum score to pass the Task
        goal_f1_score = 0.75
        goal_accuracy = 0.70

        # define the title for the report
        report_title = f"Detect if the defined Tweets are offensive or not."

        # create model
        model = self._create_cnn_model(
            embedding, self.tokenizer, self.corpus_sentence_length
        )

        model_verbosity = 1

        # train model
        model.fit(
            self.train_corpus,
            self.train_dataset[self.SENTIMENT_COLUMN_NAME],
            validation_split=0.3,
            epochs=5,
            verbose=model_verbosity,
        )

        # evaluate model
        print(model)
        actual_loss, actual_accuracy, actual_f1_score = model.evaluate(
            self.test_corpus,
            self.test_dataset[self.SENTIMENT_COLUMN_NAME],
            verbose=model_verbosity,
        )

        if goal_accuracy > actual_accuracy or goal_f1_score > actual_f1_score:
            logger.error(
                "The prediction of the model was not good enough. "
                "The Goal of %.3f accuracy and %.3f F1 score was not reached, "
                "the actual accuracy was %.3f and F1 was %.3f",
                goal_accuracy,
                goal_f1_score,
                actual_accuracy,
                actual_f1_score,
            )
            return TaskReport(
                self.NAME,
                outcome=False,
                title=report_title,
                body=(
                    f"""The prediction of the model was not good enough.
The goal of {goal_accuracy:.2} accuracy and {goal_f1_score:.2} F1 score was not reached.
The actual accuracy was {actual_accuracy:.2} and F1 score was {actual_f1_score:.2}."""
                ),
            )

        logger.debug(
            "The accuracy of the prediction was %.3f and the F1 score was %.3f",
            actual_accuracy,
            actual_f1_score,
        )

        return TaskReport(
            self.NAME,
            outcome=True,
            title=report_title,
            body=(
                f"""The prediction of the model was accurate {actual_accuracy:.2%} of the time.
The goal of minimum {goal_accuracy:.2} accuracy and {goal_f1_score:.2} F1 score was reached.
The actual accuracy was {actual_accuracy:.2} and F1 score was {actual_f1_score:.2}."""
            ),
        )

    def _create_cnn_model(self, embedding, tokenizer, sentence_length):
        # NOTE(TF): lazy import keras, because it takes forever to import
        #           and we don't want to do that just to import this task module.
        from keras.layers import Dense, Flatten  # noqa
        from keras.layers.embeddings import Embedding  # noqa
        from keras.models import Sequential  # noqa

        embedding_matrix = self._create_embedding_matrix(embedding, tokenizer)

        model = Sequential()
        model.add(
            Embedding(
                embedding_matrix.shape[0],
                embedding_matrix.shape[1],
                weights=[embedding_matrix],
                input_length=sentence_length,
                trainable=True,
            )
        )
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["acc", f1_metric]
        )

        return model

    def _create_embedding_matrix(self, embedding, tokenizer):
        # Get the Vocabulary length and add 1 for all unknown words
        vocab_length = len(tokenizer.word_index) + 1

        embedding_matrix = np.zeros((vocab_length, embedding.shape[1]))
        for word, index in tokenizer.word_index.items():
            try:
                embedding_vector = embedding.get_word_vector(word)
            except KeyError:
                embedding_vector = None

            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        return embedding_matrix
