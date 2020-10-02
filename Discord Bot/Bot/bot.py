import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import logging
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_pth = os.path.join(BASE_DIR, "model")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

logger = logging.getLogger("ChatBot")


class ChatBot(object):
    data: dict
    words: list
    labels: list
    training: list
    output: list

    @classmethod
    def train(cls, epoch: int = 5000) -> None:
        try:
            with open(os.path.join(model_pth, "intents.json")) as file:
                data = json.load(file)

        except Exception as e:
            logger.warning("Cannot Find intents.json!")
            quit()

        try:
            with open(os.path.join(model_pth, "data.pickle"), "rb") as f:
                words, labels, training, output = pickle.load(f)

        except Exception as e:
            logger.info("Optimizing Data..")
            words = []
            labels = []
            docs_x = []
            docs_y = []

            for intent in data["intents"]:
                for pattern in intent["patterns"]:
                    wrds = nltk.word_tokenize(pattern)
                    words.extend(wrds)
                    docs_x.append(wrds)
                    docs_y.append(intent["tag"])

                if intent["tag"] not in labels:
                    labels.append(intent["tag"])

            words = [stemmer.stem(w.lower()) for w in words if w != "?"]
            words = sorted(list(set(words)))

            labels = sorted(labels)

            training = []
            output = []

            out_empty = [0 for _ in range(len(labels))]

            for x, doc in enumerate(docs_x):
                bag = []

                wrds = [stemmer.stem(w.lower()) for w in doc]

                for w in words:
                    if w in wrds:
                        bag.append(1)
                    else:
                        bag.append(0)

                output_row = out_empty[:]
                output_row[labels.index(docs_y[x])] = 1

                training.append(bag)
                output.append(output_row)

            training = numpy.array(training)
            output = numpy.array(output)

            with open(os.path.join(model_pth, "data.pickle"), "wb") as f:
                pickle.dump((words, labels, training, output), f)

            logger.info("Saved Optimized Data")

        tensorflow.reset_default_graph()
        net = tflearn.input_data(shape=[None, len(training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
        net = tflearn.regression(net)
        model = tflearn.DNN(net)
        model.fit(training, output, n_epoch=epoch, batch_size=8, show_metric=True)
        model.save(os.path.join(model_pth, "model.tflearn"))

    @classmethod
    def model(cls):
        try:
            with open(os.path.join(model_pth, "data.pickle"), "rb") as f:
                words, labels, training, output = pickle.load(f)

        except Exception as e:
            logger.warning("Cannot Find Data! Please Make Sure You Have Optimized Your Data First!")
            quit()

        tensorflow.reset_default_graph()
        net = tflearn.input_data(shape=[None, len(training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
        net = tflearn.regression(net)

        model = tflearn.DNN(net)

        try:
            model.load(os.path.join(model_pth, "model.tflearn"))
            return model

        except Exception as e:
            logger.warning("Cannot Find Model! Please Make Sure You Have Trained A Model!")
            quit()

    @classmethod
    def to_bag_of_words(cls, s: str, words: str) -> numpy.array:
        bag = [0 for _ in range(len(words))]
        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return numpy.array(bag)

    @classmethod
    def chat(cls, text) -> None:
        model = cls.model()
        try:
            with open(os.path.join(model_pth, "intents.json")) as file:
                data = json.load(file)

        except Exception as e:
            logger.warning("Cannot Find intents.json!")
            quit()

        try:
            with open(os.path.join(model_pth, "data.pickle"), "rb") as f:
                words, labels, training, output = pickle.load(f)

        except Exception as e:
            logger.warning("Cannot Find Data! Please Make Sure That You Have Optimized Your Data First!")
            quit()

        results = model.predict([cls.to_bag_of_words(text, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        return random.choice(responses)

        
        else:
            return random.choice(["I dont have a reply for that",
                                  "I am just a prototype I am not finished yet. So I cant answer that for now",
                                  "Sorry I dont have a reply for you know"])
        
