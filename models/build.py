from munch import Munch

from models.classifier import Classifier


def build_model(args):
    classifier = Classifier(args)
    nets = Munch(classifier=classifier)
    return nets
