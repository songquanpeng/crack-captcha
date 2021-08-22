from munch import Munch

from models.classifier import Classifier
from models.VGG import VGG


def build_model(args):
    if args.which_model == "origin":
        classifier = Classifier(args)
    elif args.which_model == 'vgg':
        classifier = VGG(args)
    else:
        raise NotImplementedError
    nets = Munch(classifier=classifier)
    return nets
