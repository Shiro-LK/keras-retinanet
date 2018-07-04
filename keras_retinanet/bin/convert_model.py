#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys
import numpy as np
# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models


def list_callbacks(parse):
    return np.array([eval(s) for s in parse.split(',')])

def boolean_string(parse):
    if parse not in {'False', 'True'}:
        return None
    return parse == 'True'
def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for converting a training model to an inference model.')

    parser.add_argument('model_in', help='The model to convert.')
    parser.add_argument('model_out', help='Path to save the converted model to.')
    parser.add_argument('--backbone', help='The backbone of the model to convert.', default='resnet50')
    parser.add_argument('--no-nms', help='Disables non maximum suppression.', dest='nms', action='store_false')
    parser.add_argument('--use-P2', help='Use P2 layer in the network for prediction.', dest='P2', type=boolean_string, default=None)
    parser.add_argument('--scales', help='Choose the different scales for the anchor.', dest='scales', type=list_callbacks, default='2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)')
    parser.add_argument('--ratios', help='Choose the different ratios for the anchor.', dest='ratios', type=list_callbacks, default='0.5,1,2')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    print('P2 : ', args.P2)
    print(args.scales, args.ratios)
    # load and convert model
    if args.P2 is not None:
        pyramid_levels = [2, 3, 4, 5, 6, 7] if args.P2 == True else [3, 4, 5, 6, 7]
        strides = [2 ** x for x in pyramid_levels]
        sizes = [2 ** (x + 2) for x in pyramid_levels]
        model = models.load_model(args.model_in, convert=True, backbone_name=args.backbone, nms=args.nms, P2_layer=args.P2, sizes=sizes, strides=strides, ratios=args.ratios, scales=args.scales)      
    else:
    
        model = models.load_model(args.model_in, convert=True, backbone_name=args.backbone, nms=args.nms)

    # save model
    model.save(args.model_out)


if __name__ == '__main__':
    main()
