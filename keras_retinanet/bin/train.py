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
import functools
import os
import sys
import warnings

import keras
import keras.preprocessing.image
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras.optimizers import SGD, Adam, RMSprop, Nadam, Adamax, Adadelta, Adagrad
# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import layers  # noqa: F401
from .. import losses
from .. import models
from ..callbacks import RedirectModel
from ..callbacks.eval import Evaluate
from ..models.retinanet import retinanet_bbox, AnchorParameters
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.csv_generator_multi import CSVGeneratorMULTI
from ..preprocessing.kitti import KittiGenerator
from ..preprocessing.open_images import OpenImagesGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..utils.anchors import make_shapes_callback, anchor_targets_bbox
from ..utils.keras_version import check_keras_version
from ..utils.model import freeze as freeze_model
from ..utils.transform import random_transform_generator
import random
import numpy as np
from tensorflow import set_random_seed
seed = 10
random.seed(seed)
np.random.seed(seed)
set_random_seed(10)

def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def list_callbacks(parse):
    return np.array([eval(s) for s in parse.split(',')])
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights, args, multi_gpu=0, freeze_backbone=False, shape=(None, None, 3), opt=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)):
    modifier = freeze_model if freeze_backbone else None

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, modifier=modifier, shape=shape, num_anchors=len(args.ratio)*len(args.scale)), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model          = model_with_weights(backbone_retinanet(num_classes, modifier=modifier, shape=shape, num_anchors=len(args.ratio)*len(args.scale)), weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    anchors_param = AnchorParameters(sizes=args.size, strides=args.stride, ratios=args.ratio, scales=args.scale)
    prediction_model = retinanet_bbox(model=model, anchor_parameters=anchors_param, P2_layer=args.P2)

    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer= opt
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args, tensorboard_image=None):
    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        if args.dataset_type == 'coco':
            from ..callbacks.coco import CocoEval

            # use prediction model for evaluation
            evaluation = CocoEval(validation_generator, tensorboard=tensorboard_callback)
        else:
            evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback, tensorboard_image=tensorboard_image, number_images=args.tensorboxes, separate_channels=args.tensorboxes_channels)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{name}_{dataset_type}.h5'.format(name=args.name, dataset_type=args.dataset_type)
            ),
            verbose=1,
            save_best_only=True,
            monitor="mAP",
            mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)
    if args.val_loss:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.5,
        patience = 2,
        verbose  = 1,
        mode     = 'min',
        epsilon  = 0.0001,
        cooldown = 0,
        min_lr   = 0
        ))
    else:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor  = 'mAP',
            factor   = 0.5,
            patience = 2,
            verbose  = 1,
            mode     = 'max',
            epsilon  = 0.0001,
            cooldown = 0,
            min_lr   = 0
        ))

    return callbacks


def create_generators(args):
    # create random transform generator for augmenting training data
    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        train_generator = CocoGenerator(
            args.coco_path,
            'train2017',
            transform_generator=transform_generator,
            batch_size=args.batch_size,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            args=args,
        )

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            batch_size=args.batch_size,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            args=args,
        )
    elif args.dataset_type == 'pascal':
        train_generator = PascalVocGenerator(
            args.pascal_path,
            'trainval',
            transform_generator=transform_generator,
            batch_size=args.batch_size,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            args=args,
        )

        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test',
            batch_size=args.batch_size,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            args=args,
        )
    elif args.dataset_type == 'csv':
        train_generator = CSVGenerator(
            args.annotations,
            args.classes,
            base_dir=args.dataset_dir,
            transform_generator=transform_generator,
            batch_size=args.batch_size,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            args=args,
        )
        print('training generator loaded')
        if args.val_annotations:
            validation_generator = CSVGenerator(
                args.val_annotations,
                args.classes,
                base_dir=args.dataset_dir,
                batch_size=args.batch_size,
                image_min_side=args.image_min_side,
                image_max_side=args.image_max_side,
                args=args,
            )
            print('validating generator loaded')
        else:
            validation_generator = None
    elif args.dataset_type == 'csv_multi':
        train_generator = CSVGeneratorMULTI(
            args.annotations,
            args.classes,
            base_dir=args.dataset_dir,
            transform_generator=transform_generator,
            batch_size=args.batch_size,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            args=args,
        )

        if args.val_annotations:
            validation_generator = CSVGeneratorMULTI(
                args.val_annotations,
                args.classes,
                base_dir=args.dataset_dir,
                batch_size=args.batch_size,
                image_min_side=args.image_min_side,
                image_max_side=args.image_max_side,
                args=args,
            )
        else:
            validation_generator = None
            
    elif args.dataset_type == 'oid':
        train_generator = OpenImagesGenerator(
            args.main_dir,
            subset='train',
            version=args.version,
            labels_filter=args.labels_filter,
            annotation_cache_dir=args.annotation_cache_dir,
            fixed_labels=args.fixed_labels,
            transform_generator=transform_generator,
            batch_size=args.batch_size,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            args=args,
        )

        validation_generator = OpenImagesGenerator(
            args.main_dir,
            subset='validation',
            version=args.version,
            labels_filter=args.labels_filter,
            annotation_cache_dir=args.annotation_cache_dir,
            fixed_labels=args.fixed_labels,
            batch_size=args.batch_size,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            args=args,
        )
    elif args.dataset_type == 'kitti':
        train_generator = KittiGenerator(
            args.kitti_path,
            subset='train',
            transform_generator=transform_generator,
            batch_size=args.batch_size,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            args=args,
        )

        validation_generator = KittiGenerator(
            args.kitti_path,
            subset='val',
            batch_size=args.batch_size,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            args=args,
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator

def optimizers(args):
    if args.optimizer == 'adamax':
        opt = Adamax(lr=args.lr, clipnorm=args.clip_norm)
    elif args.optimizer == 'sgd':
        opt = SGD(lr=args.lr, decay=0.00005, momentum=0.9, nesterov=True, clipnorm=args.clip_norm)
    elif args.optimizer == 'adadelta':
        opt = Adadelta(lr=args.lr, clipnorm=args.clip_norm)
    elif args.optimizer == 'adagrad':
        opt = Adagrad(lr=args.lr, clipnorm=args.clip_norm)
    elif args.optimizer == 'nadam':
        opt = Nadam(lr=args.lr, clipnorm=args.clip_norm)
    elif args.optimizer == 'rmsprop':
        opt = RMSprop(lr=args.lr, clipnorm=args.clip_norm)
        
    else:
        opt = Adam(lr=args.lr, clipnorm=args.clip_norm)
        
    print(args.optimizer + ':' + str(args.lr) + ' ' + str(args.clip_norm))
    return opt
def check_args(parsed_args):
    """
    Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    :param parsed_args: parser.parse_args()
    :return: parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))
     
    parsed_args.pyramid_levels = [2, 3, 4, 5, 6, 7] if parsed_args.P2 else [3, 4, 5, 6, 7]
    parsed_args.stride = [2 ** x for x in parsed_args.pyramid_levels]
    parsed_args.size = [2 ** (x + 2) for x in parsed_args.pyramid_levels]
    return parsed_args


def parse_args(args):
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    kitti_parser = subparsers.add_parser('kitti')
    kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')

    def csv_list(string):
        return string.split(',')

    oid_parser = subparsers.add_parser('oid')
    oid_parser.add_argument('main_dir', help='Path to dataset directory.')
    oid_parser.add_argument('--version',  help='The current dataset version is v4.', default='v4')
    oid_parser.add_argument('--labels-filter',  help='A list of labels to filter.', type=csv_list, default=None)
    oid_parser.add_argument('--annotation-cache-dir', help='Path to store annotation cache.', default='.')
    oid_parser.add_argument('--fixed-labels', help='Use the exact specified labels.', default=False)

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')
    csv_parser.add_argument('--dataset_dir', help='path to the dataset', default=None)
    
    csv_multi_parser = subparsers.add_parser('csv_multi')
    csv_multi_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    csv_multi_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    csv_multi_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')
    csv_multi_parser.add_argument('--dataset_dir', help='path to the dataset', default=None)
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone',        help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',      help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',       help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',          help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps',           help='Number of steps per epoch.', type=int, default=None)
    parser.add_argument('--snapshot-path',   help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',    help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',   help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--channels', dest="channels", help='Number of channels in the input data', type=int, default=3)
    parser.add_argument('--remove-mean', dest='mean', help='Preprocessing : remove mean, default False', default='False')
    parser.add_argument('--normalize', dest='norm', help='Preprocessing : normalize image, if 1 then value = [0, 1], if -1 then value = [-1, 1] else no normalization', type=int, default=0)
    parser.add_argument('--optimizer', help='choose optimizer : sgd, adam, rmsprop, adagrad, adadelta, adamax, nadam. default=adam', default='adam')   
    parser.add_argument('--lr',  help='Choose learning rate value, default = 0.00001.', type=float, default=1e-5)
    parser.add_argument('--clip-norm',  help='Choose clip norm value, default = 0.00001.', type=float, default=0.001)
    parser.add_argument('--name',  help='Choose name of your saved file, default = 0.00001.', default=None)
    parser.add_argument('--tensorboxes', help='Number of images to store on tensorboard', type=int, default=0)
    parser.add_argument('--tensorboxes-channels', help='display each channels on tensorboard. If --channels is different from 3, True forced.', action='store_true')
    parser.add_argument('--use-val-loss', dest='val_loss',  help='Compute validation loss', action='store_true')
    parser.add_argument('--use-P2', dest='P2', help='Use P2 layer (more consuming) for training and testing in the FPN (only for resnet).', action='store_true')
    parser.add_argument('--scale', help='list of the scale use in the network.', type=list_callbacks, default='2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)')
    parser.add_argument('--ratio', help='list of the ratio use in the network.', type=list_callbacks, default='0.5, 1, 2')
    return check_args(parser.parse_args(args))


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    if args.channels != 3:
        args.tensorboxes_channels = True
    print('ratio : ', args.ratio, 'scale :', args.scale, 'P2 : ', args.P2, 'pyramid levels : ', args.pyramid_levels, ' stride :', args.stride, 'sizes : ', args.size)
    writer = tf.summary.FileWriter(args.tensorboard_dir+'/image')
    # create object that stores backbone information
    backbone = models.backbone(args.backbone, P2=args.P2)
    if args.name is None:
        args.name = args.backbone
    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    train_generator, validation_generator = create_generators(args)
    if args.steps is None:
        args.steps = int(round(train_generator.size()/args.batch_size))
        
    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model            = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model   = model
        anchors_param = AnchorParameters(sizes=args.size, strides=args.stride, ratios=args.ratio, scales=args.scale)
        prediction_model = retinanet_bbox(model=model, anchor_parameters=anchors_param, P2_layer=args.P2)
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            args=args,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            shape=(None, None, int(args.channels)),
            opt=optimizers(args)
        )

    # print model summary
    model.summary()

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        compute_anchor_targets = functools.partial(anchor_targets_bbox, shapes_callback=make_shapes_callback(model))
        train_generator.compute_anchor_targets = compute_anchor_targets
        if validation_generator is not None:
            validation_generator.compute_anchor_targets = compute_anchor_targets

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        args,
        tensorboard_image=writer
    )
    #print(callbacks)
    print("remove mean :", args.mean, ' normalize :', args.norm, 'transform: ', args.random_transform, 'separate channels : ', args.tensorboxes_channels)
    # start training
    if args.val_loss:
        
        training_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=args.steps,
            validation_data=validation_generator,
            validation_steps= int(round(validation_generator.size()/args.batch_size)),
            epochs=args.epochs,
            verbose=1,
            callbacks=callbacks
            
        )
    else:
        training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps,
        #validation_data=validation_generator,
        #validation_steps= int(round(validation_generator.size()/args.batch_size)),
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks
        
    )


if __name__ == '__main__':
    main()
