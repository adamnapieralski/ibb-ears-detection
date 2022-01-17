import json
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3

from classification_models.tfkeras import Classifiers
ResNet34, _ = Classifiers.get('resnet34')
ResNet18, _ = Classifiers.get('resnet18')
Xception, _ = Classifiers.get('xception')

import recognition_data as data


def prepare_vgg16_model():
    base_model = VGG16(
        input_shape = data.INPUT_SHAPE, include_top = False, weights = 'imagenet')

    base_model.trainable = False
    inputs = tf.keras.Input(shape=data.INPUT_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dense(4096, activation='relu')(x)
    outputs = layers.Dense(data.CLASSES_NUM, activation='softmax')(x)

    return prepare_model(inputs, outputs)

def prepare_inceptionv3_model():
    base_model = InceptionV3(
        input_shape = data.INPUT_SHAPE, include_top = False, weights = 'imagenet')

    inputs = tf.keras.Input(shape=data.INPUT_SHAPE)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(data.CLASSES_NUM, activation='softmax')(x)

    return prepare_model(inputs, outputs)

def prepare_resnet18_model():
    base_model = ResNet18(
        input_shape = (data.IMG_SIZE[0], data.IMG_SIZE[1], 3), include_top = False, weights = 'imagenet')

    inputs = tf.keras.Input(shape=data.INPUT_SHAPE)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(data.CLASSES_NUM, activation='softmax')(x)

    return prepare_model(inputs, outputs)

def prepare_resnet34_model():
    base_model = ResNet34(
        input_shape = (data.IMG_SIZE[0], data.IMG_SIZE[1], 3), include_top = False, weights = 'imagenet')

    inputs = tf.keras.Input(shape=data.INPUT_SHAPE)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(data.CLASSES_NUM, activation='softmax')(x)

    return prepare_model(inputs, outputs)

def prepare_xception_model():
    base_model = Xception(
        input_shape = (data.IMG_SIZE[0], data.IMG_SIZE[1], 3), include_top = False, weights = 'imagenet')

    inputs = tf.keras.Input(shape=data.INPUT_SHAPE)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(data.CLASSES_NUM, activation='softmax')(x)

    return prepare_model(inputs, outputs)

def prepare_model(inputs, outputs):
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics = [
            'accuracy',
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='accuracy_rank_5')
        ])

    return model

def train_model(model, epochs, checkpoint_filepath, csv_filepath, det_type):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    csv_logger_callback = tf.keras.callbacks.CSVLogger(csv_filepath, separator="\t", append=False)

    hist = model.fit(
        data.get_train_generator(det_type=det_type),
        validation_data=data.get_test_generator(det_type=det_type),
        epochs=epochs,
        callbacks=[model_checkpoint_callback, csv_logger_callback]
    )

def get_models_collection():
    return [
        {'name': 'vgg16', 'model': prepare_vgg16_model()},
        {'name': 'resnet18', 'model': prepare_resnet18_model()},
        {'name': 'resnet34', 'model': prepare_resnet34_model()},
        {'name': 'inceptionv3', 'model': prepare_inceptionv3_model()},
        {'name': 'xception', 'model': prepare_xception_model()},
    ]

def evaluate_all():
    results = {}

    for mod in get_models_collection():
        results[mod['name']] = {}
        for det_type_weight in ['perfect', 'detected']:
            for det_type_eval in ['perfect', 'detected']:
                model = mod['model']
                model.load_weights(f'output/models/{mod["name"]}_model_{det_type_weight}/ckpt').expect_partial()
                ev = model.evaluate(data.get_test_generator(det_type=det_type_eval), return_dict=True)
                del ev['loss']
                results[mod['name']][det_type_weight + '_' + det_type_eval] = ev
        with open('output/evaluation_results.json', 'wt') as f:
            json.dump(results, f, indent=2)

def train_all():
    for mod in get_models_collection():
        for det_type in ['perfect', 'detected']:
            train_model(mod['model'], 100, f'output/models/{mod["name"]}_model_{det_type}/ckpt', f'output/{mod["name"]}_model_{det_type}_logs.csv', det_type)

if __name__ == '__main__':
    train_all()
    evaluate_all()