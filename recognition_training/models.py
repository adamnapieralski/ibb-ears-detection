import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import ResNet50

from classification_models.tfkeras import Classifiers
ResNet34, _ = Classifiers.get('resnet34')
ResNet18, _ = Classifiers.get('resnet18')
Xception, _ = Classifiers.get('xception')

import data


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
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics = ['accuracy'])

    return model

def prepare_inceptionv3_model():
    base_model = InceptionV3(
        input_shape = data.INPUT_SHAPE, include_top = False, weights = 'imagenet')

    inputs = tf.keras.Input(shape=data.INPUT_SHAPE)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(data.CLASSES_NUM, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics = ['accuracy'])

    return model

def prepare_resnet50_model():
    base_model = ResNet50(
        input_shape = (data.IMG_SIZE[0], data.IMG_SIZE[1], 3), include_top = False, weights = 'imagenet')

    inputs = tf.keras.Input(shape=data.INPUT_SHAPE)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(data.CLASSES_NUM, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics = ['accuracy'])

    return model

def prepare_resnet18_model():
    base_model = ResNet18(
        input_shape = (data.IMG_SIZE[0], data.IMG_SIZE[1], 3), include_top = False, weights = 'imagenet')

    inputs = tf.keras.Input(shape=data.INPUT_SHAPE)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(data.CLASSES_NUM, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics = ['accuracy'])

    return model

def prepare_resnet34_model():
    base_model = ResNet34(
        input_shape = (data.IMG_SIZE[0], data.IMG_SIZE[1], 3), include_top = False, weights = 'imagenet')

    inputs = tf.keras.Input(shape=data.INPUT_SHAPE)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(data.CLASSES_NUM, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics = ['accuracy'])

    return model

def prepare_xception_model():
    base_model = Xception(
        input_shape = (data.IMG_SIZE[0], data.IMG_SIZE[1], 3), include_top = False, weights = 'imagenet')

    inputs = tf.keras.Input(shape=data.INPUT_SHAPE)
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(data.CLASSES_NUM, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics = ['accuracy'])

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

def train_all():
    for det_type in ['perfect', 'detected']:
        resnet18_model = prepare_resnet18_model()
        train_model(resnet18_model, 100, f'output/resnet18_model_{det_type}/ckpt', 'output/resnet18_model_{det_type}_logs.csv', det_type)

        resnet34_model = prepare_resnet34_model()
        train_model(resnet34_model, 100, f'output/resnet34_model_{det_type}/ckpt', 'output/resnet34_model_{det_type}_logs.csv', det_type)

        resnet50_model = prepare_resnet50_model()
        train_model(resnet50_model, 100, f'output/resnet50_model_{det_type}/ckpt', 'output/resnet50_model_{det_type}_logs.csv', det_type)

        inceptionv3_model = prepare_inceptionv3_model()
        train_model(inceptionv3_model, 100, f'output/inceptionv3_model_{det_type}/ckpt', 'output/inceptionv3_model_{det_type}_logs.csv', det_type)

        vgg16_model = prepare_vgg16_model()
        train_model(vgg16_model, 100, f'output/vgg16_model_{det_type}/ckpt', 'output/vgg16_model_logs_{det_type}.csv', det_type)

        xception_model = prepare_xception_model()
        train_model(xception_model, 100, f'output/xception_model_{det_type}/ckpt', 'output/xception_model_{det_type}_logs.csv', det_type)

if __name__ == '__main__':
    train_all()