import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Sequential

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import ResNet50

import data


def prepare_vgg_model():
    base_model = VGG16(
        input_shape = data.INPUT_SHAPE, include_top = False, weights = 'imagenet')

    base_model.trainable = False

    inputs = tf.keras.Input(shape=data.INPUT_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
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

    base_model.trainable = False

    inputs = tf.keras.Input(shape=data.INPUT_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(data.CLASSES_NUM, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics = ['accuracy'])

    return model

def prepare_resnet50_model():
    base_model = InceptionV3(
        input_shape = (data.IMG_SIZE[0], data.IMG_SIZE[1], 3), include_top = False, weights = 'imagenet')

    base_model.trainable = False

    inputs = tf.keras.Input(shape=data.INPUT_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
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
        monitor='val_loss',
        save_best_only=True)

    csv_logger_callback = tf.keras.callbacks.CSVLogger(csv_filepath, separator="\t", append=False)

    hist = model.fit(
        data.get_train_generator(det_type=det_type),
        validation_data=data.get_test_generator(det_type=det_type),
        epochs=epochs,
        callbacks=[model_checkpoint_callback, csv_logger_callback]
    )

if __name__ == '__main__':
    vgg_model = prepare_vgg_model()
    inception_model = prepare_inceptionv3_model()
    resnet_model = prepare_inceptionv3_model()

    train_model(vgg_model, 30, 'output/vgg_model/ckpt', 'output/vgg_model_logs_4.csv', 'perfect')
    train_model(inception_model, 30, 'output/inception_model/ckpt', 'output/inception_model_logs_4.csv', 'perfect')
    train_model(resnet_model, 30, 'output/resnet_model/ckpt', 'output/resnet_model_logs_4.csv', 'perfect')