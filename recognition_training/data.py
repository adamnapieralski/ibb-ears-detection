from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE = (96,96)
INPUT_SHAPE = (96,96,3)
RESCALE = 1./255
BATCH_SIZE = 25
CLASSES_NUM = 100


def get_train_generator(det_type='perfect', seed=123, batch_size=25):
    gen = ImageDataGenerator(
        rescale=RESCALE,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True
    )
    if det_type == 'perfect':
        return gen.flow_from_directory(
            '../data/perfectly_detected_ears/train_dir',
            target_size=IMG_SIZE,
            batch_size=batch_size,
            shuffle=True,
            class_mode='sparse',
            seed=seed
        )
    else:
        pass

def get_test_generator(det_type='perfect', seed=123, batch_size=25):
    gen = ImageDataGenerator(rescale=RESCALE)
    if det_type == 'perfect':
        return gen.flow_from_directory(
            '../data/perfectly_detected_ears/test_dir',
            target_size=IMG_SIZE,
            batch_size=batch_size,
            shuffle=True,
            class_mode='sparse',
            seed=seed
        )
    else:
        pass