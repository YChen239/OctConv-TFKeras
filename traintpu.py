import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, History
from tensorflow.contrib.tpu.python.tpu import keras_support
from keras import backend as K
from models import *

from keras.datasets import cifar10
from keras.utils import to_categorical
import pickle, os, time

def lr_scheduler(epoch):
    x = 0.1
    if epoch >= 100: x /= 5.0
    if epoch >= 150: x /= 5.0
    return x

def train(alpha, type):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    train_gen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, 
                                    width_shift_range=4.0/32.0, height_shift_range=4.0/32.0)
    test_gen = ImageDataGenerator(rescale=1.0/255)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    tf.logging.set_verbosity(tf.logging.FATAL)

    if type == 'resnet':
        if alpha <= 0:
            model = create_normal_wide_resnet()
        else:
            model = create_octconv_wide_resnet(alpha)


    if type == 'densenet':
        img_dim = (32, 32, 3)
        if alpha <= 0:
            model = DenseNet(input_shape=img_dim, nb_classes=10).build_model()
        else:
            model = DenseNet(input_shape=img_dim, nb_classes=10).build_octave_model(alpha)
    # model.compile(SGD(0.1, momentum=0.9), "categorical_crossentropy", ["acc"])
    optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    model.summary()

    # convert to tpu model
    tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    batch_size = 128
    scheduler = LearningRateScheduler(lr_scheduler)
    hist = History()

    start_time = time.time()
    model.fit_generator(train_gen.flow(X_train, y_train, batch_size, shuffle=True),
                        steps_per_epoch=X_train.shape[0]//batch_size,
                        validation_data=test_gen.flow(X_test, y_test, batch_size, shuffle=False),
                        validation_steps=X_test.shape[0]//batch_size,
                        callbacks=[scheduler, hist], max_queue_size=5, epochs=150)
    elapsed = time.time() - start_time
    print(elapsed)

    history = hist.history
    history["elapsed"] = elapsed

    with open(f"octconv_alpha_{alpha}.pkl", "wb") as fp:
        pickle.dump(history, fp)

if __name__ == "__main__":

    type = 'densenet'
    #type = 'resnet'
    alhpa = 0
    train(alhpa, type)

    with open("octconv_alpha_{alpha}.pkl", "rb") as fp:
        data = pickle.load(fp)
        print(f"Max test accuracy = {max(data['val_acc']):.04}")
