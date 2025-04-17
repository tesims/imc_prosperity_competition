import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam

def create_generator(hidden=32):
    m = Sequential([
        Dense(hidden, activation="relu", input_dim=1),
        Dense(hidden, activation="relu"),
        Dense(1, activation="linear")
    ])
    return m

def create_discriminator(hidden=32):
    m = Sequential([
        Dense(hidden, activation="relu", input_dim=1),
        Dense(hidden, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    m.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])
    return m

def create_gan(gen, disc, lr=1e-4):
    disc.trainable = False
    gan = Sequential([gen, disc])
    gan.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=lr))
    return gan
