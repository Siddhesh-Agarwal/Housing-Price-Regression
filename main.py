from bitarray import test
import tensorflow as tf
import wandb
from keras import Sequential
from keras.layers import Dense
from regressionmetrics.keras import *

wandb.init(project="Housing-Price-Regression")

data = tf.keras.datasets.boston_housing
(X_train, y_train), (X_test, y_test) = data.load_data(test_split=0.2, seed=42)

X_train = X_train.astype("float32")
y_train = y_train.astype("float32")
X_test = X_test.astype("float32")
y_test = y_test.astype("float32")

model = Sequential(
    [
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(64, activation="relu"),
        Dense(1),
    ]
)
model.compile(optimizer="rmsprop", loss="mae", metrics=[R2CoefScore, AdjR2CoefScore])

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    validation_data=(X_test, y_test),
    validation_freq=1,
    callbacks=[wandb.keras.WandbCallback()],
)
model.save("model.h5")

print(model.evaluate(X_test, y_test))
