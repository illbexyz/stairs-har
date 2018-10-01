import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense, GRU

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# EPOCHS = 50
# BATCH_SIZE = 128
# SEQUENCE_LENGTH = 128
LABELS = ["walking", "stairs_down", "stairs_up", "other"]
# LABELS = ["walking", "stairs_down", "stairs_up", "sitting", "standing", "laying"]


def load_data(filename, sequence_length, labels):
    df = pd.read_csv(filename)
    lb = LabelBinarizer().fit(labels)

    x = []
    y = []

    for i in df.id.unique():
        x_subset = df[df.id == i][["x", "y", "z"]]
        y_subset = df[df.id == i][["label"]]
        count = y_subset.count()[0]
        curr_label = y_subset.iloc[0]["label"]
        np_x = np.array(x_subset.values)
        for j in range(0, count, sequence_length):
            if j + sequence_length <= count:
                x.append(np_x[j : j + sequence_length])
                y.append(curr_label)

    np_x = np.array(x)
    np_y = np.array(lb.transform(y))

    return np_x, np_y


def create_model(sequence_length, labels):
    model = Sequential()
    model.add(GRU(128, activation="relu", input_shape=(sequence_length, 3)))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(len(labels), activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()

    return model


def train(model, x_train, y_train, x_test, y_test, epochs, batch_size, save_path=""):
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
    )

    if save_path != "":
        model.save(save_path)

    # summarize history for accuracy
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


def main():
    sequence_length = 128

    model = create_model(sequence_length, LABELS)

    # x_train, x_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.2, random_state=np.random.randint(0, 42)
    # )

    x_train, y_train = load_data("../data/uci_train_4_labels.csv", sequence_length, LABELS)
    x_test, y_test = load_data("../data/uci_test_4_labels.csv", sequence_length, LABELS)

    filepath = f"../models/uci-{sequence_length}.h5"

    # train(
    #     model,
    #     x_train,
    #     y_train,
    #     x_test,
    #     y_test,
    #     epochs=50,
    #     batch_size=32,
    #     filepath=filepath,
    # )

    i = 0
    best_acc = 0
    while True:
        history = model.fit(
            x_train, y_train, epochs=1, batch_size=256, validation_data=(x_test, y_test)
        )
        acc = history.history["val_acc"][0]
        if acc > best_acc:
            model.save("../models/uci4-128.h5")
            best_acc = acc
            print(f"New best accuracy found: {best_acc}")
            print("Saving the model...")
        i = i + 1


if __name__ == "__main__":
    main()
