#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.models import load_model

from core.generate_sequences import load_sequences


WINDOW_SIZE = 128
OFFSET = 30


def arg_to_label_6(arg):
    if arg == 2:
        return "stairs_down"
    elif arg == 3:
        return "stairs_up"
    elif arg == 5:
        return "walking"
    else:
        return "other"


def arg_to_label_4(arg):
    if arg == 1:
        return "stairs_down"
    elif arg == 2:
        return "stairs_up"
    elif arg == 3:
        return "walking"
    else:
        return "other"


def calc_floor(sequence):
    curr_floor = 0
    last_action = ""
    for action in sequence:
        if last_action != action:
            if action == "stairs_up":
                curr_floor += 1
            elif action == "stairs_down":
                curr_floor -= 1

            last_action = action

    return curr_floor


def predict(model, sequences, verbose=False):
    successes = 0
    for i, (sequence, floor, labels) in enumerate(sequences):
        flat_list = [item for sublist in sequence for item in sublist]
        n_seq = np.array(flat_list)
        predictions = []
        for j in range(0, len(n_seq), OFFSET):
            if j + WINDOW_SIZE <= len(n_seq):
                window = n_seq[j : j + WINDOW_SIZE]
                exp_window = np.expand_dims(window, axis=0)
                y = model.predict(exp_window)
                argmax = np.argmax(y)
                prediction = arg_to_label_4(argmax)
                predictions.append(prediction)
        # print(predictions)
        predicted_floor = calc_floor(predictions)
        if verbose:
            print(
                f"{i + 1}/{len(sequences)} Predicted floor: {predicted_floor}, Actual floor: {floor}"
            )
        if floor == predicted_floor:
            successes += 1

    return successes


def main():
    print("Loading model...")
    model = load_model("models/uci4-128.h5")
    # model = create_model()

    # print("Reading data...")
    # x_train, y_train = load_data("data/uci_train2.csv")
    # x_test, y_test = load_data("data/uci_test2.csv")

    # # x_train, x_test, y_train, y_test = train_test_split(
    # #     x_train, y_train, test_size=0.1, random_state=np.random.randint(0, 42)
    # # )

    # model.fit(
    #     x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test)
    # )

    print("Loading sequences...")
    sequences_floor_labels = load_sequences("pickle/sequences_2.pkl")

    print("Predicting sequences floor...")
    successes = predict(model, sequences_floor_labels, verbose=True)

    success_rate = float(successes) / float(len(sequences_floor_labels))

    print(f"Accuracy: {success_rate:.2f} ({successes}/{len(sequences_floor_labels)})")


if __name__ == "__main__":
    main()
