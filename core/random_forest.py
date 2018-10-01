import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier

# from core.generate_sequences import generate_sequences

SEQUENCE_LENGTH = 128

LABELS_4 = ["walking", "stairs_down", "stairs_up", "other"]
LABELS_6 = ["walking", "stairs_down", "stairs_up", "sitting", "standing", "laying"]


def load_data(filename):
    df = pd.read_csv(filename)

    x = []
    y = []

    for i in df.id.unique():
        x_subset = df[df.id == i][["x", "y", "z"]]
        y_subset = df[df.id == i][["label"]]

        count = y_subset.count()[0]
        curr_label = y_subset.iloc[0]["label"]
        np_x = np.array(x_subset.values)
        for j in range(0, count, SEQUENCE_LENGTH):
            if j + SEQUENCE_LENGTH <= count:
                x.append(
                    [
                        extract_features(np_x[j : j + SEQUENCE_LENGTH, 0]),
                        extract_features(np_x[j : j + SEQUENCE_LENGTH, 1]),
                        extract_features(np_x[j : j + SEQUENCE_LENGTH, 2]),
                    ]
                )
                y.append(curr_label)

    np_x = np.array(x)
    np_y = np.array(y)

    return np_x.reshape(len(np_x), -1), np_y


def extract_features(data):
    x_min = np.min(data)
    x_max = np.max(data)
    x_std = np.std(data)
    x_mean = np.mean(data)
    return [x_min, x_max, x_std, x_mean]


def train_random_forest(x_train, y_train):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    return model


def eval_random_forest(model, x_test, y_test):
    return model.score(x_test, y_test)


def main():
    x_train, y_train = load_data("../data/uci_train_6.csv")
    x_test, y_test = load_data("../data/uci_test_6.csv")

    model = train_random_forest(x_train, y_train)
    score = eval_random_forest(model, x_test, y_test)
    print(f"Score: {score}")

    # sequences_with_floor = generate_sequences(10)
    #
    # for (sequence, floor) in sequences_with_floor:
    #     for subsequence in sequence:
    #         features = extract_features(subsequence)
    #         x = np.array(features).reshape(len(features), -1)
    #         y = clf.predict(x)
    #         print(y)


if __name__ == "__main__":
    main()
