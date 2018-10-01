import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier

from core.generate_sequences import generate_sequences

SEQUENCE_LENGTH = 128

LABELS_4 = ["walking", "stairs_down", "stairs_up", "other"]
LABELS_6 = ["walking", "stairs_down", "stairs_up", "sitting", "standing", "laying"]


def load_data(filename):
    df = pd.read_csv(filename)
    # df = df[df.user == 22]
    # lb = LabelBinarizer().fit(labels)

    # print(lb.classes_)

    n_sequences = df[["id"]].max()[0] + 1

    x = []
    y = []

    for i in range(n_sequences):
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

    return np_x, np_y


def extract_features(data):
    x_min = np.min(data)
    x_max = np.max(data)
    x_std = np.std(data)
    x_mean = np.mean(data)
    return [x_min, x_max, x_std, x_mean]


def main():
    x_train, y_train = load_data("../data/uci_train2.csv")

    # x_train, x_test, y_train, y_test = train_test_split(
    #     x, y, test_size=0.2, random_state=np.random.randint(0, 42)
    # )

    x_test, y_test = load_data("../data/uci_test2.csv")

    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

    clf = RandomForestClassifier()

    for _ in range(1):
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
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
