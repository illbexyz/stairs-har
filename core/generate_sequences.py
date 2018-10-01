import numpy as np
import pandas as pd

np.random.seed(0)


def floor(sequence):
    curr_floor = 0
    for action in sequence:
        if action == "walking":
            curr_floor += 0
        elif action == "stairs_up":
            curr_floor += 1
        elif action == "stairs_down":
            curr_floor -= 1
        else:
            raise Exception(f"Unexpected action: {action}")
    return curr_floor


def generate_labels(labels, sequence_lengths):
    sequence_labels = []
    sequence_length = np.random.choice(sequence_lengths)

    initial_action = np.random.choice(labels)
    sequence_labels.append(initial_action)
    for _ in range(sequence_length - 1):
        if sequence_labels[-1] == "walking":
            # pick_from = ["stairs_up", "stairs_down"]
            pick_from = ["stairs_up"]
        else:
            pick_from = ["walking"]
        action = np.random.choice(pick_from)
        sequence_labels.append(action)
    return sequence_labels


def generate_sequence(filename, sequence_labels):
    df = pd.read_csv(filename)
    walking_df = df[df.label == "walking"]
    stairs_up_df = df[df.label == "stairs_up"]
    stairs_down_df = df[df.label == "stairs_down"]

    sequence = []

    for action in sequence_labels:
        if action == "walking":
            curr_df = walking_df
            action_length = 50
        elif action == "stairs_up":
            curr_df = stairs_up_df
            action_length = 256
        elif action == "stairs_down":
            curr_df = stairs_down_df
            action_length = 256
        else:
            raise Exception(f"Unexpected action: {action}")

        random_id = np.random.choice(curr_df["id"].unique())
        m_df = curr_df[curr_df.id == random_id][["x", "y", "z"]]

        values = m_df.iloc[0:action_length].values
        sequence.append(values)

    return sequence


def generate_sequences(n, filepath, labels, sequence_lengths):
    sequences_with_floor_labels = []
    for i in range(n):
        print(f"{i + 1}/{n}")
        l = generate_labels(labels, sequence_lengths)
        s = generate_sequence(filepath, l)
        f = floor(l)
        # print(f, l)
        # print()
        sequences_with_floor_labels.append((s, f, l))

    return sequences_with_floor_labels


def save_sequences(sequences, filepath):
    import pickle

    with open(filepath, "wb") as f:
        pickle.dump(sequences, f)


def load_sequences(filename):
    import pickle

    with open(filename, "rb") as f:
        ss = pickle.load(f)

        # for (sequence, floor, labels) in ss:
        #     print(floor, labels)

    return ss


if __name__ == "__main__":
    labels = ["stairs_up", "stairs_down", "walking"]
    sequence_lengths = [3]
    for i in range(2, 6):
        sequences = generate_sequences(
            1000, "../data/uci_test_6_labels.csv", labels, [i]
        )
        save_sequences(sequences, f"../pickle/sequences_{i}.pkl")

    sequences = generate_sequences(
        1000, "../data/uci_test_6_labels.csv", labels, [2, 3, 4, 5]
    )
    save_sequences(sequences, "../pickle/sequences_2345.pkl")
