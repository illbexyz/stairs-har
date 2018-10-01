file_out = "data/uci_train_3.csv"


def read_values(filename):
    content = ""
    with open(filename, "r") as f:
        content += f.read()

    lines = content.split("\n")

    valuess = [l.split(" ")[1:] for l in lines]

    m_valuess = []
    for values in valuess:
        vvv = list(filter(lambda v: v != "", values))
        m_valuess.append(vvv)

    return m_valuess[:-1]


def read_labels(filename):
    content = ""
    with open(filename, "r") as f:
        content += f.read()

    values = content.split("\n")

    return values[:-1]


def convert_label(label):
    if label == "1":
        return "walking"
    elif label == "2":
        return "stairs_up"
    elif label == "3":
        return "stairs_down"
    elif label == "4":
        return "sitting"
    elif label == "5":
        return "standing"
    elif label == "6":
        return "laying"
    else:
        raise Exception("Unexpected label")
    # else:
    #     return "other"


def convert_label_3(label):
    if label == "1":
        return "walking"
    elif label == "2":
        return "stairs_up"
    elif label == "3":
        return "stairs_down"
    else:
        return "other"


if __name__ == "__main__":
    xss = read_values("data/uci/total_acc_x_train.txt")
    yss = read_values("data/uci/total_acc_y_train.txt")
    zss = read_values("data/uci/total_acc_z_train.txt")
    labels = read_labels("data/uci/y_train.txt")
    users = read_labels("data/uci/subject_train.txt")

    # xss = read_values("data/uci/total_acc_x_test.txt")
    # yss = read_values("data/uci/total_acc_y_test.txt")
    # zss = read_values("data/uci/total_acc_z_test.txt")
    # labels = read_labels("data/uci/y_test.txt")
    # users = read_labels("data/uci/subject_test.txt")

    if len(xss) == len(yss) == len(zss) == len(labels) == len(users):
        print("ok")
    else:
        print(len(xss), len(yss), len(zss), len(labels), len(users))
        exit(1)

    with open(file_out, "w") as f:
        f.write("id,label,x,y,z,user\n")
        for m_id, (xs, ys, zs, l, u) in enumerate(zip(xss, yss, zss, labels, users)):
            # if len(xs) == len(ys) == len(zs):
            #     print("ok")
            # else:
            #     print(len(xs), len(ys), len(zs))
            #     exit(1)
            for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
                f.write(f"{m_id},{convert_label_3(l)},{x},{y},{z},{u}\n")
