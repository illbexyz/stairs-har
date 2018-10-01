# file = "../data/WISDM_ar_v1.1_raw.txt"
# file_out = "../data/wisdm_ar.csv"

file = "../data/WISDM_at_v2.0_raw.txt"
file_out = "../data/wisdm_at.csv"

# allowed_labels = ["Walking", "Upstairs", "Downstairs"]
allowed_labels = ["Walking", "Stairs"]


def convert_label(label):
    label_dict = {
        "Walking": "walking",
        # "Upstairs": "stairs_up",
        # "Downstairs": "stairs_down",
        "Stairs": "stairs"
    }
    if label in allowed_labels:
        return label_dict[label]
    else:
        return "other"


data = ""
with open(file, "r") as f:
    data += f.read()

lines = data.split(";\n")

with open(file_out, "w") as f:
    f.write("id,label,x,y,z,ms,phone,position,floors\n")
    curr_id = -1
    last_time = int(int(lines[0].split(",")[2]))
    initial_time = 0
    last_user = ""
    last_label = ""
    for line in lines:
        values = line.split(",")
    if len(values) == 6:
            [user, label, time, x, y, z] = values
            time = int(int(time))
            time_diff = time - last_time
            # print(time_diff)
            if time_diff != 0 and x != "0" and y != "0" and z != "0":
                if user != last_user or last_label != label or time_diff > 1000 or time_diff < 0:
                    curr_id += 1
                    last_user = user
                    last_label = label
                    initial_time = time

                elapsed_ms = time - initial_time
                # print(elapsed_ms)
                last_time = time
                f.write(f"{curr_id},{convert_label(label)},{x},{y},{z},{elapsed_ms},null,null,null\n")
