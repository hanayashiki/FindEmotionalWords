from get_data_cnn import *
from cnn_processor import *

def run_training_set():
    x_data, jumped = getTestData(0, 20)
    y_list = process(x_data)
    f = open(pair_indices, "w", encoding='gbk', newline='')
    dump_csv_writer = csv.writer(f, dialect="excel")

    total = 0
    for pair_list in y_list:
        while len(jumped)>0 and (total == jumped[0] - 1):
            jumped = jumped[1:]
            total += 1
            dump_csv_writer.writerow(["Not judged"])
        total += 1
        emotion_list = []
        object_list = []
        for (emotion, object) in pair_list:
            emotion_list.append(emotion)
            object_list.append(object)
        list_ = [object_list, emotion_list]
        dump_csv_writer.writerow(list_)

if __name__ == '__main__':
    run_training_set()
