import random
from operator import itemgetter

from data_enrich import DataEnrich


class DataLoader():

    label_mapping = {
        'car': 0,
        'walk': 1,
        'bus': 2,
        'train': 3,
        'boat': 4,
        'subway': 5,
        'motorcycle': 6,
        'run': 7,
        'airplane': 8,
        'bike': 9,
        'taxi': 10
    }

    fields_to_feed = ["altitude", "dist", "speed", "accel"]
    labels_to_remove = ["boat", "motorcycle", "airplane"]


    def __init__(self, test_ratio=0.2, val_ratio=0.1, batchsize=8):
        de = DataEnrich()
        self._raw = de.get_enriched_data(True)
        self._test_ratio = test_ratio
        self._val_ratio = val_ratio
        self._batchsize = batchsize
        self.prepared_data = []
        self.prepared_labels = []

    def _remove_traj_containing_labels(self, labels):
        cleaned = []
        for elem in self._raw:
            if all(x not in list(elem["label"]) for x in self.labels_to_remove):
                cleaned.append(elem)
        self._raw = cleaned

    def _merge_labels(self, target_label, label_to_remove):
        for elem in self._raw:
            if label_to_remove in list(elem["label"]):
                elem["label"] = elem["label"].replace(to_replace=label_to_remove, value=target_label)

    def _labels_to_int_repr(self):
        for elem in self._raw:
            elem["label"] = elem["label"].apply(lambda x: self.label_mapping[x])

    def _get_split_indices(self, traj):
        train_size = int((1 - self._test_ratio) * len(traj))
        val_size = len(traj) - int((1 - self._val_ratio) * len(traj))

        indices = [x for x in range(len(traj))]

        indices_for_training = random.sample(indices, train_size)
        indices_for_validation = random.sample(indices_for_training, val_size)
        indices_for_training = set(indices_for_training) - set(indices_for_validation)
        indices_for_testing = set(indices) - indices_for_training
        indices_for_testing = list(indices_for_testing)

        return list(indices_for_training), list(indices_for_testing), list(indices_for_validation)

    def _set_splitted_data(self, traj, labels):

        i_train, i_test, i_val = self._get_split_indices(traj)

        random.shuffle(i_train)

        self.test_data = list(itemgetter(*i_test)(traj))
        self.val_data = list(itemgetter(*i_val)(traj))
        self.train_data = list(itemgetter(*i_train)(traj))
        self.test_labels = list(itemgetter(*i_test)(labels))
        self.val_labels = list(itemgetter(*i_val)(labels))
        self.train_labels = list(itemgetter(*i_train)(labels))

    def prepare_data(self):
        trajs = []
        labels = []

        self._remove_traj_containing_labels(self.labels_to_remove)
        self._merge_labels("car", "taxi")
        self._labels_to_int_repr()

        for elem in self._raw:
            data_ = elem[self.fields_to_feed].values.tolist()
            label_ = elem["label"].values.tolist()
            trajs.append(data_)
            labels.append(label_)
        self.prepared_data = trajs
        self.prepared_labels = labels # todo needed?

        self._set_splitted_data(self.prepared_data, self.prepared_labels)

    def batches(self):
        for i in range(0, len(self.train_data), self._batchsize):

            if len(self.train_data[i:i + self._batchsize]) < self._batchsize:
                break  # drop last incomplete batch

            labels_sorted = sorted(self.train_labels[i:i + self._batchsize:], key=len, reverse=True)
            train_sorted = sorted(self.train_data[i:i + self._batchsize:], key=len, reverse=True)
            for p in range(len(labels_sorted)):
                    assert len(labels_sorted[p]) == len(train_sorted[p])
            yield train_sorted, labels_sorted