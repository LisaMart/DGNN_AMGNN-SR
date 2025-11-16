import numpy as np

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def data_masks(all_usr_pois, item_tail, len_max):
    us_lens = [len(upois) for upois in all_usr_pois]
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


class Data:
    def __init__(self, data, len_max, shuffle=False):
        inputs_raw = data[0]
        targets = data[1]
        inputs, mask, len_max = data_masks(inputs_raw, [0], len_max)
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.targets = np.asarray(targets)
        self.length = len(inputs)
        self.shuffle = shuffle
        self.batch_size = 100  # default, можно переопределить

    def generate_batch(self, batch_size, seed=None):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.seed(seed)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]

        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        return self.inputs[i], self.mask[i], self.targets[i]