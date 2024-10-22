import torch
import random
from tqdm import tqdm

PAD, CLS = '[PAD]', '[CLS]'

def build_dataset(args, cover, stego):
    def load_dataset(paths, pad_size=32):
        """
        paths: ['cover.txt', 'stego.txt']
        """
        contents = []
        for path in paths:
            with open(path, 'r') as f:
                for line in tqdm(f):
                    lin = line.strip()
                    if 'cover' in path:
                        label = 0
                    else:
                        label = 1
                    token = args.tokenizer.tokenize(lin)
                    token = [CLS] + token
                    seq_len = len(token)
                    mask = []
                    token_ids = args.tokenizer.convert_tokens_to_ids(token)
                    if pad_size:
                        if len(token) < pad_size:
                            mask = [1] * len(token_ids) + \
                                   [0] * (pad_size - len(token))
                            token_ids += ([0] * (pad_size - len(token)))
                            #
                        else:
                            mask = [1] * pad_size
                            token_ids = token_ids[:pad_size]
                            seq_len = pad_size
                    contents.append((token_ids, label, seq_len, mask))
        # 对 contents 列表进行随机重排的操作
        random.shuffle(contents)
        return contents

    data = load_dataset([cover, stego])
    return data


class DatasetIterater(object):
    # 初始化
    # 类的初始化方法（构造函数），用于初始化对象的属性。self 代表对象本身
    def __init__(self, batches, args):
        self.batch_size = args.batch_size
        self.batches = batches
        self.n_batches = len(batches) // args.batch_size  # 向下取
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = args.device
    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size:len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif not self.residue and self.index == self.n_batches:
            self.index = 0
            batches = self.batches[
                      self.index * self.batch_size:(self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            batches = self.batches[
                      self.index * self.batch_size:(self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
            # raise StopIteration

        else:
            batches = self.batches[
                      self.index * self.batch_size:(self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, args):
    iters = DatasetIterater(dataset, args)
    return iters