import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from gensim.models import KeyedVectors


class BasePretrainedVector(object):

    @classmethod
    def load(cls, token2id, test=False, limit=None):
        embed_shape = (len(token2id), 300)
        freqs = np.zeros((len(token2id)), dtype='f')

        if test:
            np.random.seed(0)
            vectors = np.random.normal(0, 1, embed_shape)
            vectors[0] = 0
            vectors[len(token2id) // 2:] = 0
        else:
            vectors = np.zeros(embed_shape, dtype='f')
            #path = f'{os.environ["DATADIR"]}/{cls.path}'
            path = "{}".format(cls.path)
            for i, o in enumerate(
                    open(path, encoding="utf8", errors='ignore')):
                token, *vector = o.split(' ')
                token = str.lower(token)
                if token not in token2id or len(o) <= 100:
                    continue
                if limit is not None and i > limit:
                    break
                freqs[token2id[token]] += 1
                vectors[token2id[token]] += np.array(vector, 'f')

        vectors[freqs != 0] /= freqs[freqs != 0][:, None]
        vec = KeyedVectors(300)
        vec.add(list(token2id.keys()), vectors, replace=True)

        return vec


def load_pretrained_vectors(names, token2id, test=False):
    assert isinstance(names, list)
    with Pool(processes=len(names)) as pool:
        f = partial(load_pretrained_vector, token2id=token2id, test=test)
        vectors = pool.map(f, names)
    return dict([(n, v) for n, v in zip(names, vectors)])

def load_pretrained_vector(name, token2id, test=False):
    loader = dict(
        glove=GlovePretrainedVector,
    )
    return loader[name].load(token2id, test)


class GlovePretrainedVector(BasePretrainedVector):
    name = 'glove.6B.300d'
    path = '/home/sunil/Downloads/glove.6B/glove.6B.300d.txt'