import contextlib
import inspect
from collections import defaultdict, deque
from copy import deepcopy

import numpy as np


class KFSaver:
    def __init__(
        self, kf, save_current=False, skip_private=False, skip_callable=False, ignore=()
    ):
        self._kf = kf
        self._DL = defaultdict(list)
        self._skip_private = skip_private
        self._skip_callable = skip_callable
        self._ignore = ignore
        self._len = 0

        properties = inspect.getmembers(type(kf), lambda o: isinstance(o, property))
        self.properties = [p for p in properties if p[0] not in ignore]

        if save_current:
            self.save()

    def save(self):
        kf = self._kf

        # force all attributes to be computed.
        for prop in self.properties:
            self._DL[prop[0]].append(getattr(kf, prop[0]))

        v = deepcopy(kf.__dict__)

        if self._skip_private:
            for key in list(v.keys()):
                if key.startswith("_"):
                    del v[key]

        if self._skip_callable:
            for key in list(v.keys()):
                if callable(v[key]):
                    del v[key]

        for ig in self._ignore:
            if ig in v:
                del v[ig]

        for key in list(v.keys()):
            self._DL[key].append(v[key])

        self.__dict__.update(self._DL)
        self._len += 1

    def __getitem__(self, key):
        return self._DL[key]

    def __setitem__(self, key, newvalue):
        self._DL[key] = newvalue
        self.__dict__.update(self._DL)

    def __len__(self):
        return self._len

    @property
    def keys(self):
        """list of all keys"""
        return list(self._DL.keys())

    def to_array(self, flatten=False):
        for key in self.keys:
            try:
                self.__dict__[key] = np.array(self._DL[key])
            except Exception as e:
                # get back to lists so we are in a valid state
                self.__dict__.update(self._DL)
                raise ValueError(f"could not convert {key} into np.array") from e
        if flatten:
            self.flatten()

    def flatten(self):
        for key in self.keys:
            with contextlib.suppress(Exception):
                arr = self.__dict__[key]
                shape = arr.shape
                if shape[2] == 1:
                    self.__dict__[key] = arr.reshape(shape[0], shape[1])
                arr = self.__dict__[key]
                shape = arr.shape
                if len(shape) == 2 and shape[1] == 1:
                    self.__dict__[key] = arr.ravel()

    def __repr__(self):
        return f'<KFSaver object at {hex(id(self))}\n  Keys: {" ".join(self.keys)}>'


def pretty_str(label, arr):
    def is_col(a):
        """return true if a is a column vector"""
        try:
            return a.shape[0] > 1 and a.shape[1] == 1
        except (AttributeError, IndexError):
            return False

    # display empty lists correctly
    with contextlib.suppress(TypeError):
        if len(arr) == 0:
            return f"{label} = {str(type(arr)())}"
    if isinstance(arr, (list, tuple, deque)):
        return "\n".join(
            [pretty_str(f"{label}[{str(i)}]", x) for (i, x) in enumerate(arr)]
        )

    if label is None:
        label = ""

    if label:
        label += " = "

    if is_col(arr):
        return label + str(arr.T).replace("\n", "") + ".T"

    rows = str(arr).split("\n")
    if not rows:
        return ""

    s = label + rows[0]
    pad = " " * len(label)
    for line in rows[1:]:
        s = s + "\n" + pad + line

    return s
