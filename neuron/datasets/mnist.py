
from urllib import request
from typing import Any

import numpy as np
import hashlib
import os

class MNIST:

    def __init__(self, path: str) -> None:
        self.path = path + ".npz"
        self.url = r"https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

        self.__is_downloaded = False
        self.__md5_hash = "8a61469f7ea1b51cbae51d4f78837e45"

    def load(self) -> Any:

        if not self.__is_downloaded:
            self.download()

        with np.load(self.path) as f:
            x_train, y_train = f["x_train"], f["y_train"]
            x_test, y_test = f["x_test"], f["y_test"]

        return (x_train, y_train), (x_test, y_test)

    def download(self) -> None:

        if os.path.isfile(self.path):
            print(f"Found existing file {self.path}")
            return

        print("Downloading MNIST dataset ...")

        request.urlretrieve(self.url, self.path)

        print("Checking md5 hash ...")

        # https://stackoverflow.com/a/59056837
        with open(self.path, "rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)

        if not file_hash.hexdigest() == self.__md5_hash:
            print(f"Got wrong md5 hash, expected: {self.__md5_hash}, got: {file_hash.hexdigest()}")
            os.remove(self.path)

        self.__is_downloaded = True

        print(f"Successfully downloaded the mnist dataset, stored it in: {self.path}.")
