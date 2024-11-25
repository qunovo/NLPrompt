import os
import pickle
import random
# from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing, listdir_nohidden

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD


@DATASET_REGISTRY.register()
class Food101N(DatasetBase):
    dataset_dir = "food-101n"
    test_dataset_dir = 'food-101'
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.test_dataset_dir = os.path.join(root, self.test_dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.test_image_dir = os.path.join(self.test_dataset_dir, "images")
        self.split_path = os.path.join(self.test_dataset_dir, "split_zhou_Food101.json")
        self.split_fewshot_dir = os.path.join(self.test_dataset_dir, "split_fewshot")
        # self.baseline_dir = os.path.join(self.dataset_dir, "baseline")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            _, _, test = OxfordPets.read_split(self.split_path, self.test_image_dir)
        else:
            total_train, val, test = DTD.read_and_split_data(self.test_image_dir)
            OxfordPets.save_split(total_train, val, test, self.split_path, self.test_image_dir)

        total_train, val = self.read_and_split_data(self.image_dir) # load dataset and split data

        num_shots = cfg.DATASET.NUM_SHOTS
        backbone = cfg.MODEL.HEAD.NAME
        if num_shots >= 1:
            seed = cfg.SEED
            if cfg.TRAINER.NAME == "Baseline":
                preprocessed = os.path.join(self.baseline_dir, backbone, f"shot_{num_shots}-seed_{seed}.pkl")
            else:
                preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(total_train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                # with open(preprocessed, "wb") as file:
                #     pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_and_split_data(image_dir, p_trn=1.0, ignored=[], new_cnames=None):
        # The data are supposed to be organized into the following structure
        # =============
        # images/
        #     dog/
        #     cat/
        #     horse/
        # =============

        # convert all the class under image dir to a list
        categories = listdir_nohidden(image_dir)
        categories = [c for c in categories if c not in ignored]
        categories.sort()

        p_val = 1 - p_trn
        print(f"Splitting into {p_trn:.0%} train, {p_val:.0%} val")

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(impath=im, label=y, classname=c)  # is already 0-based
                items.append(item)
            return items

        train, val, test = [], [], []
        for label, category in enumerate(categories):
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = n_total - n_train
            assert n_train > 0 and n_val >= 0

            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]

            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train:], label, category))

        return train, val
