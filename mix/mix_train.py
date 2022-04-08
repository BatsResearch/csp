#
import argparse
import os
import pickle
import pprint
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
from datasets.composition_dataset import (CompositionDataset, ImageLoader,
                                          transform_image)
from datasets.read_datasets import DATASET_PATHS
from models.compositional_modules import get_model
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from utils import set_seed

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class CustomCompositionDataset(CompositionDataset):
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            imagenet=False,
            attr_keep_ratio = 1.
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.attr_keep_ratio = attr_keep_ratio
        # self.removed_attr_list = removed_attr_list

        self.feat_dim = None
        self.transform = transform_image(phase, imagenet=imagenet)
        self.loader = ImageLoader(self.root + '/images/')

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        # remove data where labels contain the removed attributes;
        num_keep_attr = int(len(self.attrs) * self.attr_keep_ratio)
        self.subset_attrs = random.sample(self.attrs, num_keep_attr)
        self.subset_attr2idx = {attr: idx for idx, attr in enumerate(self.subset_attrs)}

        self.data = [(image, attr, obj) for image, attr, obj in self.data if attr in self.subset_attrs]
        self.subset_train_pairs = [(attr, obj) for attr, obj in self.train_pairs if attr in self.subset_attrs]
        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.subset_train_pairs)]
        )

        # select the indices that we are training
        self.attr_indices = [self.attr2idx[attr] for attr in self.subset_attrs]
        self.obj_indices = [len(self.attrs) + idx for idx in range(len(self.objs))]
        self.indices = self.attr_indices + self.obj_indices

        print('train data after removing attributes: ', len(self.data))
        print("total pairs in subset training", len(self.subset_train_pairs))



def train_model(model, optimizer, train_dataset, config, device):
    """Function to train the model to predict attributes with cross entropy loss.

    Args:
        model (nn.Module): the model to compute the similarity score with the images.
        optimizer (nn.optim): the optimizer with the learnable parameters.
        train_dataloader (DataLoader): the train data loader containing the train images and labels
        train_idx (list): list of train idx
        config (argparse.ArgumentParser): the config
        device (...): torch device

    Returns:
        tuple: the trained model (or the best model) and the optimizer
    """
    # TODO: train dataset
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )

    model.train()

    loss_fn = CrossEntropyLoss()
    #
    attr2idx = train_dataset.subset_attr2idx
    obj2idx = train_dataset.obj2idx

    #     train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj]) \
    #                                 for attr, obj in train_dataset.pairs]).to(device)
    # else:
    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj]) \
                                for attr, obj in train_dataset.subset_train_pairs]).to(device)
    i = 0
    best_model = None
    train_losses = []

    torch.autograd.set_detect_anomaly(True)

    for i in range(config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        for bid, batch in enumerate(train_dataloader):
            batch_img, batch_target = batch[0], batch[3]
            batch_target = batch_target.to(device)
            batch_img = batch_img.to(device)
            batch_feat = model.encode_image(batch_img)

            logits = model(batch_feat, train_pairs)

            loss = loss_fn(logits, batch_target)

            # normalize loss to account for batch accumulation
            loss = loss / config.gradient_accumulation_steps

            # backward pass
            loss.backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or \
                (bid + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item())
            progress_bar.set_postfix(
                {"train loss": np.mean(epoch_train_losses[-50:])}
            )

            progress_bar.update()

        progress_bar.close()
        progress_bar.write(
            f"epoch {i +1} train loss {np.mean(epoch_train_losses)}"
        )
        train_losses.append(np.mean(epoch_train_losses))

        if (i + 1) % config.save_every_n == 0:
            save_soft_embeddings_with_indices(model, config, train_dataset, epoch=i + 1)

    return model, optimizer


def save_soft_embeddings_with_indices(model, config, dataset, epoch=None):

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    # save the soft embedding
    with torch.no_grad():
        if epoch:
            soft_emb_path = os.path.join(
                config.save_path, f"soft_embeddings_epoch_{epoch}.pt"
            )
        else:
            soft_emb_path = os.path.join(
                config.save_path, "soft_embeddings.pt"
            )

        torch.save({"soft_embeddings": model.soft_embeddings, 'indices': dataset.indices}, soft_emb_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        help="name of the experiment",
        type=str,
        default="csp"
    )
    parser.add_argument("--dataset", help="name of the dataset", type=str)
    parser.add_argument(
        "--lr", help="learning rate", type=float, default=1e-04
    )
    parser.add_argument(
        "--weight_decay", help="weight decay", type=float, default=1e-05
    )
    parser.add_argument(
        "--clip_model", help="clip model type", type=str, default="ViT-B/32"
    )
    parser.add_argument(
        "--epochs", help="number of epochs", default=20, type=int
    )
    parser.add_argument(
        "--train_batch_size", help="train batch size", default=64, type=int
    )
    parser.add_argument(
        "--eval_batch_size", help="eval batch size", default=1024, type=int
    )
    parser.add_argument(
        "--evaluate_only",
        help="directly evaluate on the" "dataset without any training",
        action="store_true",
    )
    parser.add_argument(
        "--context_length",
        help="sets the context length of the clip model",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--attr_dropout",
        help="add dropout to attributes",
        type=float,
        default=0.0,
    )
    parser.add_argument("--save_path", help="save path", type=str)
    parser.add_argument(
        "--save_every_n",
        default=100,
        type=int,
        help="saves the model every n epochs; "
        "this is useful for validation/grid search",
    )
    parser.add_argument(
        "--save_model",
        help="indicate if you want to save the model state dict()",
        action="store_true",
    )
    parser.add_argument("--seed", help="seed value", default=0, type=int)
    parser.add_argument(
        '--attr_keep_ratio',
        help="amount of attribute to be included in the training",
        default=1.,
        type=float
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        help="number of gradient accumulation steps",
        default=1,
        type=int
    )

    config = parser.parse_args()

    # set the seed value
    set_seed(config.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pprint.pprint(config)
    print("training details: mix training")
    print("----")
    print(f"dataset: {config.dataset}")
    print(f"experiment name: {config.experiment_name}")
    print(f"lr: {config.lr}")
    print(f"epochs: {config.epochs}")

    if os.path.exists(config.save_path):
        print('file already exists')
        print('exiting!')
        exit(0)

    # This should work for mit-states, ut-zappos, and maybe c-gqa.
    dataset_path = DATASET_PATHS[config.dataset]
    train_dataset = CustomCompositionDataset(dataset_path,
                                      phase='train',
                                      split='compositional-split-natural',
                                      attr_keep_ratio=config.attr_keep_ratio)

    model, optimizer = get_model(train_dataset, config, device)

    print("model dtype", model.dtype)
    print("soft embedding dtype", model.soft_embeddings.dtype)


    if not config.evaluate_only:
        model, optimizer = train_model(
            model,
            optimizer,
            train_dataset,
            config,
            device,
        )

    save_soft_embeddings_with_indices(
        model,
        config,
        train_dataset
    )

    with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
        pickle.dump(config, fp)


    print("done!")
