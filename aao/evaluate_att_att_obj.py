# experiment to see if CSP can generalize vocabulary beyond the trained objective.

import argparse
import json
import os
import sys
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import torch

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load
from datasets.composition_dataset import ImageLoader, transform_image
from datasets.read_datasets import DATASET_PATHS
from models.coop import coop
from models.csp import CSPInterface, csp_init
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

## load the images with the attribute attribute object pairs
class AAODataset(Dataset):
    def __init__(self, root, images, labels, pairs) -> None:
        super().__init__()
        self.root = root
        self.images = images
        self.labels = labels
        self.user_att_label, self.att_label, self.obj_label = zip(*labels)
        self.pairs = pairs
        self.split = 'compositional-split-natural'
        self.pair_to_idx = dict([(pair, idx) for idx, pair in enumerate(pairs)])
        self.loader = ImageLoader(root + '/images/')
        self.transform = transform_image('test')

        self.attrs, self.objs, _, _, _, _ = self.parse_split()
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.loader(self.images[index])
        label = self.pair_to_idx[
            (self.user_att_label[index], self.att_label[index], self.obj_label[index])
        ]
        img = self.transform(img)
        return img, label

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            '%s/%s/val_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            '%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

## get the representations for both clip.
def clip_baseline(model, pairs, config, device):
    pairs = [(user_attr.replace(".", " ").lower(), \
              attr.replace(".", " ").lower(), \
              obj.replace(".", " ").lower())
             for user_attr, attr, obj in pairs]

    prompts = [f"a photo of {user_attr} {attr} {obj}" for user_attr, attr, obj in pairs]
    tokenized_prompts = clip.tokenize(prompts, context_length=config.context_length)
    test_batch_tokens = np.array_split(
        tokenized_prompts, len(tokenized_prompts) // config.text_encoder_batch_size
    )
    rep = torch.Tensor().to(device).type(model.dtype)
    with torch.no_grad():
        for batch_tokens in test_batch_tokens:
            batch_tokens = batch_tokens.to(device)
            _text_features = model.text_encoder(batch_tokens, enable_pos_emb=True)
            text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            rep = torch.cat((rep, text_features), dim=0)

    return rep

## csp needs a custom encoder because of the attribute-attribute-object instead of just attribute-object.
def compute_representations(model, test_dataset, config, device):
    """Function computes the attribute-object representations using the
    text encoder.
    Args:
        model (nn.Module): the model
        test_dataset (Dataset): CompositionDataset object with phase = 'test'
        config (argparse.ArgumentParser): config/args
        device (object): device
    Returns:
        torch.Tensor: returns the tensor with the class representations (N x D) where D=512;
    """
    obj2idx = test_dataset.obj2idx
    attr2idx = test_dataset.attr2idx
    print(test_dataset.pairs[:5])
    pairs = torch.tensor([(attr2idx[user_attr], attr2idx[attr], obj2idx[obj]) \
        for user_attr, attr, obj in test_dataset.pairs]).to(device)

    test_pairs = np.array_split(
        pairs, len(pairs) // config.text_encoder_batch_size
    )

    rep = torch.Tensor().to(device).type(model.dtype)
    with torch.no_grad():
        for batch_attr_obj in tqdm(test_pairs):
            batch_attr_obj = batch_attr_obj.to(device)

            # compute token tensors
            user_attr_idx = batch_attr_obj[:, 0]
            attr_idx = batch_attr_obj[:, 1]
            obj_idx = batch_attr_obj[:, 2]

            class_token_ids = model.token_ids.repeat(len(batch_attr_obj), 1)
            token_tensor = model.clip_model.token_embedding(
                class_token_ids.to(model.device)
            ).type(model.clip_model.dtype)

            eos_idx = int(model.token_ids[0].argmax())
            soft_embeddings = model.attr_dropout(model.soft_embeddings)
            token_tensor[:, eos_idx - 3, :] = soft_embeddings[
                user_attr_idx
            ].type(model.clip_model.dtype)
            token_tensor[:, eos_idx - 2, :] = soft_embeddings[
                attr_idx
            ].type(model.clip_model.dtype)
            token_tensor[:, eos_idx - 1, :] = soft_embeddings[
                obj_idx + model.offset
            ].type(model.clip_model.dtype)
            # end compute token tensor

            text_features = model.text_encoder(
                model.token_ids,
                token_tensor,
                enable_pos_emb=model.enable_pos_emb,
            )

            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

            rep = torch.cat([rep, text_features], dim=0)

    return rep


def compute_coop_representations(model, test_dataset, config, device):
    """Function computes the attribute-object representations using the
    text encoder.
    Args:
        model (nn.Module): the model
        test_dataset (Dataset): CompositionDataset object with phase = 'test'
        config (argparse.ArgumentParser): config/args
        device (object): device
    Returns:
        torch.Tensor: returns the tensor with the class representations (N x D) where D=512;
    """
    obj2idx = test_dataset.obj2idx
    attr2idx = test_dataset.attr2idx
    print(test_dataset.pairs[:5])
    pairs = torch.tensor([(attr2idx[user_attr], attr2idx[attr], obj2idx[obj]) \
        for user_attr, attr, obj in test_dataset.pairs]).to(device)

    test_pairs = np.array_split(
        pairs, len(pairs) // config.text_encoder_batch_size
    )

    rep = torch.Tensor().to(device).type(model.dtype)
    with torch.no_grad():
        for batch_attr_obj in tqdm(test_pairs):
            batch_attr_obj = batch_attr_obj.to(device)

            # compute token tensors
            user_attr_idx = batch_attr_obj[:, 0]
            attr_idx = batch_attr_obj[:, 1]
            obj_idx = batch_attr_obj[:, 2]

            class_token_ids = model.token_ids.repeat(len(batch_attr_obj), 1)
            token_tensor = model.clip_model.token_embedding(
                class_token_ids.to(model.device)
            ).type(model.clip_model.dtype)

            eos_idx = int(model.token_ids[0].argmax())
            token_tensor[:, eos_idx - 3, :] = model.frozen_embeddings[
                user_attr_idx
            ].type(model.clip_model.dtype)
            token_tensor[:, eos_idx - 2, :] = model.frozen_embeddings[
                attr_idx
            ].type(model.clip_model.dtype)
            token_tensor[:, eos_idx - 1, :] = model.frozen_embeddings[
                obj_idx + model.offset
            ].type(model.clip_model.dtype)
            # end compute token tensor

            # adding the correct learnable context
            token_tensor[
                :, 1 : len(model.soft_embeddings) + 1, :
            ] = model.soft_embeddings.type(model.clip_model.dtype)


            text_features = model.text_encoder(
                model.token_ids,
                token_tensor,
                enable_pos_emb=model.enable_pos_emb,
            )

            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

            rep = torch.cat([rep, text_features], dim=0)

    return rep



## compute the logits
def test(
    model,
    test_dataset,
    testloader,
    config,
    device="cuda:0",
):
    model.to(device)
    model.eval()

    # hot fix
    if config.experiment_name == 'clip':
        text_rep = clip_baseline(model, test_dataset.pairs, config, device)
    elif config.experiment_name == 'czsl':
        text_rep = compute_representations(model, test_dataset, config, device)
    elif config.experiment_name == 'coop':
        text_rep = compute_coop_representations(model, test_dataset, config, device)

    all_labels = torch.Tensor().to(device)
    all_logits = torch.Tensor().to(device)
    with torch.no_grad():
        for idx, data in tqdm(
            enumerate(testloader), total=len(testloader), desc="Testing"
        ):
            batch_img = data[0].to(device)
            batch_img_feat = model.encode_image(batch_img)
            normalized_img = batch_img_feat / batch_img_feat.norm(
                dim=-1, keepdim=True
            )

            logits = (
                model.clip_model.logit_scale.exp()
                * normalized_img
                @ text_rep.t()
            )

            all_labels = torch.cat([all_labels, data[1].to(device)], dim=0)
            all_logits = torch.cat([all_logits, logits], dim=0)

        acc = torch.sum(all_labels == torch.argmax(all_logits, dim=1)) / len(all_labels)
        return acc

def get_custom_czsl(train_dataset, config, device, prompt_template="a photo of X X X",):
    (
        clip_model,
        soft_embedding,
        class_token_ids,
        offset
    ) = csp_init(train_dataset, config, device, prompt_template=prompt_template)

    interface = CSPInterface(
        clip_model,
        config,
        offset,
        soft_embedding,
        class_token_ids,
        device,
        attr_dropout=config.attr_dropout
    )

    return interface

def get_custom_coop(train_dataset, config, device, prompt_template="a photo of X X X",):

    config.lr = 5e-05
    config.weight_decay = 5e-05

    model, optimizer = coop(train_dataset, config, device, prompt_template=prompt_template)

    return model


## main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_model", help="clip model type", type=str, default="ViT-B/32"
    )
    parser.add_argument(
        "--eval_batch_size", help="eval batch size", default=256, type=int
    )
    parser.add_argument(
        "--experiment_name",
        help="name of the experiment",
        type=str,
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
    parser.add_argument(
        "--soft_embeddings",
        help="location for softembeddings",
        type=str,
        default="./soft_embeddings.pt",
    )
    parser.add_argument(
        "--text_encoder_batch_size",
        help="batch size of the text encoder",
        default=16,
        type=int,
    )

    config = parser.parse_args()
    # this is the annotated data
    test_data = pd.read_csv(os.path.join(DIR_PATH, 'datasets/aao_mit_states/193-annot.csv'))

    input_images = test_data['Input.image_url'].tolist()
    input_user_attr = test_data['Answer.ao-image-attr.label'].tolist()
    input_attr = test_data['Input.attr'].tolist()
    input_obj = test_data['Input.obj'].tolist()

    label_pairs = sorted(list(set([(att, user_att, obj)
                            for user_att, att, obj in \
                                zip(input_user_attr, input_attr, input_obj)])))

    test_dataset = AAODataset(DATASET_PATHS['mit-states'],
                              input_images,
                              zip(input_attr, input_user_attr, input_obj),
                              label_pairs)

    test_dataloader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if config.experiment_name == 'clip':
        clip_model, preprocess = load(
            config.clip_model, device=device, context_length=config.context_length
        )
        model = CLIPInterface(clip_model, config, token_ids=None, device=device, enable_pos_emb=True)

    elif config.experiment_name == 'czsl':
        model = get_custom_czsl(test_dataset, config, device)
        soft_embs = torch.load(config.soft_embeddings)['soft_embeddings']
        model.set_soft_embeddings(soft_embs)

    elif config.experiment_name == 'coop':
        model = get_custom_coop(test_dataset, config, device)
        soft_embs = torch.load(config.soft_embeddings, map_location='cpu')['soft_embeddings']
        model.set_soft_embeddings(soft_embs)

    acc = test(model, test_dataset, test_dataloader, config, device)

    print('method: ', config.experiment_name)
    print('unseen accuracy: ', acc)

    if config.experiment_name != 'clip':
        result_path = config.soft_embeddings[:-2] + "aao.json"

        with open(result_path, 'w+') as fp:
            json.dump({'acc': acc.item()}, fp)

    return acc

if __name__ == "__main__":
    main()
