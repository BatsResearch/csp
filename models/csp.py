import os

import clip
import pandas as pd
import torch
import torch.nn as nn
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class CSPInterface(CLIPInterface):
    def __init__(
        self,
        clip_model,
        config,
        offset,
        soft_embeddings,
        class_token_ids,
        device="cuda:0",
        enable_pos_emb=True,
        attr_dropout=0.0,
    ):
        super().__init__(
            clip_model,
            config,
            class_token_ids,
            soft_embeddings,
            device=device,
            enable_pos_emb=enable_pos_emb,
        )

        self.offset = offset
        self.attr_dropout = nn.Dropout(attr_dropout)

    def construct_token_tensors(self, pair_idx):
        """Function creates the token tensor for further inference.

        Args:
            pair_idx (torch.Tensor): Shape [N x 2], where N is the number
                of pairs of attr and obj

        Returns:
            torch.Tensor: token tensor passed to the text encoder;
                shape [N x context_length x 512]
        """
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.clip_model.token_embedding(
            class_token_ids.to(self.device)
        ).type(self.clip_model.dtype)

        eos_idx = int(self.token_ids[0].argmax())
        soft_embeddings = self.attr_dropout(self.soft_embeddings)
        token_tensor[:, eos_idx - 2, :] = soft_embeddings[
            attr_idx
        ].type(self.clip_model.dtype)
        token_tensor[:, eos_idx - 1, :] = soft_embeddings[
            obj_idx + self.offset
        ].type(self.clip_model.dtype)

        return token_tensor


def csp_init(
    train_dataset,
    config,
    device,
    prompt_template="a photo of X X",
):

    clip_model, preprocess = load(
        config.clip_model, device=device, context_length=config.context_length
    )

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs

    # cleaning the classes and the attributes
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]

    tokenized = torch.cat(
        [
            clip.tokenize(tok, context_length=config.context_length)
            for tok in attributes + classes
        ]
    )

    orig_token_embedding = clip_model.token_embedding(tokenized.to(device))

    soft_embedding = torch.zeros(
        (len(attributes) + len(classes), orig_token_embedding.size(-1)),
    )
    for idx, rep in enumerate(orig_token_embedding):
        eos_idx = tokenized[idx].argmax()
        soft_embedding[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

    soft_embedding = nn.Parameter(soft_embedding)

    class_token_ids = clip.tokenize(
        [prompt_template],
        context_length=config.context_length,
    )
    offset = len(attributes)

    return (
        clip_model,
        soft_embedding,
        class_token_ids,
        offset
    )



def get_csp(train_dataset, config, device):

    (
        clip_model,
        soft_embedding,
        class_token_ids,
        offset
    ) = csp_init(train_dataset, config, device)

    optimizer = torch.optim.Adam(
        [soft_embedding],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    interface = CSPInterface(
        clip_model,
        config,
        offset,
        soft_embedding,
        class_token_ids,
        device,
        attr_dropout=config.attr_dropout
    )

    return interface, optimizer


def get_mix_csp(train_dataset, config, device):

    (
        clip_model,
        soft_embedding,
        class_token_ids,
        offset
    ) = csp_init(train_dataset, config, device)

    with torch.no_grad():
        subset_soft_embeddings = soft_embedding[train_dataset.indices, :]

    subset_soft_embeddings.requires_grad = True

    optimizer = torch.optim.Adam(
        [subset_soft_embeddings],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # reduce the offset to selected offset
    offset = len(train_dataset.attr_indices)
    interface = CSPInterface(
        clip_model,
        config,
        offset,
        subset_soft_embeddings,
        class_token_ids,
        device,
        attr_dropout=config.attr_dropout
    )

    return interface, optimizer
