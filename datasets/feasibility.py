import argparse
import os
from itertools import product

import clip
import torch
import torch.nn.functional as F
from clip_modules.model_loader import load

from datasets.composition_dataset import CompositionDataset
from datasets.read_datasets import DATASET_PATHS

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def compute_cosine_similarity(names, weights, return_dict=True):
    pairing_names = list(product(names, names))
    normed_weights = F.normalize(weights,dim=1)
    similarity = torch.mm(normed_weights, normed_weights.t())
    if return_dict:
        dict_sim = {}
        for i,n in enumerate(names):
            for j,m in enumerate(names):
                dict_sim[(n,m)]=similarity[i,j].item()
        return dict_sim
    return pairing_names, similarity.to(device)


def load_glove_embeddings(vocab):
    '''
    Inputs
        emb_file: Text file with word embedding pairs e.g. Glove, Processed in lower case.
        vocab: List of words
    Returns
        Embedding Matrix
    '''
    vocab = [v.lower() for v in vocab]
    emb_file = os.path.join(DIR_PATH, 'data/glove.6B.300d.txt')
    model = {}  # populating a dictionary of word and embeddings
    for line in open(emb_file, 'r'):
        line = line.strip().split(' ')  # Word-embedding
        wvec = torch.FloatTensor(list(map(float, line[1:])))
        model[line[0]] = wvec

    # Adding some vectors for UT Zappos
    custom_map = {
        'faux.fur': 'fake_fur',
        'faux.leather': 'fake_leather',
        'full.grain.leather': 'thick_leather',
        'hair.calf': 'hair_leather',
        'patent.leather': 'shiny_leather',
        'boots.ankle': 'ankle_boots',
        'boots.knee.high': 'knee_high_boots',
        'boots.mid-calf': 'midcalf_boots',
        'shoes.boat.shoes': 'boat_shoes',
        'shoes.clogs.and.mules': 'clogs_shoes',
        'shoes.flats': 'flats_shoes',
        'shoes.heels': 'heels',
        'shoes.loafers': 'loafers',
        'shoes.oxfords': 'oxford_shoes',
        'shoes.sneakers.and.athletic.shoes': 'sneakers',
        'traffic_light': 'traffic_light',
        'trash_can': 'trashcan',
        'dry-erase_board' : 'dry_erase_board',
        'black_and_white' : 'black_white',
        'eiffel_tower' : 'tower',
        'nubuck' : 'grainy_leather',
    }

    embeds = []
    for k in vocab:
        if k in custom_map:
            k = custom_map[k]
        if '_' in k:
            ks = k.split('_')
            emb = torch.stack([model[it] for it in ks]).mean(dim=0)
        else:
            emb = model[k]
        embeds.append(emb)
    embeds = torch.stack(embeds)
    print('Glove Embeddings loaded, total embeddings: {}'.format(embeds.size()))
    return embeds


def clip_embeddings(model, words_list):
    words_list = [word.replace(".", " ").lower() for word in words_list]
    prompts = [f"a photo of {word}" for word in words_list]

    tokenized_prompts = clip.tokenize(prompts)
    with torch.no_grad():
        _text_features = model.text_encoder(tokenized_prompts, enable_pos_emb=True)
        text_features = _text_features / _text_features.norm(
            dim=-1, keepdim=True
        )
        return text_features

def get_pair_scores_objs(attr, obj, all_objs, attrs_by_obj_train, obj_embedding_sim):
    score = -1.
    for o in all_objs:
        if o!=obj and attr in attrs_by_obj_train[o]:
            temp_score = obj_embedding_sim[(obj,o)]
            if temp_score>score:
                score=temp_score
    return score

def get_pair_scores_attrs(attr, obj, all_attrs, obj_by_attrs_train, attr_embedding_sim):
    score = -1.
    for a in all_attrs:
        if a != attr and obj in obj_by_attrs_train[a]:
            temp_score = attr_embedding_sim[(attr, a)]
            if temp_score > score:
                score = temp_score
    return score

def compute_feasibility(dataset):
    objs = dataset.objs
    attrs = dataset.attrs

    print('computing the obj embeddings')
    obj_embeddings = load_glove_embeddings(objs).to(device)
    obj_embedding_sim = compute_cosine_similarity(objs, obj_embeddings,
                                                        return_dict=True)

    print('computing the attr embeddings')
    attr_embeddings = load_glove_embeddings(attrs).to(device)
    attr_embedding_sim = compute_cosine_similarity(attrs, attr_embeddings,
                                                        return_dict=True)

    print('computing the feasibilty score')
    feasibility_scores = dataset.seen_mask.clone().float()
    for a in attrs:
        print('Attribute', a)
        for o in objs:
            if (a, o) not in dataset.train_pairs:
                idx = dataset.pair2idx[(a, o)]
                score_obj = get_pair_scores_objs(
                    a,
                    o,
                    dataset.objs,
                    dataset.attrs_by_obj_train,
                    obj_embedding_sim
                )
                score_attr = get_pair_scores_attrs(
                    a,
                    o,
                    dataset.attrs,
                    dataset.obj_by_attrs_train,
                    attr_embedding_sim
                )
                score = (score_obj + score_attr) / 2
                feasibility_scores[idx] = score

    # feasibility_scores = feasibility_scores

    return feasibility_scores * (1 - dataset.seen_mask.float())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="name of the dataset", type=str)
    parser.add_argument(
        "--clip_model", help="clip model type", type=str, default="ViT-B/32"
    )
    config = parser.parse_args()

    dataset_path = DATASET_PATHS[config.dataset]
    dataset =  CompositionDataset(dataset_path,
                                    phase='test',
                                    split='compositional-split-natural',
                                    open_world=True)

    feasibility = compute_feasibility(dataset)

    save_path = os.path.join(DIR_PATH, f'data/feasibility_{config.dataset}.pt')
    torch.save({
        'feasibility': feasibility,
    }, save_path)

    print('done!')
