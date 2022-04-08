
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
from clip_modules.interface import CLIPInterface
from clip_modules.model_loader import load
from datasets.composition_dataset import CompositionDataset
from datasets.read_datasets import DATASET_PATHS
from evaluate import Evaluator, clip_baseline, compute_representations
from models.compositional_modules import get_model
from scipy.stats import hmean
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

class MixTrainEvaluator(Evaluator):
    """
    Evaluator class, adapted from:
    https://github.com/Tushar-N/attributes-as-operators

    With modifications from:
    https://github.com/ExplainableML/czsl
    """
    def __init__(self, dset, model, seen_pairs):

        self.dset = dset
        self.subset_seen_pairs = seen_pairs

        # Convert text pairs to idx tensors: [('sliced', 'apple'), ('ripe', 'apple'), ...] --> torch.LongTensor([[0,1],[1,1], ...])
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.pairs]
        self.train_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.train_pairs]
        self.pairs = torch.LongTensor(pairs)

        # Mask over pairs that occur in closed world
        # Select set based on phase
        if dset.phase == 'train':
            print('Evaluating with train pairs')
            test_pair_set = set(dset.train_pairs)
            test_pair_gt = set(dset.train_pairs)
        elif dset.phase == 'val':
            print('Evaluating with validation pairs')
            test_pair_set = set(dset.val_pairs + dset.train_pairs)
            test_pair_gt = set(dset.val_pairs)
        else:
            print('Evaluating with test pairs')
            test_pair_set = set(dset.test_pairs + dset.train_pairs)
            test_pair_gt = set(dset.test_pairs)

        self.test_pair_dict = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in test_pair_gt]
        self.test_pair_dict = dict.fromkeys(self.test_pair_dict, 0)

        # dict values are pair val, score, total
        for attr, obj in test_pair_gt:
            pair_val = dset.pair2idx[(attr, obj)]
            key = (dset.attr2idx[attr], dset.obj2idx[obj])
            self.test_pair_dict[key] = [pair_val, 0, 0]

        # open world
        if dset.open_world:
            masks = [1 for _ in dset.pairs]
        else:
            masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        # masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        self.closed_mask = torch.BoolTensor(masks)
        # Mask of seen concepts
        # changed (new)
        # seen_pair_set = set(dset.train_pairs)
        seen_pair_set = set(self.subset_seen_pairs)
        mask = [1 if pair in seen_pair_set else 0 for pair in pairs]
        self.seen_mask = torch.BoolTensor(mask)

        # Object specific mask over which pairs occur in the object oracle setting
        oracle_obj_mask = []
        for _obj in dset.objs:
            mask = [1 if _obj == obj else 0 for attr, obj in dset.pairs]
            oracle_obj_mask.append(torch.BoolTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

        # Decide if the model under evaluation is a manifold model or not
        self.score_model = self.score_manifold_model

    # Generate mask for each settings, mask scores, and get prediction labels
    def generate_predictions(self, scores, obj_truth, bias=0.0, topk=1):  # (Batch, #pairs)
        '''
        Inputs
            scores: Output scores
            obj_truth: Ground truth object
        Returns
            results: dict of results in 3 settings
        '''

        def get_pred_from_scores(_scores, topk):
            """
            Given list of scores, returns top 10 attr and obj predictions
            Check later
            """
            _, pair_pred = _scores.topk(topk, dim=1)  # sort returns indices of k largest values
            pair_pred = pair_pred.contiguous().view(-1)
            attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(
                -1, topk
            ), self.pairs[pair_pred][:, 1].view(-1, topk)
            return (attr_pred, obj_pred)

        results = {}
        orig_scores = scores.clone()
        mask = self.seen_mask.repeat(
            scores.shape[0], 1
        )  # Repeat mask along pairs dimension
        scores[~mask] += bias  # Add bias to test pairs

        # Unbiased setting

        # Open world setting --no mask, all pairs of the dataset
        results.update({"open": get_pred_from_scores(scores, topk)})
        results.update(
            {"unbiased_open": get_pred_from_scores(orig_scores, topk)}
        )
        # Closed world setting - set the score for all Non test pairs to -1e10,
        # this excludes the pairs from set not in evaluation
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10
        closed_orig_scores = orig_scores.clone()
        closed_orig_scores[~mask] = -1e10
        results.update({"closed": get_pred_from_scores(closed_scores, topk)})
        results.update(
            {"unbiased_closed": get_pred_from_scores(closed_orig_scores, topk)}
        )

        return results

    def score_clf_model(self, scores, obj_truth, topk=1):
        '''
        Wrapper function to call generate_predictions for CLF models
        '''
        attr_pred, obj_pred = scores

        # Go to CPU
        attr_pred, obj_pred, obj_truth = attr_pred.to('cpu'), obj_pred.to('cpu'), obj_truth.to('cpu')

        # Gather scores (P(a), P(o)) for all relevant (a,o) pairs
        # Multiply P(a) * P(o) to get P(pair)
        attr_subset = attr_pred.index_select(1, self.pairs[:, 0])  # Return only attributes that are in our pairs
        obj_subset = obj_pred.index_select(1, self.pairs[:, 1])
        scores = (attr_subset * obj_subset)  # (Batch, #pairs)

        results = self.generate_predictions(scores, obj_truth)
        results['biased_scores'] = scores

        return results

    def score_manifold_model(self, scores, obj_truth, bias=0.0, topk=1):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''
        # Go to CPU
        scores = {k: v.to('cpu') for k, v in scores.items()}
        obj_truth = obj_truth.to(device)

        # Gather scores for all relevant (a,o) pairs
        scores = torch.stack(
            [scores[(attr, obj)] for attr, obj in self.dset.pairs], 1
        )  # (Batch, #pairs)
        orig_scores = scores.clone()
        results = self.generate_predictions(scores, obj_truth, bias, topk)
        results['scores'] = orig_scores
        return results

    def score_fast_model(self, scores, obj_truth, bias=0.0, topk=1, pair_truth=None):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''

        results = {}
        mask = self.seen_mask.repeat(scores.shape[0], 1)  # Repeat mask along pairs dimension
        scores[~mask] += bias  # Add bias to test pairs

        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10

        _, pair_pred = closed_scores.topk(topk, dim=1)  # sort returns indices of k largest values
        # _, pair_pred = scores.topk(topk, dim=1)  # sort returns indices of k largest values
        pair_pred = pair_pred.contiguous().view(-1)
        attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
                              self.pairs[pair_pred][:, 1].view(-1, topk)

        results.update({'closed': (attr_pred, obj_pred)})
        return results
    def evaluate_predictions(self, predictions, attr_truth, obj_truth, pair_truth, allpred, topk=1):
        # Go to CPU
        attr_truth, obj_truth, pair_truth = (
            attr_truth.to("cpu"),
            obj_truth.to("cpu"),
            pair_truth.to("cpu"),
        )

        pairs = list(zip(list(attr_truth.numpy()), list(obj_truth.numpy())))

        seen_ind, unseen_ind = [], []
        for i in range(len(attr_truth)):
            # if pairs[i] in self.train_pairs:
            if pairs[i] in self.subset_seen_pairs:
                seen_ind.append(i)
            else:
                unseen_ind.append(i)

        seen_ind, unseen_ind = torch.LongTensor(seen_ind), torch.LongTensor(
            unseen_ind
        )

        def _process(_scores):
            # Top k pair accuracy
            # Attribute, object and pair
            attr_match = (
                attr_truth.unsqueeze(1).repeat(1, topk) == _scores[0][:, :topk]
            )
            obj_match = (
                obj_truth.unsqueeze(1).repeat(1, topk) == _scores[1][:, :topk]
            )

            # Match of object pair
            match = (attr_match * obj_match).any(1).float()
            attr_match = attr_match.any(1).float()
            obj_match = obj_match.any(1).float()
            # Match of seen and unseen pairs
            seen_match = match[seen_ind]
            unseen_match = match[unseen_ind]
            ### Calculating class average accuracy

            seen_score, unseen_score = torch.ones(512, 5), torch.ones(512, 5)

            return attr_match, obj_match, match, seen_match, unseen_match, \
                   torch.Tensor(seen_score + unseen_score), torch.Tensor(seen_score), torch.Tensor(unseen_score),

        def _add_to_dict(_scores, type_name, stats):
            base = [
                "_attr_match",
                "_obj_match",
                "_match",
                "_seen_match",
                "_unseen_match",
                "_ca",
                "_seen_ca",
                "_unseen_ca",
            ]
            for val, name in zip(_scores, base):
                stats[type_name + name] = val

        stats = dict()

        #################### Closed world
        closed_scores = _process(predictions["closed"])
        unbiased_closed = _process(predictions["unbiased_closed"])
        _add_to_dict(closed_scores, "closed", stats)
        _add_to_dict(unbiased_closed, "closed_ub", stats)

        #################### Calculating AUC
        scores = predictions["scores"]
        # getting score for each ground truth class
        correct_scores = scores[torch.arange(scores.shape[0]), pair_truth][
            unseen_ind
        ]

        # Getting top predicted score for these unseen classes
        max_seen_scores = predictions['scores'][unseen_ind][:, self.seen_mask].topk(topk, dim=1)[0][:, topk - 1]

        # Getting difference between these scores
        unseen_score_diff = max_seen_scores - correct_scores

        # Getting matched classes at max bias for diff
        unseen_matches = stats["closed_unseen_match"].bool()
        correct_unseen_score_diff = unseen_score_diff[unseen_matches] - 1e-4

        # sorting these diffs
        correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
        magic_binsize = 20
        # getting step size for these bias values
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        # Getting list
        biaslist = correct_unseen_score_diff[::bias_skip]

        seen_match_max = float(stats["closed_seen_match"].mean())
        unseen_match_max = float(stats["closed_unseen_match"].mean())
        seen_accuracy, unseen_accuracy = [], []
        match_list = []

        # Go to CPU
        base_scores = {k: v.to("cpu") for k, v in allpred.items()}
        obj_truth = obj_truth.to("cpu")

        # Gather scores for all relevant (a,o) pairs
        base_scores = torch.stack(
            [allpred[(attr, obj)] for attr, obj in self.dset.pairs], 1
        )  # (Batch, #pairs)

        for bias in biaslist:
            scores = base_scores.clone()
            results = self.score_fast_model(scores, obj_truth, bias=bias, topk=topk)
            results = results['closed']  # we only need biased
            results = _process(results)
            match = results[2]
            seen_match = float(results[3].mean())
            unseen_match = float(results[4].mean())
            seen_accuracy.append(seen_match)
            unseen_accuracy.append(unseen_match)
            match_list.append(match)

        seen_accuracy.append(seen_match_max)
        unseen_accuracy.append(unseen_match_max)
        seen_accuracy, unseen_accuracy = np.array(seen_accuracy), np.array(
            unseen_accuracy
        )
        area = np.trapz(seen_accuracy, unseen_accuracy)

        for key in stats:
            stats[key] = float(stats[key].mean())

        try:
            harmonic_mean = hmean([seen_accuracy, unseen_accuracy], axis=0)
        except:
            harmonic_mean = 0

        max_hm = np.max(harmonic_mean)
        idx = np.argmax(harmonic_mean)
        if idx == len(biaslist):
            bias_term = 1e3
        else:
            bias_term = biaslist[idx]
        stats["biasterm"] = float(bias_term)
        stats["best_unseen"] = np.max(unseen_accuracy)
        stats["best_seen"] = np.max(seen_accuracy)
        stats['match_list'] = match_list[idx].tolist()
        stats["AUC"] = area
        stats["hm_unseen"] = unseen_accuracy[idx]
        stats["hm_seen"] = seen_accuracy[idx]
        stats["best_hm"] = max_hm
        return stats


def test(
    model,
    test_dataset,
    testloader,
    evaluator,
    config,
    device="cuda:0",
):

    model.eval()
    accuracies, all_sub_gt, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # hot fix
    if config.experiment_name == 'clip':
    # text_rep = compute_representations(model, test_index, config, device)
        text_rep = clip_baseline(model, test_dataset, config, device)
    else:
        text_rep = compute_representations(model, test_dataset, config, device)
    with torch.no_grad():
        for idx, data in tqdm(
            enumerate(testloader), total=len(testloader), desc="Testing"
        ):
            batch_img = data[0].to(device)
            batch_img_feat = (
                batch_img.to(torch.float16)
                if config.use_features
                else model.encode_image(batch_img)
            )
            normalized_img = batch_img_feat / batch_img_feat.norm(
                dim=-1, keepdim=True
            )

            logits = (
                model.clip_model.logit_scale.exp()
                * normalized_img
                @ text_rep.t()
            )

            attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
            # Translate Batch Logits to DICT (PAIR: LOGITS)
            # logits = logits if config.use_features else logits.detach().cpu()
            logits = logits.cpu()
            predictions = {
                pair_name: logits[:, i]
                for i, pair_name in enumerate(test_dataset.pairs)
            }

            all_pred.append(predictions)
            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)

    all_attr_gt, all_obj_gt, all_pair_gt = (
        torch.cat(all_attr_gt).to("cpu"),
        torch.cat(all_obj_gt).to("cpu"),
        torch.cat(all_pair_gt).to("cpu"),
    )

    all_pred_dict = {}
    # Gather values as dict of (attr, obj) as key and list of predictions as values

    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k] for i in range(len(all_pred))]
        ).float()

    results = evaluator.score_model(
        all_pred_dict, all_obj_gt, bias=config.bias, topk=config.topk
    )

    attr_acc = float(torch.mean((results['unbiased_closed'][0].squeeze(-1) == all_attr_gt).float()))
    obj_acc = float(torch.mean((results['unbiased_closed'][1].squeeze(-1) == all_obj_gt).float()))

    stats = evaluator.evaluate_predictions(
        results,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        all_pred_dict,
        topk=config.topk,
    )

    stats['attr_acc'] = attr_acc
    stats['obj_acc'] = obj_acc

    return results, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--eval_batch_size", help="eval batch size", default=64, type=int
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
        "--open_world",
        help="evaluate on open world setup",
        action="store_true",
    )
    parser.add_argument(
        "--bias",
        help="eval bias",
        type=float,
        default=1e3,
    )
    parser.add_argument(
        "--topk",
        help="eval topk",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--soft_embeddings",
        help="location for softembeddings",
        type=str,
        default="./soft_embeddings.pt",
    )

    config = parser.parse_args()

    # set the seed value

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("evaluation details")
    print("----")
    print(f"dataset: {config.dataset}")
    print(f"experiment name: {config.experiment_name}")

    #
    if config.experiment_name != 'clip':
        if not os.path.exists(config.soft_embeddings):
            print(f'{config.soft_embeddings} not found')
            print('code exiting!')
            exit(0)

    dataset_path = DATASET_PATHS[config.dataset]

    val_dataset =  CompositionDataset(dataset_path,
                                      phase='val',
                                      split='compositional-split-natural',
                                      open_world=config.open_world)

    soft_embeddings = torch.load(config.soft_embeddings)
    if 'indices' in soft_embeddings:
        indices = soft_embeddings['indices']
    else:
        indices = list(range(len(val_dataset.attrs) + len(val_dataset.objs)))

    if config.experiment_name == 'clip':

        clip_model, preprocess = load(
            config.clip_model, device=device, context_length=config.context_length
        )

        model = CLIPInterface(clip_model, config, token_ids=None, device=device, enable_pos_emb=True)
    else:
        model, optimizer = get_model(val_dataset, config, device)

        se = soft_embeddings['soft_embeddings']
        if model.soft_embeddings[indices, :].shape == se.shape:
            with torch.no_grad():
                model.soft_embeddings[indices, :] = se
        else:
            raise RuntimeError(f"Error: Incorrect Soft Embedding Shape {se.shape}, Expecting {model.soft_embeddings.shape}!")


    test_dataset = CompositionDataset(dataset_path,
                                      phase='test',
                                      split='compositional-split-natural',
                                      open_world=config.open_world)

    test_dataloader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)


    seen_pairs = []
    offset = len(test_dataset.attrs)
    train_pairs = [(test_dataset.attr2idx[attr], test_dataset.obj2idx[obj]) for attr, obj in test_dataset.train_pairs]
    seen_pairs = [(attr_idx, obj_idx) for attr_idx, obj_idx in train_pairs if attr_idx in indices and (obj_idx + offset) in indices]

    evaluator = MixTrainEvaluator(test_dataset, model=None, seen_pairs=seen_pairs)
    model.to(device)
    _, test_stats = test(model, test_dataset, test_dataloader, evaluator, config, device,)


    bucket_keys = ['pretrained_attr_pretrained_obj', 'pretrained_attr_fine_tuned_obj', 'fine_tuned_attr_fine_tuned_obj',
                   'seen_fine_tuned_attr_fine_tuned_obj', 'unseen_fine_tuned_attr_fine_tuned_obj']

    bucket_pairs = {key: [] for key in bucket_keys}
    bucket_instance_idx = {key: [] for key in bucket_keys}

    offset = len(test_dataset.attrs)
    for pair in evaluator.pairs:
        attr_idx = pair[0]
        obj_idx = pair[1]
        if attr_idx not in indices and (obj_idx + offset) not in indices:
            bucket_pairs['pretrained_attr_pretrained_obj'].append((attr_idx, obj_idx))
        elif attr_idx not in indices and (obj_idx + offset) in indices:
            bucket_pairs['pretrained_attr_fine_tuned_obj'].append((attr_idx, obj_idx))
        elif attr_idx in indices and (obj_idx + offset) in indices:
            bucket_pairs['fine_tuned_attr_fine_tuned_obj'].append((attr_idx, obj_idx))

        if attr_idx in indices and (obj_idx + offset) in indices and \
            (attr_idx, obj_idx) in evaluator.subset_seen_pairs:
            bucket_pairs['seen_fine_tuned_attr_fine_tuned_obj'].append((attr_idx, obj_idx))

        if attr_idx in indices and (obj_idx + offset) in indices and \
            (attr_idx, obj_idx) not in evaluator.subset_seen_pairs:
            bucket_pairs['unseen_fine_tuned_attr_fine_tuned_obj'].append((attr_idx, obj_idx))

    for idx, instance in enumerate(test_dataset.data):
        attr_idx = test_dataset.attr2idx[instance[1]]
        obj_idx = test_dataset.obj2idx[instance[2]]
        for key in bucket_keys:
            if (attr_idx, obj_idx) in bucket_pairs[key]:
                bucket_instance_idx[key].append(idx)

    print([len(bucket_instance_idx[key]) for key in bucket_keys])

    subset_acc = {}
    match_array = np.array(test_stats['match_list'])
    for key in bucket_keys:
        if bucket_instance_idx[key]:
            subset_acc[key] = float(match_array[bucket_instance_idx[key]].mean())
        else:
            subset_acc[key] = 0

    subset_acc['overall'] = float(match_array.mean())
    print(subset_acc)

    results = {
        'test': test_stats,
        'subset_acc': subset_acc,
        'bucket_instance_idx': bucket_instance_idx,
    }

    if config.experiment_name != 'clip':
        result_path = config.soft_embeddings[:-2] + "mix_train.json"

        with open(result_path, 'w+') as fp:
            json.dump(results, fp)

        print('saved at ', result_path)
    else:
        result_path = config.soft_embeddings[:-2] + "mix_train.clip.json"

        with open(result_path, 'w+') as fp:
            json.dump(results, fp)

        print('saved at ', result_path)

    print("done!")
