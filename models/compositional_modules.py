import os

from models.coop import coop
from models.csp import get_csp, get_mix_csp

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_model(train_dataset, config, device):
    if config.experiment_name == "coop":
        return coop(train_dataset, config, device)

    elif config.experiment_name == "csp":
        return get_csp(train_dataset, config, device)

    # special experimental setup
    elif config.experiment_name == "mix_csp":
        return get_mix_csp(train_dataset, config, device)
    else:
        raise NotImplementedError(
            "Error: Unrecognized Experiment Name {:s}.".format(
                config.experiment_name
            )
        )
