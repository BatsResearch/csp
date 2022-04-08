import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

DATASET_PATHS = {
    "mit-states": os.path.join(DIR_PATH, "../data/mit-states"),
    "ut-zappos": os.path.join(DIR_PATH, "../data/ut-zappos"),
    "cgqa": os.path.join(DIR_PATH, "../data/cgqa")
}