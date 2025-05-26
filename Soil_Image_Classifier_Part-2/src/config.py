"""
Team Name: Creatix
Team Members: Siddharth Malkania, Krishan Verma , Rishi Mehrotra
Leaderboard Rank: 69

"""

import os

class Config:
    def __init__(self, base_path="/kaggle/input/path-to-base-directory/"):
        self.BASE_PATH = base_path
        self.TRAIN_LABELS_CSV = os.path.join(base_path, "train_labels.csv")
        self.TRAIN_IMAGES_DIR = os.path.join(base_path, "train")
        self.TEST_IMAGES_DIR = os.path.join(base_path, "test")
        self.TEST_IDS_CSV = os.path.join(base_path, "test_ids.csv")
        self.OUTPUT_CSV = "result_output.csv"