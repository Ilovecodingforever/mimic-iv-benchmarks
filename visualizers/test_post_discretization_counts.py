from __future__ import absolute_import
from __future__ import print_function

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mimic4models.preprocessing import Discretizer
from mimic4models.fixed_horizon_icu_exit.raw import RawSequenceEncoder
from visualize_utils import observed_mask_information


HEADER = ["Hours", "Heart Rate", "Glucose", "pH"]


def test_fixed_grid_counts_post_discretization_masks():
    X = np.array([
        ["0.13", "82", "", ""],
        ["0.27", "83", "", ""],
        ["1.27", "", "110", ""],
    ], dtype=object)
    discretizer = Discretizer(timestep=1.0, store_masks=True,
                              impute_strategy="previous", start_time="zero")
    data, header = discretizer.transform(X, header=HEADER, end=24.0)
    info = observed_mask_information(data, header)
    assert info["token_count"] == 2
    assert info["occupied_1h_bins"] == 2
    assert info["channels_observed"] == 2
    assert info["channel_counts"]["Heart Rate"] == 1
    assert info["channel_counts"]["Glucose"] == 1


def test_raw_encoder_counts_raw_post_transform_masks():
    X = np.array([
        ["0.13", "82", "", ""],
        ["0.27", "83", "", ""],
        ["1.27", "", "110", ""],
    ], dtype=object)
    data, header = RawSequenceEncoder().transform(X, header=HEADER, end=24.0)
    info = observed_mask_information(data, header)
    assert info["token_count"] == 3
    assert info["channels_observed"] == 2
    assert info["channel_counts"]["Heart Rate"] == 2
    assert info["channel_counts"]["Glucose"] == 1


if __name__ == "__main__":
    test_fixed_grid_counts_post_discretization_masks()
    test_raw_encoder_counts_raw_post_transform_masks()
    print("post-discretization count checks passed")
