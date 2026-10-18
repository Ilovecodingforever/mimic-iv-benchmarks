from __future__ import absolute_import
from __future__ import print_function


VALID_HORIZONS = (12, 24, 48, 96, 168)
PREDICTION_TIME = 24.0
NO_DEATH_TIME = 1e18


def mortality_label(mortality, lived_time, prediction_time, future_time_interval):
    """Return the label used by the original decompensation generator."""
    if int(mortality) == 0:
        return 0
    return int(float(lived_time) - float(prediction_time) < float(future_time_interval))
