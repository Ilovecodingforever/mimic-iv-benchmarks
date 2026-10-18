from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import random
random.seed(49297)
from tqdm import tqdm

from mimic4benchmark.decompensation import NO_DEATH_TIME, PREDICTION_TIME, mortality_label


METADATA_HEADER = ('stay', 'mortality', 'lived_time', 'intime', 'deathtime',
                   'label_24h')


def _mortality_metadata(stay_name, mortality, lived_time, intime, deathtime):
    return (stay_name, int(mortality), float(lived_time), str(intime),
            '' if pd.isnull(deathtime) else str(deathtime),
            mortality_label(mortality, lived_time, PREDICTION_TIME, 24.0))


def _write_metadata(path, rows):
    with open(path, 'w') as metadata:
        metadata.write(','.join(METADATA_HEADER) + '\n')
        for row in sorted(rows):
            metadata.write('{},{},{:.12g},{},{},{}\n'.format(*row))


def _metadata_for_existing_partition(args, partition):
    """Recover timing metadata without rereading or rewriting time series."""
    output_dir = os.path.join(args.output_path, partition)
    listfile_path = os.path.join(output_dir, 'listfile.csv')
    rows = []
    seen = set()
    with open(listfile_path) as listfile:
        next(listfile)
        for line in listfile:
            if not line.strip():
                continue
            stay_name = line.split(',', 1)[0]
            if stay_name in seen:
                continue
            seen.add(stay_name)
            patient, ts_filename = stay_name.split('_', 1)
            patient_folder = os.path.join(args.root_path, partition, patient)
            label_filename = ts_filename.replace('_timeseries', '')
            label_df = pd.read_csv(os.path.join(patient_folder, label_filename))
            if label_df.shape[0] == 0:
                raise ValueError('Empty label file for {}'.format(stay_name))
            label = label_df.iloc[0]
            stays_df = pd.read_csv(os.path.join(patient_folder, 'stays.csv'))
            stay = stays_df[stays_df.ICUSTAY_ID == label['Icustay']]
            if stay.shape[0] != 1:
                raise ValueError('Expected one ICU stay for {}'.format(stay_name))
            stay = stay.iloc[0]
            deathtime = stay['DEATHTIME']
            intime = stay['INTIME']
            if pd.isnull(deathtime):
                lived_time = NO_DEATH_TIME
            else:
                lived_time = (datetime.strptime(str(deathtime), '%Y-%m-%d %H:%M:%S') -
                              datetime.strptime(str(intime), '%Y-%m-%d %H:%M:%S')).total_seconds() / 3600.0
            rows.append(_mortality_metadata(
                stay_name, int(label['Mortality']), lived_time, intime, deathtime))
    return rows


def process_partition(args, partition, sample_rate=1.0, shortest_length=4.0,
                      eps=1e-6, future_time_interval=24.0):

    output_dir = os.path.join(args.output_path, partition)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    xty_triples = []
    metadata_rows = []
    patients = list(filter(str.isdigit, os.listdir(os.path.join(args.root_path, partition))))
    for patient in tqdm(patients, desc='Iterating over patients in {}'.format(partition)):
        patient_folder = os.path.join(args.root_path, partition, patient)
        patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(patient_folder)))
        stays_df = pd.read_csv(os.path.join(patient_folder, "stays.csv"))

        for ts_filename in patient_ts_files:
            with open(os.path.join(patient_folder, ts_filename)) as tsfile:
                lb_filename = ts_filename.replace("_timeseries", "")
                label_df = pd.read_csv(os.path.join(patient_folder, lb_filename))

                # empty label file
                if label_df.shape[0] == 0:
                    continue

                mortality = int(label_df.iloc[0]["Mortality"])

                los = 24.0 * label_df.iloc[0]['Length of Stay']  # in hours
                if pd.isnull(los):
                    print("(length of stay is missing)", patient, ts_filename)
                    continue

                stay = stays_df[stays_df.ICUSTAY_ID == label_df.iloc[0]['Icustay']]
                deathtime = stay['DEATHTIME'].iloc[0]
                intime = stay['INTIME'].iloc[0]
                if pd.isnull(deathtime):
                    lived_time = NO_DEATH_TIME
                else:
                    lived_time = (datetime.strptime(deathtime, "%Y-%m-%d %H:%M:%S") -
                                  datetime.strptime(intime, "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600.0

                ts_lines = tsfile.readlines()
                header = ts_lines[0]
                ts_lines = ts_lines[1:]
                event_times = [float(line.split(',')[0]) for line in ts_lines]

                ts_lines = [line for (line, t) in zip(ts_lines, event_times)
                            if -eps < t < los + eps]
                event_times = [t for t in event_times
                               if -eps < t < los + eps]

                # no measurements in ICU
                if len(ts_lines) == 0:
                    print("(no events in ICU) ", patient, ts_filename)
                    continue

                sample_times = np.arange(0.0, min(los, lived_time) + eps, sample_rate)

                sample_times = list(filter(lambda x: x > shortest_length, sample_times))

                # At least one measurement
                sample_times = list(filter(lambda x: x > event_times[0], sample_times))

                output_ts_filename = patient + "_" + ts_filename
                with open(os.path.join(output_dir, output_ts_filename), "w") as outfile:
                    outfile.write(header)
                    for line in ts_lines:
                        outfile.write(line)

                metadata_rows.append(_mortality_metadata(
                    output_ts_filename, mortality, lived_time, intime, deathtime))

                for t in sample_times:
                    cur_mortality = mortality_label(
                        mortality, lived_time, t, future_time_interval)
                    xty_triples.append((output_ts_filename, t, cur_mortality))

    print("Number of created samples:", len(xty_triples))
    if partition == "train":
        random.shuffle(xty_triples)
    if partition == "test":
        xty_triples = sorted(xty_triples)

    with open(os.path.join(output_dir, "listfile.csv"), "w") as listfile:
        listfile.write('stay,period_length,y_true\n')
        for (x, t, y) in xty_triples:
            listfile.write('{},{:.6f},{:d}\n'.format(x, t, y))

    return metadata_rows


def main():
    parser = argparse.ArgumentParser(description="Create data for decompensation prediction task.")
    parser.add_argument('root_path', type=str, help="Path to root folder containing train and test sets.")
    parser.add_argument('output_path', type=str, help="Directory where the created data should be stored.")
    parser.add_argument('--metadata-only', action='store_true',
                        help='Write mortality metadata from existing decompensation files without regenerating time series.')
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.metadata_only:
        metadata_rows = (_metadata_for_existing_partition(args, 'test') +
                         _metadata_for_existing_partition(args, 'train'))
    else:
        metadata_rows = process_partition(args, 'test') + process_partition(args, 'train')
    _write_metadata(os.path.join(args.output_path, 'mortality_metadata.csv'), metadata_rows)


if __name__ == '__main__':
    main()
