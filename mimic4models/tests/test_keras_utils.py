from __future__ import absolute_import

import os
import tempfile

import h5py

from mimic4models import keras_utils


def test_legacy_h5py_attrs_are_bytes_for_old_keras():
    fd, path = tempfile.mkstemp(suffix='.h5')
    os.close(fd)
    with h5py.File(path, 'w') as handle:
        handle.attrs['keras_version'] = '2.1.2'
        handle.attrs['backend'] = 'tensorflow'
        handle.attrs['layer_names'] = ['lstm']
        handle.attrs['weight_names'] = ['kernel']

    try:
        keras_utils.patch_legacy_keras_h5py_attrs()
        with h5py.File(path, 'r') as handle:
            assert handle.attrs['keras_version'] == b'2.1.2'
            assert handle.attrs['backend'] == b'tensorflow'
            assert handle.attrs['layer_names'][0] == b'lstm'
            assert handle.attrs['weight_names'][0] == b'kernel'
    finally:
        os.remove(path)


if __name__ == '__main__':
    test_legacy_h5py_attrs_are_bytes_for_old_keras()
    print('ok')
