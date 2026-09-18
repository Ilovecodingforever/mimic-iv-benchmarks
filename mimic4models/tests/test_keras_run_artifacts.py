from __future__ import absolute_import
from __future__ import print_function

import os
import shutil
import sys
import tempfile
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# The artifact helper lives in keras_utils.py, whose metric callbacks import
# Keras. The cleanup tests do not need real Keras, so provide a tiny import stub
# when the test environment does not have the old benchmark dependencies.
if 'keras' not in sys.modules:
    keras = types.ModuleType('keras')
    callbacks = types.ModuleType('keras.callbacks')
    backend = types.ModuleType('keras.backend')
    layers = types.ModuleType('keras.layers')

    class Callback(object):
        pass

    class Layer(object):
        pass

    callbacks.Callback = Callback
    backend.backend = lambda: 'stub'
    layers.Layer = Layer
    keras.callbacks = callbacks
    sys.modules['keras'] = keras
    sys.modules['keras.callbacks'] = callbacks
    sys.modules['keras.backend'] = backend
    sys.modules['keras.layers'] = layers

from mimic4models.keras_utils import cleanup_existing_run_artifacts
from mimic4models.keras_utils import prepare_keras_training_run


def touch(path):
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(path, 'w') as f:
        f.write('x')


def exists(root, rel):
    return os.path.exists(os.path.join(root, rel))


def make_artifacts(root):
    current = 'modelA.seed0'
    other_seed = 'modelA.seed1'
    other_model = 'modelB.seed0'

    touch(os.path.join(root, 'keras_logs', current + '.csv'))
    touch(os.path.join(root, 'keras_logs', other_seed + '.csv'))
    touch(os.path.join(root, 'keras_logs', other_model + '.csv'))

    for name in [current, other_seed, other_model]:
        touch(os.path.join(root, 'keras_states', name + '.epoch1.test0.1.state'))
        touch(os.path.join(root, 'test_predictions', name + '.epoch1.test0.1.state.csv'))
        touch(os.path.join(root, 'test_predictions', 'ihm', name + '.epoch1.test0.1.state.csv'))

    # Prefix trap: must not match because basename does not start with current + '.'.
    touch(os.path.join(root, 'keras_states', 'modelA.seed00.epoch1.test0.1.state'))
    touch(os.path.join(root, 'test_predictions', 'modelA.seed00.epoch1.test0.1.state.csv'))
    return current


def test_fresh_cleanup_removes_only_current_run():
    root = tempfile.mkdtemp()
    try:
        current = make_artifacts(root)
        append = prepare_keras_training_run(root, current, '', mode='train')
        assert append is False
        assert not exists(root, 'keras_logs/modelA.seed0.csv')
        assert not exists(root, 'keras_states/modelA.seed0.epoch1.test0.1.state')
        assert not exists(root, 'test_predictions/modelA.seed0.epoch1.test0.1.state.csv')
        assert not exists(root, 'test_predictions/ihm/modelA.seed0.epoch1.test0.1.state.csv')

        assert exists(root, 'keras_logs/modelA.seed1.csv')
        assert exists(root, 'keras_logs/modelB.seed0.csv')
        assert exists(root, 'keras_states/modelA.seed1.epoch1.test0.1.state')
        assert exists(root, 'keras_states/modelB.seed0.epoch1.test0.1.state')
        assert exists(root, 'keras_states/modelA.seed00.epoch1.test0.1.state')
        assert exists(root, 'test_predictions/modelA.seed00.epoch1.test0.1.state.csv')
    finally:
        shutil.rmtree(root)


def test_resume_and_test_mode_remove_nothing():
    root = tempfile.mkdtemp()
    try:
        current = make_artifacts(root)
        assert prepare_keras_training_run(root, current, '/tmp/state', mode='train') is True
        assert exists(root, 'keras_logs/modelA.seed0.csv')
        assert exists(root, 'keras_states/modelA.seed0.epoch1.test0.1.state')
        assert exists(root, 'test_predictions/modelA.seed0.epoch1.test0.1.state.csv')

        assert prepare_keras_training_run(root, current, '', mode='test') is False
        assert exists(root, 'keras_logs/modelA.seed0.csv')
        assert exists(root, 'keras_states/modelA.seed0.epoch1.test0.1.state')
        assert exists(root, 'test_predictions/modelA.seed0.epoch1.test0.1.state.csv')
    finally:
        shutil.rmtree(root)


def test_no_matching_files_and_empty_name():
    root = tempfile.mkdtemp()
    try:
        removed = cleanup_existing_run_artifacts(root, 'missing.seed0')
        assert removed == {'logs': 0, 'checkpoints': 0, 'predictions': 0}
        try:
            cleanup_existing_run_artifacts(root, '')
        except ValueError as exc:
            assert 'empty model_final_name' in str(exc)
        else:
            raise AssertionError('empty model name did not fail')
    finally:
        shutil.rmtree(root)


if __name__ == '__main__':
    test_fresh_cleanup_removes_only_current_run()
    test_resume_and_test_mode_remove_nothing()
    test_no_matching_files_and_empty_name()
    print('ok')
