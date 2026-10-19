from __future__ import absolute_import

import ast
import glob
import os
import shutil
import tempfile


def load_normalizer_helpers():
    path = os.path.join(
        os.path.dirname(__file__), '..', 'length_of_stay', 'main.py')
    with open(path) as source_file:
        tree = ast.parse(source_file.read(), filename=path)
    tree.body = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in (
            'los_normalizer_pattern', 'resolve_normalizer_state')
    ]
    namespace = {'glob': glob, 'os': os, '__file__': path}
    exec(compile(tree, path, 'exec'), namespace)
    return namespace['los_normalizer_pattern'], namespace['resolve_normalizer_state']


def test_length_of_stay_normalizer_dir_resolution():
    pattern, resolve = load_normalizer_helpers()
    root = tempfile.mkdtemp()
    try:
        normalizer_dir = os.path.join(root, 'normalizers')
        os.makedirs(normalizer_dir)
        expected = os.path.join(
            normalizer_dir,
            'los_ts:1.00_impute:previous_start:zero_masks:True_n:123.normalizer')
        open(expected, 'w').close()

        assert resolve(None, normalizer_dir, 1.0, 'previous') == expected
        assert resolve(None, None, 1.0, 'previous').endswith(
            'los_ts1.0.input_str_previous.start_time_zero.n5e4.normalizer')
        assert pattern(normalizer_dir, 1.0, 'previous').endswith(
            'los_ts:1.00_impute:previous_start:zero_masks:True_n:*.normalizer')
        explicit = os.path.join(root, 'explicit.normalizer')
        assert resolve(explicit, normalizer_dir, 1.0, 'previous') == explicit

        open(os.path.join(
            normalizer_dir,
            'los_ts:1.00_impute:previous_start:zero_masks:True_n:456.normalizer'),
            'w').close()
        try:
            resolve(None, normalizer_dir, 1.0, 'previous')
        except IOError as exc:
            assert 'Multiple length-of-stay normalizers' in str(exc)
        else:
            raise AssertionError('ambiguous normalizer directory did not fail')
    finally:
        shutil.rmtree(root)


if __name__ == '__main__':
    test_length_of_stay_normalizer_dir_resolution()
    print('ok')
