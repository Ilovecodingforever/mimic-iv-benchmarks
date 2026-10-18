from __future__ import absolute_import

import json
import os


def test_exp2a_notebook_has_isolated_task_configurations():
    notebook_path = os.path.join(os.path.dirname(__file__), 'visualize_exp2a_horizon.ipynb')
    with open(notebook_path) as notebook_file:
        notebook = json.load(notebook_file)
    source = '\n'.join(''.join(cell.get('source', [])) for cell in notebook['cells'])

    assert 'fixed_horizon_icu_exit' in source
    assert 'fixed_horizon_decompensation' in source
    assert 'data/decompensation' in source
    assert 'results/fixed_horizon_decompensation' in source
    assert 'metadata_path' in source
    assert 'EXP2A_TASK_CONFIG["module"]' in source


if __name__ == '__main__':
    test_exp2a_notebook_has_isolated_task_configurations()
    print('ok')
