"""Console-script entry points.

Each CLI takes a single ``--config path/to.yaml`` and is wired into
``[project.scripts]`` in pyproject.toml:

    neuropaths-generate --config configs/square_32.yaml
    neuropaths-train    --config configs/square_32.yaml
    neuropaths-evaluate --config configs/square_32.yaml

These replace the three legacy bare-main scripts and guarantee nothing
runs at import time (the original `FNO_1.py` and `training_data_1_generator.py`
call ``main()`` at module level, so importing them triggers training).
"""
