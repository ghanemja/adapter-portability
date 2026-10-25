"""adaptport: cross-version LoRA adapter portability via representational alignment.

Pipeline:
    anchors -> extract per-module activations on M_v and M_{v+1}
            -> cka.py        : measure representational drift
            -> alignment.py  : fit P_in, P_out per module
            -> transfer.py   : transplant adapter (naive or aligned)
            -> eval.py       : score transplanted adapter on task data
"""

from .config import ExperimentConfig, load_config

__all__ = ["ExperimentConfig", "load_config"]
__version__ = "0.1.0"
