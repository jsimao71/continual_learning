import json
from pathlib import Path

import torch

from cl.experiments.paper05_rnn_comparison import RecurrentLM, axis_values, evaluation, make_example

CONFIG=json.loads(Path("configs/paper05/rnn_comparison.json").read_text())


def test_rnn_axes_complete_and_deterministic():
    rows=evaluation(CONFIG,2);assert len(rows)==sum(map(len,axis_values(CONFIG).values()))*2
    a=make_example(CONFIG,"span",64,7,"test");b=make_example(CONFIG,"span",64,7,"test")
    assert a==b and len(a["tokens"])==CONFIG["sequence_length"] and a["dependency_span"]==64


def test_recurrent_shapes():
    x=torch.randint(0,CONFIG["vocab_size"],(3,10))
    for kind in ("rnn","gru","lstm"):
        logits,hidden=RecurrentLM(CONFIG["vocab_size"],16,kind)(x)
        assert logits.shape==(3,10,CONFIG["vocab_size"]) and hidden.shape==(3,10,16)
