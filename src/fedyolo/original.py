"""The averaging function from the 2023 notebooks, kept word for word.

This is the function Experiments 2 to 5 of the paper used (cell 10 of each
notebook in ``original/notebooks``). It does not average anything:
``model1.state_dict()`` returns a new dictionary on every call, so assigning
into it leaves the model untouched, and the "average" it returns is the first
client's model. ``tests/test_averaging.py`` demonstrates this.

It is here so that the error can be shown, and so that the paper's procedure
can be re-run as it was (``--averaging original``). Use
:func:`fedyolo.averaging.average_checkpoints` for anything else.

Under PyTorch 2.6 or later, ``torch.load`` refuses these checkpoints unless
``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`` is set, which the runner does.
"""

import torch


def average_weights(weight_files):
    avg_weights = {}
    weights = [torch.load(wf) for wf in weight_files]
    model1 = weights[0]['model']
    for name, param1 in model1.named_parameters():
      params = param1
      for idx in range(1, len(weights)):
        param = weights[idx]['model'].state_dict()[name]
        params = params + param
      params = params / len(weights)
      model1.state_dict()[name] = params

    for key in weights[0].keys():
        weight_values = [w[key] for w in weights if w[key] is not None]
        if weight_values:
          if key != 'model':
            avg_weights[key] = sum(weight_values) / len(weights)
          else:
            avg_weights['model'] = model1
            # print(avg_weights)
    for key in weights[0].keys():
      if key not in avg_weights.keys():
        avg_weights[key] = None
    avg_weights['epoch'] = int(avg_weights['epoch'])
    return avg_weights
