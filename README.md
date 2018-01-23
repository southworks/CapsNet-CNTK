# CapsNet-CNTK

A CNTK implementation of CapsNet based on Geoffrey Hinton's paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

## Requeriments

- [Python](https://www.python.org/)
- [CNTK 2.3.1](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-Windows-Python?tabs=cntkpy231)
- Tensorboard (optional)

## Architecture

<a href="images/CapsNetArch.png"><img src="images/CapsNetArch.png"  width="70" height="450"></a>

## Training

```
git clone https://github.com/southworkscom/CapsNet-CNTK.git
cd CapsNet-CNTK
python get_data.py
python main.py
```

### Tensorboard

To activate tensorboard, run the following command from the CapsNet-CNTK folder.

```
tensorboard --logdir tensorboard
```

Then navigate to http://localhost:6006

## TODO

- Add Reconstruction Layer
- Add Benchmarks