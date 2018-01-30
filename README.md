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

## Results

### Test Error

   Method     |   Routing   |   Reconstruction  |  MNIST (%)  |  *Paper*
   :---------|:------:|:---:|:----:|:----:
   Baseline |  -- | -- | --             | *0.39*
   CapsNet  |  3 | no | 0.55 | 0.35 (0.036)
   CapsNet  |  3 | yes| WIP | 0.25 (0.005)

### Reconstruction layer

<a href="images/reconstruction.png"><img src="images/reconstruction.png"  width="364" height="224"></a>

### Dimension perturbations

<a href="images/manipulated/0.png"><img src="images/manipulated/0.png"  width="60" height="96"></a>|<a href="images/manipulated/1.png"><img src="images/manipulated/1.png"  width="60" height="96"></a>|<a href="images/manipulated/2.png"><img src="images/manipulated/2.png"  width="60" height="96"></a>|<a href="images/manipulated/3.png"><img src="images/manipulated/3.png"  width="60" height="96"></a>|<a href="images/manipulated/4.png"><img src="images/manipulated/4.png"  width="60" height="96"></a>

<a href="images/manipulated/5.png"><img src="images/manipulated/5.png"  width="60" height="96"></a>|<a href="images/manipulated/6.png"><img src="images/manipulated/6.png"  width="60" height="96"></a>|<a href="images/manipulated/7.png"><img src="images/manipulated/7.png"  width="60" height="96"></a>|<a href="images/manipulated/8.png"><img src="images/manipulated/8.png"  width="60" height="96"></a>|<a href="images/manipulated/9.png"><img src="images/manipulated/9.png"  width="60" height="96"></a>

## TODO

- Complete benchmarks
- Capsule visualization
- Improve the API
- Try other datasets