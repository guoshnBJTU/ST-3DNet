# ST3DNet

Deep Spatial–Temporal 3D Convolutional Neural Networks for Trafﬁc Data Forecasting

<img src="fig/ST3DNet architecture.png" alt="image-20200103164326338" style="zoom:50%;" />

# Reference

```latex
@article{guo2019deep,
  title={Deep Spatial-Temporal 3D Convolutional Neural Networks for Traffic Data Forecasting},
  author={Guo, Shengnan and Lin, Youfang and Li, Shijie and Chen, Zhaoming and Wan, Huaiyu},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2019},
  publisher={IEEE}
}
```

# Datasets

Step 1: Download the datasets provided by the paper 'Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction' (https://ojs.aaai.org/index.php/AAAI/article/view/10735)  

Step 2: process dataset

```shell
python prepareDataNY.py
python prepareDataBJ.py
```

# Train and Test

```shell
python trainNY.py
python trainBJ.py
```

