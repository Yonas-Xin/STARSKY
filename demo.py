import skystar.model
import numpy as np
import os
import skystar.sky_dataset as sky_dataset
# sky_dataset.create_dataset.data_to_npz("dataset.txt",'data.npz')
x,t =sky_dataset.load('data.npz')
print(x.shape)