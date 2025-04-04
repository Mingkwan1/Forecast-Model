import os
import timesfm
import gc
import numpy as np
import pandas as pd
# from timesfm import patched_decoder
# from timesfm import data_loader
from tqdm import tqdm
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
from forecast import Forecast
from loader import Load

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PMAP_USE_TENSORSTORE'] = 'false'

# Assuming the dataframe is already loaded into a variable called `df`
# with columns "ds" for datetime and "y" for the target variable

input_data = Load().load()

# Define the necessary parameters
context_len = 172
pred_len = 32
batch_size = 32

train_batches = input_data.tf_dataset(mode="train", shift=1).batch(batch_size)
val_batches = input_data.tf_dataset(mode="val", shift=pred_len)
test_batches = input_data.tf_dataset(mode="test", shift=pred_len)

# train_batches, val_batches, test_batches = input_data.train_test_split(
# print(train_batches, val_batches, test_batches)

# Iterate through the train batches to check the shape
for tbatch in tqdm(train_batches.as_numpy_iterator()):
    break
print(tbatch[0].shape)

# Calculate MAE loss


mae_losses = []
tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            horizon_len=64,
            num_layers=50,
            context_len=1024,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
    )
for batch in tqdm(test_batches.as_numpy_iterator()):
    past = batch[0]
    actuals = batch[3]
    forecasts, _ = tfm.forecast(list(past), [0] * past.shape[0], normalize=True)
    forecasts = forecasts[:, 0 : actuals.shape[1]]
    mae_losses.append(np.abs(forecasts - actuals).mean())

print(f"MAE: {np.mean(mae_losses)}")
