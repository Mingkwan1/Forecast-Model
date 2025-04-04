import timesfm
import os
# from loader import Load
from multiprocessing import Process
from multiprocessing import freeze_support

os.environ['HF_HOME'] = r'C:\Users\(Ming)MingkwanRattan\OneDrive - STelligence Co., Ltd\Play\Forecast\OilPrice\cache'

class Forecast:
  def __init__(self):
    pass
  def forecast(self, input_data):
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu",
            input_patch_len= 32,
            output_patch_len = 128,
            per_core_batch_size=32,
            horizon_len=50,
            num_layers=20,
            # num_heads=8,
            model_dims=1280,
            context_len=512,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
            # huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
    )

    forecast_df = tfm.forecast_on_df(
        inputs=input_data,
        freq="D",  #Daily
        value_name="y",
        num_jobs=-1,
    )

    return forecast_df
  
if __name__ == "__main__":
  freeze_support()
