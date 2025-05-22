import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # 로그 최대치
import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build_info

print(tf_build_info.build_info)

print("버전:", tf.__version__)
print("GPU 디바이스:", tf.config.list_physical_devices('GPU'))