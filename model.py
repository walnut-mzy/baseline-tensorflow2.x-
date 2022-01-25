import tensorflow as tf
import numpy as np
from tensorflow import keras
# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0],True)
#     logical_devices = tf.config.list_logical_devices("CPU")

def create_model(is_radual=True):
    ## unet网络结构下采样部分
    # 输入层 第一部分
    inputs = tf.keras.layers.Input(shape=(17,2))
    x1 =tf.keras.layers.Dense(1024)(inputs)
    x = tf.keras.layers.BatchNormalization()(x1)
    x=tf.keras.layers.ReLU()(x)
    x1 = tf.keras.layers.Dense(1024)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    X=tf.keras.layers.Dropout(0.5)(x)
    if is_radual:
        x=x1+x
        #print(x)
    x2 = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.BatchNormalization()(x2)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    X = tf.keras.layers.Dropout(0.5)(x)
    if is_radual:
        x=x1+x2
       # print(x)
    output=tf.keras.layers.Dense(3)(x)
    output=tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.ReLU()(output)
   # print(output)
    return tf.keras.Model(inputs=inputs,outputs=output)
class linear_two(keras.layers.Layer):
    def __init__(self,filters=256,keras_size=3,stride=1,is_begin=False):
        super(linear_two, self).__init__()
        self.dn1=keras.layers.Dense(1024)
        self.dn2 = keras.layers.Dense(1024)
        self.bn1=keras.layers.BatchNormalization()
        self.bn2=keras.layers.BatchNormalization()
        self.relu1=keras.layers.ReLU()
        self.relu2 = keras.layers.ReLU()
        self.drop1=keras.layers.Dropout(0.5)
        self.drop2 = keras.layers.Dropout(0.5)
        self.dn3 = keras.layers.Dense(1024)
        self.dn4 = keras.layers.Dense(1024)
        self.bn3 = keras.layers.BatchNormalization()
        self.bn4 = keras.layers.BatchNormalization()
        self.relu3 = keras.layers.ReLU()
        self.relu4 = keras.layers.ReLU()
        self.drop3 = keras.layers.Dropout(0.5)
        self.drop4 = keras.layers.Dropout(0.5)
        self.dn5=keras.layers.Dense(3)
        self.bn5 = keras.layers.BatchNormalization()
        self.relu5 = keras.layers.ReLU()
        self.drop5 = keras.layers.Dropout(0.5)
    def call(self, inputs, **kwargs):
        x1=self.dn1(inputs)
        x=self.bn1(x1)
        x=self.relu1(x)
        x=self.drop1(x)

        x = self.dn2(x)
        x = self.bn2(x1)
        x = self.relu2(x)
        x = self.drop2(x)
        x=x+x1

        x2 = self.dn3(x)
        x = self.bn3(x2)
        x = self.relu3(x)
        x = self.drop3(x)

        x = self.dn4(x)
        x = self.bn4(x1)
        x = self.relu4(x)
        x = self.drop4(x)

        x=x+x2
        x = self.dn5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.drop5(x)
        print(x)
        return x
model=tf.keras.Sequential([linear_two()])
model.build((None,17,2))
model.summary()
# model=create_model()
# tf.keras.utils.plot_model(model) # 绘制模型图