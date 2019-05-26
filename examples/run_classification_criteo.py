import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import multi_gpu_model

from tensorflow.python.keras.callbacks import TensorBoard

from deepctr.models import DeepFM
from deepctr.utils import SingleFeat

if __name__ == "__main__":

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    names = list(['label'])
    names.extend(dense_features)
    names.extend(sparse_features)
    dtypes = {
        'label': np.float32,
    }
    for item in dense_features:
        dtypes[item] = np.float32
    for item in sparse_features:
        dtypes[item] = str

    file = '../../arboretum_benchmark/data/dac/train.txt'

    data = pd.read_csv(file,
                       sep='\t', header=None, names=names, dtype=dtypes)

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat]).astype(np.int32)
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features]).astype(np.float32)

    # 2.count #unique features for each sparse field,and record dense feature field name

    sparse_feature_list = [SingleFeat(feat, data[feat].nunique())
                           for feat in sparse_features]
    dense_feature_list = [SingleFeat(feat, 0,)
                          for feat in dense_features]

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2)
    train_model_input = [train[feat.name].values for feat in sparse_feature_list] + \
                        [train[feat.name].values for feat in dense_feature_list]
    test_model_input = [test[feat.name].values for feat in sparse_feature_list] + \
                       [test[feat.name].values for feat in dense_feature_list]

    # 4.Define Model,train,predict and evaluate
    model = DeepFM({"sparse": sparse_feature_list,
                    "dense": dense_feature_list}, task='binary', embedding_size=4, dnn_hidden_units=(64, 64))
    # parallel_model = multi_gpu_model(model, gpus=2)
    model.compile(tf.keras.optimizers.Adam(1e-4), "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    tensorboard = TensorBoard(log_dir="logs/DeepFM")

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2, validation_split=0.2, callbacks=[tensorboard])
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
