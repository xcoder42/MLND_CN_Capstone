'''
深度神经网络 模型
'''
from keras.layers import Dense, Input, Lambda, Dropout, GlobalAveragePooling1D, Activation, Reshape ,GaussianNoise
from keras.layers import Embedding, LSTM, Conv1D, Bidirectional
from keras.layers.merge import concatenate, add, multiply, average
from keras.models import Model
from keras.layers.normalization import BatchNormalization
import keras.backend as K


def euclidean_distance(tensors):
    x, y = tensors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def manhattan_distance(tensors):
    x, y = tensors
    return K.exp(-K.sum(K.abs(x - y), axis=1, keepdims=True))

def cosine_similarity(tensors):
    x, y = tensors
    norm_x = K.sqrt(K.sum(K.square(x), axis=1, keepdims=True))
    norm_y = K.sqrt(K.sum(K.square(x), axis=1, keepdims=True))
    x_y = K.sum(x * y, axis=1, keepdims=True)
    return x_y / (norm_x * norm_y)
    
def build_model_lstm_cnn_fs(
    feature_num,
    num_words,
    embedding_dim,
    embedding_matrix,
    max_sequence_length,
    rate_drop,
    act
):
    # 模型的输入部分
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_sequence_length,
                            trainable=False)
    # 句子1embedding
    sequence_1_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    # 句子2embedding
    sequence_2_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    #embedded_shape = embedded_sequences_1._keras_shape
    
    # 1d cnn模型层
    # ps. 这个写法略显臃肿，之后可以找找看其他的写法
    # layer的输入必须是张量（tensor）而不是layer实例
    cnn_layer = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')
    cnn_1 = cnn_layer(embedded_sequences_1)
    cnn_2 = cnn_layer(embedded_sequences_2)
    cnn_layer = Dropout(rate_drop)
    cnn_1 = cnn_layer(cnn_1)
    cnn_2 = cnn_layer(cnn_2)
    cnn_layer = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')
    cnn_1 = cnn_layer(cnn_1)
    cnn_2 = cnn_layer(cnn_2)
    cnn_layer = BatchNormalization()
    cnn_1 = cnn_layer(cnn_1)
    cnn_2 = cnn_layer(cnn_2)
    cnn_layer = GlobalAveragePooling1D()
    cnn_1 = cnn_layer(cnn_1)
    cnn_2 = cnn_layer(cnn_2)
    # 使用张量差和积来作为两个句子的特征
    cnn_layer_diff = Lambda(lambda x: K.abs(x[0] - x[1]))([cnn_1, cnn_2])
    cnn_layer_multiply = Lambda(lambda x: K.abs(x[0] * x[1]))([cnn_1, cnn_2])
    
    
    # lstm模型层 
    lstm_layer = LSTM(150, dropout=rate_drop, recurrent_dropout=rate_drop)
    lstm_1 = lstm_layer(embedded_sequences_1)
    lstm_2 = lstm_layer(embedded_sequences_2)
    lstm_layer = BatchNormalization()
    lstm_1 = lstm_layer(lstm_1)
    lstm_2 = lstm_layer(lstm_2)
    # 使用张量差和积来作为两个句子的特征
    lstm_layer_diff = Lambda(lambda x: K.abs(x[0] - x[1]))([lstm_1, lstm_2])
    lstm_layer_multiply = Lambda(lambda x: K.abs(x[0] * x[1]))([lstm_1, lstm_2])
    
    
    # feature dense层
    features_input = Input(shape=(feature_num,), dtype="float32")
    features_dense = BatchNormalization()(features_input)
    features_dense = Dense(200, activation="relu")(features_dense)
    features_dense = Dropout(rate_drop)(features_dense)
    
    
    # merge 所有的模型层
    merge = concatenate([cnn_layer_diff, cnn_layer_multiply, lstm_layer_diff, lstm_layer_multiply, features_dense])
    
    # 加入MLP层
    merge = Dropout(rate_drop)(merge)
    merge = BatchNormalization()(merge)
    merge = Dense(50, activation='relu')(merge)
    
    merge = Dropout(rate_drop)(merge)
    merge = BatchNormalization()(merge)
    final = Dense(1, activation='sigmoid')(merge)
    
    model = Model(inputs=[sequence_1_input, sequence_2_input, features_input], outputs=final)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    
    return model

def build_model_lstm_cnn_fs_v2(
    feature_num,
    num_words,
    embedding_dim,
    embedding_matrix,
    max_sequence_length,
    rate_drop_lstm,
    rate_drop_dense
):
        # 模型的输入部分
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_sequence_length,
                            trainable=False)
    # 句子1embedding
    sequence_1_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    # 句子2embedding
    sequence_2_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    #embedded_shape = embedded_sequences_1._keras_shape
    
    # cnn层 不同的kernel_size相当于ngram
    conv1 = Conv1D(filters=64, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
    conv3 = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')
    
    # 用全局平均池化降维
    conv1_l = conv1(embedded_sequences_1)
    global1_l = GlobalAveragePooling1D()(conv1_l)
    conv1_r = conv1(embedded_sequences_2)
    global1_r = GlobalAveragePooling1D()(conv1_r)
    
    conv2_l = conv2(embedded_sequences_1)
    global2_l = GlobalAveragePooling1D()(conv2_l)
    conv2_r = conv2(embedded_sequences_2)
    global2_r = GlobalAveragePooling1D()(conv2_r)
    
    conv3_l = conv3(embedded_sequences_1)
    global3_l = GlobalAveragePooling1D()(conv3_l)
    conv3_r = conv3(embedded_sequences_2)
    global3_r = GlobalAveragePooling1D()(conv3_r)
    
    cnn_l_concat = concatenate([global1_l, global2_l, global3_l])
    cnn_r_concat = concatenate([global1_r, global2_r, global3_r])
    #cnn_l_average = average([global1_l, global2_l, global3_l])
    #cnn_r_average = average([global1_r, global2_r, global3_r])
    
    #cnn_diff = Lambda(lambda x: K.abs(x[0] - x[1]))([cnn_left_concat, cnn_right_concat])
    #cnn_multiply = Lambda(lambda x: x[0] * x[1])([cnn_left_concat, cnn_right_concat])
    cnn_diff = Lambda(lambda x: K.abs(x[0] - x[1]))([cnn_l_concat, cnn_r_concat])
    cnn_multiply = Lambda(lambda x: x[0] * x[1])([cnn_l_concat, cnn_r_concat])
    cnn_merge = concatenate([cnn_diff, cnn_multiply])
    cnn_merge = Dropout(rate_drop_dense)(cnn_merge)
    
    
    # lstm模型层 ,采用了双向LSTM 
    lstm_layer = Bidirectional(LSTM(64, recurrent_dropout=rate_drop_lstm))
    lstm_1 = lstm_layer(embedded_sequences_1)
    lstm_2 = lstm_layer(embedded_sequences_2)
    # 使用张量差和积来作为两个句子的特征
    #lstm_diff = Lambda(lambda x: K.abs(x[0] - x[1]))([lstm_1, lstm_2])
    #lstm_multiply = Lambda(lambda x: x[0] * x[1])([lstm_1, lstm_2])
    lstm_add = add([lstm_1, lstm_2])
    lstm_square = Lambda(lambda x: K.square(x[0] - x[1]))([lstm_1, lstm_2])
    lstm_mergre = concatenate([lstm_add, lstm_square])
    lstm_mergre = Dropout(rate_drop_dense)(lstm_mergre)
    
    # feature dense层
    features_input = Input(shape=(feature_num,), dtype="float32")
    features_dense = BatchNormalization()(features_input)
    features_dense = Dense(200, activation="relu")(features_dense)
    features_dense = Dropout(rate_drop_dense)(features_dense)
    
    
    # merge 所有的模型层
    merge = concatenate([cnn_merge,  features_dense, lstm_mergre])
    
    # 加入MLP层
    merge = Dropout(rate_drop_dense)(merge)
    merge = BatchNormalization()(merge)
    merge = Dense(150, activation='relu')(merge)
    
    merge = Dropout(rate_drop_dense)(merge)
    merge = BatchNormalization()(merge)
    final = Dense(1, activation='sigmoid')(merge)
    
    model = Model(inputs=[sequence_1_input, sequence_2_input, features_input], outputs=final)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    
    return model