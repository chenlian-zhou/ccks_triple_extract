# -*- coding: utf-8 -*-
import json
import numpy as np
from keras.models import Model, Input
from keras.layers import Dense, Bidirectional, Dropout, LSTM, TimeDistributed, Masking
from keras.utils import to_categorical, plot_model
from seqeval.metrics import classification_report
import matplotlib.pyplot as plt

from utils import event_type
from utils import MAX_SEQ_LEN, train_file_path, test_file_path, dev_file_path
from load_data import read_data
from albert_zh.extract_feature import BertVector

from tqdm import tqdm

# 利用ALBERT提取文本特征
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=MAX_SEQ_LEN)
f = lambda text: bert_model.encode([text])["encodes"][0]

# 读取label2id字典
with open("%s_label2id.json" % event_type, "r", encoding="utf-8") as h:
    label_id_dict = json.loads(h.read())

id_label_dict = {v: k for k, v in label_id_dict.items()}

# 载入数据
def input_data(file_path):

    sentences, tags = read_data(file_path)
    print("sentences length: %s " % len(sentences))
    print("last sentence: ", sentences[-1])

    # ALBERT ERCODING
    print("start ALBERT encding")
    x = np.array([f(sent) for sent in sentences])
    print("end ALBERT encoding")

    # 对y值统一长度为MAX_SEQ_LEN
    new_y = []
    for seq in tags:
        num_tag = [label_id_dict[_] for _ in seq]
        if len(seq) < MAX_SEQ_LEN:
            num_tag = num_tag + [0] * (MAX_SEQ_LEN-len(seq))
        else:
            num_tag = num_tag[: MAX_SEQ_LEN]

        new_y.append(num_tag)

    # 将y中的元素编码成ont-hot encoding
    y = np.empty(shape=(len(tags), MAX_SEQ_LEN, len(label_id_dict.keys())+1))

    for i, seq in enumerate(new_y):
        y[i, :, :] = to_categorical(seq, num_classes=len(label_id_dict.keys())+1)

    return x, y

# Build model
def build_model(max_para_length, n_tags):
    # Bert Embeddings
    bert_output = Input(shape=(max_para_length, 312, ), name="bert_output")
    # LSTM model
    lstm = Bidirectional(LSTM(units=128, return_sequences=True), name="bi_lstm")(bert_output)
    drop = Dropout(0.1, name="dropout")(lstm)
    out = TimeDistributed(Dense(n_tags, activation="softmax"), name="time_distributed")(drop)
    model = Model(inputs=bert_output, outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 模型结构总结
    model.summary()
    plot_model(model, to_file="albert_bi_lstm.png", show_shapes=True)
    plt.show()

    return model

# 模型训练
def train_model():

    # 读取训练集，验证集和测试集数据
    train_x, train_y = input_data(train_file_path)
    dev_x, dev_y = input_data(dev_file_path)
    test_x, test_y = input_data(test_file_path)

    # 模型训练
    model = build_model(MAX_SEQ_LEN, len(label_id_dict.keys())+1)

    history = model.fit(train_x, train_y, validation_data=(dev_x, dev_y), batch_size=32, epochs=10)

    model.save("%s_ner.h5" % event_type)

    # 绘制loss和accuracy图像
    plt.subplot(2, 1, 1)
    epochs = len(history.history['loss'])
    plt.plot(range(epochs), history.history['loss'], label='loss')
    plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    epochs = len(history.history['accuracy'])
    plt.plot(range(epochs), history.history['accuracy'], label='accuracy')
    plt.plot(range(epochs), history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.savefig("%s_loss_accuracy.png" % event_type)
    plt.show()

    # 模型在测试集上的表现
    # 预测标签
    y = np.argmax(model.predict(test_x), axis=2)
    pred_tags = []
    for i in range(y.shape[0]):
        pred_tags.append([id_label_dict[_] for _ in y[i] if _])

    # 因为存在预测的标签长度与原来的标注长度不一致的情况，因此需要调整预测的标签
    test_sents, test_tags = read_data(test_file_path)
    final_tags = []
    for test_tag, pred_tag in zip(test_tags, pred_tags):
        if len(test_tag) == len(pred_tag):
            final_tags.append(test_tag)
        elif len(test_tag) < len(pred_tag):
            final_tags.append(pred_tag[:len(test_tag)])
        else:
            final_tags.append(pred_tag + ['O'] * (len(test_tag) - len(pred_tag)))

    # 利用seqeval对测试集进行验证
    print(classification_report(test_tags, final_tags, digits=4))


if __name__ == '__main__':
    train_model()

