#! -*- coding: utf-8 -*-
# 用GlobalPointer做中文命名实体识别

import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.backend import multilabel_categorical_crossentropy
from bert4keras.layers import GlobalPointer
from bert4keras.layers import EfficientGlobalPointer as GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.models import Model
from tqdm import tqdm

import jieba
jieba.initialize()

maxlen = 128
epochs = 30
batch_size = 64
learning_rate = 2e-5
categories = set()

# large
config_path = '../../nlp_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_config.json'
checkpoint_path = '../../nlp_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/bert_model.ckpt'
dict_path = '../../nlp_model/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16/vocab.txt'

def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    for d in json.load(open(filename)):
        D.append([d['text']])
        for e in d['entities']:
            start, end, label = e['start_idx'], e['end_idx'], e['type']
            if start <= end:
                D[-1].append((start, end, label))
            categories.add(label)
    return D


# 标注数据
train_data = load_data('./data/train.json')
valid_data = load_data('./data/dev.json')
categories = list(sorted(categories))

# 建立分词器
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros((len(categories), maxlen, maxlen))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    label = categories.index(label)
                    labels[label, start, end] = 1
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels[:, :len(token_ids), :len(token_ids)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels, seq_dims=3)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    bh = K.prod(K.shape(y_pred)[:2])
    y_true = K.reshape(y_true, (bh, -1))
    y_pred = K.reshape(y_pred, (bh, -1))
    return K.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def global_pointer_f1_score(y_true, y_pred):
    """给GlobalPointer设计的F1
    """
    y_pred = K.cast(K.greater(y_pred, 0), K.floatx())
    return 2 * K.sum(y_true * y_pred) / K.sum(y_true + y_pred)


model = build_transformer_model(config_path, checkpoint_path)
output = GlobalPointer(len(categories), 64)(model.output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=global_pointer_crossentropy,
    optimizer=Adam(learning_rate),
    metrics=[global_pointer_f1_score]
)


class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def recognize(self, text, threshold=0):
        tokens = tokenizer.tokenize(text, maxlen=512)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        scores = model.predict([token_ids, segment_ids])[0]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        for l, start, end in zip(*np.where(scores > threshold)):
            entities.append(
                (mapping[start][0], mapping[end][-1], categories[l])
            )
        return entities


NER = NamedEntityRecognizer()


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(NER.recognize(d[0]))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('./globalpointer_best_f1_%.5f.weights'%f1)
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


# 保存两个格式： BIO 和 json
def predict_to_file(in_file, out_file):
    """预测到文件
    """
    D = []
    outf = open(out_file, 'w') # BIO 文件
    data = open(in_file).readlines() # text 文本

    for d in tqdm(data, ncols=100):
        d2 = {
            'entities' : [],
            'text' : d[:-1] if d[-1]=='\n' else d # 去掉末尾 \n
        }

        if len(d2['text'])==0: # 忽略空行
            continue

        # 初始化 BIO 标记
        char = [c for c in d2['text']]
        label = ['O']*len(d2['text'])

        # 识别
        entities = NER.recognize(d2['text'])
        for e in entities:
            d2['entities'].append({
                'start_idx': e[0],
                'end_idx': e[1],
                'type': e[2]
            })

            # 生成 BIO标记
            label[e[0]] = 'B-'+e[2]
            for x in range(e[0]+1, e[1]+1):
                label[x] = 'I-'+e[2]

        D.append(d2)

        # 写入 BIO文件
        for i in range(len(char)):
            outf.write(char[i]+' '+label[i]+'\n')

        outf.write('\n') # 写入一个空行结束一个text

    outf.close()

    # 保存json格式
    #json.dump(
    #    D,
    #    open(out_file+'.json', 'w', encoding='utf-8'),
    #    indent=4,
    #    ensure_ascii=False
    #)


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    #model.load_weights('./globalpointer_best_f1_0.80232.weights')

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:
    model.load_weights('./globalpointer_best_f1_0.80232.weights')
    predict_to_file('../data/preliminary_test_a/sample_per_line_preliminary_A.txt', './test.txt')