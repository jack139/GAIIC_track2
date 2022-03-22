#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 数据源 ccks2019

import json
import os
import numpy as np

# The JSON keys used in the original data files
JSON_ORI_TXT_KEY = "originalText"
JSON_ENTITIES_KEY = "entities"
JSON_START_POS_KEY = "start_pos"
JSON_END_POS_KEY = "end_pos"
JSON_LABEL_KEY = "label_type"
JSON_OVERLAP_KEY = "overlap"

ORI_DATA_DIR = "../../data/test/original_data/"
PROC_DATA_DIR = "../../data/test/processed_data/"

vocab_attr = set(['null'])

def preprocess_tagged_data(ori_data_dir, train_data_filepath, test_data_filepath="", test_split=0):
    train_total_num = 0
    test_total_num = 0

    for data_filename in os.listdir(ori_data_dir):
        data_filepath = os.path.join(ori_data_dir, data_filename)
        samples_list = np.loadtxt(data_filepath,
                                  dtype="str", comments=None, delimiter="\r\n", encoding="utf-8-sig")
        test_sample_num = int(len(samples_list) * test_split)
        train_sample_num = int(len(samples_list) - test_sample_num)
        train_total_num += train_sample_num
        test_total_num += test_sample_num

        if len(test_data_filepath) > 0 and test_split > 0:
            __preprocess_tagged_data(samples_list[0:train_sample_num], train_data_filepath)
            __preprocess_tagged_data(samples_list[train_sample_num:], test_data_filepath)
        else:
            __preprocess_tagged_data(samples_list, train_data_filepath)

    print("Training samples: {}, Testing samples: {}".format(train_total_num, test_total_num))



def __preprocess_tagged_data(samples_list, tagged_data_filepath, delimiter="\n"):
    f_in_char = open(os.path.join(tagged_data_filepath, 'input.seq.char'), "a", encoding="utf-8")
    f_out_attr = open(os.path.join(tagged_data_filepath, 'output.seq.attr'), "a", encoding="utf-8")
    f_out_bio = open(os.path.join(tagged_data_filepath, 'output.seq.bio'), "a", encoding="utf-8")

    max_len = 0

    for i in range(len(samples_list)):
        word2tag = []
        sample = json.loads(samples_list[i])

        original_text = sample[JSON_ORI_TXT_KEY]

        for w in original_text:
            word2tag.append([w, 'O', 'null'])

        entities = sample[JSON_ENTITIES_KEY]
        for entity in entities:
            if len(entity) < 1:
                continue
            start_pos = entity[JSON_START_POS_KEY]
            end_pos = entity[JSON_END_POS_KEY]
            label_type = entity[JSON_LABEL_KEY]
            vocab_attr.add(label_type)
            if end_pos-start_pos==1:
                word2tag[start_pos][1] = "S"
                word2tag[start_pos][2] = label_type
            else:
                word2tag[start_pos][1] = "B"
                word2tag[start_pos][2] = label_type
                for j in range(start_pos + 1, end_pos - 1):
                    word2tag[j][1] = "I"
                    word2tag[j][2] = label_type
                word2tag[end_pos-1][1] = "E"
                word2tag[end_pos-1][2] = label_type

        # 写入文件
        length = 0 
        tmp_char = []
        tmp_bio = []
        tmp_attr = []

        for i in word2tag:
            tmp_char.append(i[0])
            tmp_bio.append(i[1])
            tmp_attr.append(i[2])

            length += 1

            # 接近100个字就要换行
            if  (length>50) and (i[0] in ['；', '，', '。', ',', '）', '、', ';']): 
                if len(''.join(tmp_bio).replace('O',''))>0: # 只有O的行不保存
                    f_in_char.write(' '.join(tmp_char))
                    f_in_char.write(delimiter)
                    f_out_bio.write(' '.join(tmp_bio))
                    f_out_bio.write(delimiter)
                    f_out_attr.write(' '.join(tmp_attr))
                    f_out_attr.write(delimiter)
                    max_len = max(max_len, length)

                    if length>200:
                        print(''.join(tmp_char))

                length = 0
                tmp_char = []
                tmp_bio = []
                tmp_attr = []

        # 一条结束后，如果还有剩余字符，都进行换行
        if length>0:
            if len(''.join(tmp_bio).replace('O',''))>0: # 只有O的行不保存
                f_in_char.write(' '.join(tmp_char))
                f_in_char.write(delimiter)
                f_out_bio.write(' '.join(tmp_bio))
                f_out_bio.write(delimiter)
                f_out_attr.write(' '.join(tmp_attr))
                f_out_attr.write(delimiter)

        #for i in range(len(word2tag)):
        #    print('%s\t%s\t%s'%(word2tag[i][0],word2tag[i][1],word2tag[i][2]))

    f_in_char.close() 
    f_out_attr.close()
    f_out_bio.close()

    print("max length= ", max_len)


if __name__ == '__main__':
    preprocess_tagged_data(ori_data_dir=ORI_DATA_DIR+'train', train_data_filepath=PROC_DATA_DIR+'train')
    preprocess_tagged_data(ori_data_dir=ORI_DATA_DIR+'val', train_data_filepath=PROC_DATA_DIR+'val')

    # 保存属性值
    with open(os.path.join(PROC_DATA_DIR, 'vocab_attr.txt'), "a", encoding="utf-8") as f:
        for i in vocab_attr:
            f.write(i)
            f.write("\n")
