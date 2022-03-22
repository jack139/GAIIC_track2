#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 转换为 ccks2019 格式

import json
import os
import numpy as np


ORI_DATA_DIR = "../../data/test/1/"
TARGET_DATA_DIR = "../../data/test/2/"


def convert_data(ori_data_dir, target_data_dir):
    total_num = 0

    for data_filename in os.listdir(ori_data_dir):
        data_filepath = os.path.join(ori_data_dir, data_filename)
        target_filepath = os.path.join(target_data_dir, data_filename)
        samples_list = np.loadtxt(data_filepath,
                                  dtype="str", comments=None, delimiter="\r\n", encoding="utf-8")
        total_num += len(samples_list)

        __convert_data(samples_list, target_filepath)

    print("Converting samples: {}".format(total_num))


def __convert_data(samples_list, target_filepath, delimiter="\n"):
    ff = open(target_filepath, "a", encoding="utf-8-sig")

    for i in range(len(samples_list)):
        word2tag = []
        sample = json.loads(samples_list[i])

        original_text = sample["text"]

        new_item = {
            "originalText" : original_text,
            "entities" : []
        }

        entities = sample["mention_data"]
        for entity in entities:
            if len(entity) < 1:
                continue
            start_pos = int(entity["offset"])
            end_pos = start_pos + len(entity["mention"])
            label_type = entity["label"]
            if original_text[start_pos:end_pos]!=entity["mention"]:
                print("warning: ", i, entity["mention"])

            new_item["entities"].append({
                "label_type" : label_type,
                "start_pos" : start_pos,
                "end_pos" : end_pos,
            })

        json_text = json.dumps(new_item, ensure_ascii=False)

        # 写入文件
        ff.write(json_text)
        ff.write(delimiter)

    ff.close() 


if __name__ == '__main__':
    convert_data(ori_data_dir=ORI_DATA_DIR, target_data_dir=TARGET_DATA_DIR)
