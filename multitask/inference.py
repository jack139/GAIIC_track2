import logging
import tensorflow as tf
import numpy as np
import os
from datetime import datetime

from ner.model_multitask_bert import MyModel
from ner.bert import modeling as bert_modeling
from ner.utils import DataProcessor_MTL_BERT as DataProcessor
from ner.utils import load_vocabulary
from ner.utils import extract_kvpairs_in_bioes
from ner.utils import cal_f1_score
from ner.utils import prepare_data


# base model: chinese_bert_L-12_H-768_A-12
bert_vocab_path = "../../nlp_model/chinese_bert_L-12_H-768_A-12/vocab.txt"
bert_config_path = "../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_config.json"
#bert_ckpt_path = "../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_model.ckpt"
bert_ckpt_path = "./ckpt/model.ckpt.batch4700_0.7603"

# set logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
logger.addHandler(chlr)

logger.info("loading vocab...")

data_path = "./data"
w2i_char, i2w_char = load_vocabulary(bert_vocab_path)
w2i_bio, i2w_bio = load_vocabulary(os.path.join(data_path, "vocab_bio.txt"))
w2i_attr, i2w_attr = load_vocabulary(os.path.join(data_path, "vocab_attr.txt"))


logger.info("building model...")

bert_config = bert_modeling.BertConfig.from_json_file(bert_config_path)
logger.info(bert_config.to_json_string())
        
model = MyModel(bert_config=bert_config, 
                vocab_size_bio=len(w2i_bio), 
                vocab_size_attr=len(w2i_attr), 
                O_tag_index=w2i_bio["O"],
                use_crf=True)

logger.info("model params:")
params_num_all = 0
for variable in tf.trainable_variables():
    params_num = 1
    for dim in variable.shape:
        params_num *= dim
    params_num_all += params_num
    logger.info("\t {} {} {}".format(variable.name, variable.shape, params_num))
logger.info("all params num: " + str(params_num_all))
        
logger.info("loading bert pretrained parameters...")
tvars = tf.trainable_variables()
(assignment_map, initialized_variable_names) = bert_modeling.get_assignment_map_from_checkpoint(tvars, bert_ckpt_path)
tf.train.init_from_checkpoint(bert_ckpt_path, assignment_map)

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True

# 全局session
sess = tf.Session(config=tf_config)
sess.run(tf.global_variables_initializer())

# 模型推理
def inference(original_text):
    # 准备数据
    logger.info("prepare data...")

    # 准备输入数据格式
    input_char_list, output_bio_list, output_attr_list, max_len = prepare_data(original_text)

    data_processor_valid = DataProcessor(
        input_char_list,
        output_bio_list,
        output_attr_list,
        w2i_char,
        w2i_bio, 
        w2i_attr, 
        shuffling=False
    )

    logger.info("start inference...")
        
    def valid(data_processor, max_batches=None, batch_size=1024):
        preds_kvpair = []
        golds_kvpair = []
        batches_sample = 0
        
        while True:
            (inputs_seq_batch, 
             inputs_mask_batch,
             inputs_segment_batch,
             _,
             _) = data_processor.get_batch(batch_size)

            feed_dict = {
                model.inputs_seq: inputs_seq_batch,
                model.inputs_mask: inputs_mask_batch,
                model.inputs_segment: inputs_segment_batch
            }
            
            start_time = datetime.now()
            preds_seq_bio_batch, preds_seq_attr_batch = sess.run(model.outputs, feed_dict)
            logger.info('[Time taken: {!s}]'.format(datetime.now() - start_time))
            
            for pred_seq_bio, pred_seq_attr, input_seq, mask in zip(preds_seq_bio_batch,
                                                                     preds_seq_attr_batch,
                                                                     inputs_seq_batch,
                                                                     inputs_mask_batch):
                l = sum(mask) - 2
                pred_seq_bio = [i2w_bio[i] for i in pred_seq_bio[1:-1][:l]]
                char_seq = [i2w_char[i] for i in input_seq[1:-1][:l]]
                pred_seq_attr = [i2w_attr[i] for i in pred_seq_attr[1:-1][:l]]

                pred_kvpair = extract_kvpairs_in_bioes(pred_seq_bio, char_seq, pred_seq_attr, True)
                
                preds_kvpair.append(pred_kvpair)
                
            if data_processor.end_flag:
                data_processor.refresh()
                break
            
            batches_sample += 1
            if (max_batches is not None) and (batches_sample >= max_batches):
                break
        
        #print(preds_kvpair)

        logger.info("Valid Samples: {}".format(len(preds_kvpair)))
        
        return preds_kvpair

    logger.info("Max length = %d"%max_len)
    #print(input_char_list)
    #print(output_bio_list)
    #print(output_attr_list)

    preds_kvpair = valid(data_processor_valid, max_batches=10)

    # 合并，生成最终结果（处理start_pos的在全文中的位置）
    entities = []
    pos_offset = 0
    for i in range(len(preds_kvpair)):
        #print(input_char_list[i].replace(' ',''))
        #print(preds_kvpair[i])
        for j in preds_kvpair[i]:
            entities.append({
                'label' : j[0],
                'value' : j[1],
                'start_pos' : j[2]+pos_offset,
            })
        pos_offset += len(input_char_list[i].replace(' ',''))

    # 按起始位置排序
    entities = sorted(entities, key=lambda x: x['start_pos'])

    return entities


if __name__ == '__main__':
    # 原始输入：不限制长度，会被自动分段，以“，”和“。”分割，建议间隔不要大于100字。
    original_text = ",2009年12月底出现黑便,,于当地行胃镜检查并行病理检查示:叒胃体中下部溃疡,叒病理示中分化腺癌,叒无腹胀、泛酸、嗳气、恶心、呕吐、叒无头晕、叒心悸、乏力等症,叒2010年1月13日于我院胃胰科行胃癌根治术,叒2010年1月18日,我院病理:切缘未见癌,叒胃体可见3x2x1cm3溃疡型肿物,叒镜上为中分化腺癌侵及胃壁全层至浆膜层,网膜未见癌,叒肝总动脉旁(0/1)、叒胃大弯(0/1)淋巴结未见癌,叒贲门左(3/3)、叒胃小弯(8/9)、幽门上(2/2)淋巴结可见腺癌转移,,免疫组化:cea(+)、叒p53(+)、叒pr(-)、叒er-b(+)、叒er(+++)、叒共计,ln:叒13/16转移,叒术后于2010年2月-2010年8月行术后化疗6程,叒具体用药为艾素100mg叒静点+叒希罗达1500mg叒bid叒po,2014年6月初出现右侧下上肢活动受限,叒7月份症状逐渐加重,叒7月10日就诊于*****,叒,行mri检查提示:胃癌术后多发脑转移,叒行甘露醇及地塞米松、叒洛赛克治疗后效果不佳。遂于我院就诊,2014-8-5行奥沙利铂150mg叒d1+叒替吉奥叒50mg叒bid叒d1-14化疗一程,2014-08-18开始行三维适形全脑放疗,剂量30gy/10f。2014-09-19始行替吉奥叒50mg叒bid叒d1-14单药化疗一程。本次为行上一程化疗收入我科,叒我科以“胃癌术后脑转移叒rtxnxm1叒iv期”收入,叒入科以来,叒精神饮食尚可,叒无恶心、叒呕吐,二便正常,体重无明显减低。"

    entities = inference(original_text)

    print(len(original_text), original_text)
    print(entities)

    # 核对结果是否与原始文本对应
    for i in entities:
        ori_value = original_text[i['start_pos']:i['start_pos']+len(i['value'])]
        if ori_value != i['value']:
            print("?", i, ori_value)
