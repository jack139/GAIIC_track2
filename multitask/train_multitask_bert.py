import logging
import tensorflow as tf
import numpy as np
import os

from ner.model_multitask_bert import MyModel
from ner.bert import modeling as bert_modeling
from ner.utils import DataProcessor_MTL_BERT as DataProcessor
from ner.utils import load_vocabulary
from ner.utils import extract_kvpairs_in_bioes
from ner.utils import cal_f1_score

data_path = "./data"

bert_vocab_path = "../../nlp_model/chinese_bert_L-12_H-768_A-12/vocab.txt"
bert_config_path = "../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_config.json"
bert_ckpt_path = "../../nlp_model/chinese_bert_L-12_H-768_A-12/bert_model.ckpt"
#bert_ckpt_path = "./ckpt/model.ckpt.batch4700_0.7603"  # 继续训练

# set logging
log_file_path = "./ckpt/run.log"
if os.path.exists(log_file_path): os.remove(log_file_path)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
fhlr = logging.FileHandler(log_file_path)
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)

logger.info("loading vocab...")

w2i_char, i2w_char = load_vocabulary(bert_vocab_path)
w2i_bio, i2w_bio = load_vocabulary(data_path+"/vocab_bio.txt")
w2i_attr, i2w_attr = load_vocabulary(data_path+"/vocab_attr.txt")

logger.info("loading data...")

data_processor_train = DataProcessor(
    data_path+"/train/input.seq.char",
    data_path+"/train/output.seq.bio",
    data_path+"/train/output.seq.attr",
    w2i_char,
    w2i_bio, 
    w2i_attr,
    shuffling=True
)

data_processor_valid = DataProcessor(
    data_path+"/dev/input.seq.char",
    data_path+"/dev/output.seq.bio",
    data_path+"/dev/output.seq.attr",
    w2i_char,
    w2i_bio, 
    w2i_attr, 
    shuffling=True
)

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

logger.info("start training...")

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True

with tf.Session(config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=50)
    
    epoches = 0
    losses = []
    batches = 0
    best_f1 = 0
    batch_size = 128

    while epoches < 30:
        (inputs_seq_batch, 
         inputs_mask_batch,
         inputs_segment_batch,
         outputs_seq_bio_batch,
         outputs_seq_attr_batch) = data_processor_train.get_batch(batch_size)
        
        feed_dict = {
            model.inputs_seq: inputs_seq_batch,
            model.inputs_mask: inputs_mask_batch,
            model.inputs_segment: inputs_segment_batch,
            model.outputs_seq_bio: outputs_seq_bio_batch,
            model.outputs_seq_attr: outputs_seq_attr_batch
        }
        
        if batches == 0: 
            logger.info("###### shape of a batch #######")
            logger.info("inputs_seq: " + str(inputs_seq_batch.shape))
            logger.info("inputs_mask: " + str(inputs_mask_batch.shape))
            logger.info("inputs_segment: " + str(inputs_segment_batch.shape))
            logger.info("outputs_seq_bio: " + str(outputs_seq_bio_batch.shape))
            logger.info("outputs_seq_attr: " + str(outputs_seq_attr_batch.shape))
            logger.info("###### preview a sample #######")
            logger.info("input_seq:" + " ".join([i2w_char[i] for i in inputs_seq_batch[0]]))
            logger.info("input_mask :" + " ".join([str(i) for i in inputs_mask_batch[0]]))
            logger.info("input_segment :" + " ".join([str(i) for i in inputs_segment_batch[0]]))
            logger.info("output_seq_bio: " + " ".join([i2w_bio[i] for i in outputs_seq_bio_batch[0]]))
            logger.info("output_seq_attr: " + " ".join([i2w_attr[i] for i in outputs_seq_attr_batch[0]]))
            logger.info("###############################")
        
        loss, _ = sess.run([model.loss, model.train_op], feed_dict)
        losses.append(loss)
        batches += 1
        
        if data_processor_train.end_flag:
            data_processor_train.refresh()
            epoches += 1

        def valid(data_processor, max_batches=None, batch_size=1024):
            preds_kvpair = []
            golds_kvpair = []
            batches_sample = 0
            
            while True:
                (inputs_seq_batch, 
                 inputs_mask_batch,
                 inputs_segment_batch,
                 outputs_seq_bio_batch,
                 outputs_seq_attr_batch) = data_processor.get_batch(batch_size)

                feed_dict = {
                    model.inputs_seq: inputs_seq_batch,
                    model.inputs_mask: inputs_mask_batch,
                    model.inputs_segment: inputs_segment_batch
                }
                
                preds_seq_bio_batch, preds_seq_attr_batch = sess.run(model.outputs, feed_dict)
                
                for pred_seq_bio, gold_seq_bio, pred_seq_attr, gold_seq_attr, input_seq, mask in zip(preds_seq_bio_batch,
                                                                                                     outputs_seq_bio_batch,
                                                                                                     preds_seq_attr_batch,
                                                                                                     outputs_seq_attr_batch,
                                                                                                     inputs_seq_batch,
                                                                                                     inputs_mask_batch):
                    l = sum(mask) - 2
                    pred_seq_bio = [i2w_bio[i] for i in pred_seq_bio[1:-1][:l]]
                    gold_seq_bio = [i2w_bio[i] for i in gold_seq_bio[1:-1][:l]]
                    char_seq = [i2w_char[i] for i in input_seq[1:-1][:l]]
                    pred_seq_attr = [i2w_attr[i] for i in pred_seq_attr[1:-1][:l]]
                    gold_seq_attr = [i2w_attr[i] for i in gold_seq_attr[1:-1][:l]]
                    
                    pred_kvpair = extract_kvpairs_in_bioes(pred_seq_bio, char_seq, pred_seq_attr)
                    gold_kvpair = extract_kvpairs_in_bioes(gold_seq_bio, char_seq, gold_seq_attr)
                    
                    preds_kvpair.append(pred_kvpair)
                    golds_kvpair.append(gold_kvpair)
                    
                if data_processor.end_flag:
                    data_processor.refresh()
                    break
                
                batches_sample += 1
                if (max_batches is not None) and (batches_sample >= max_batches):
                    break
            
            p, r, f1 = cal_f1_score(preds_kvpair, golds_kvpair)

            logger.info("Valid Samples: {}".format(len(preds_kvpair)))
            logger.info("Valid P/R/F1: {} / {} / {}".format(round(p*100, 2), round(r*100, 2), round(f1*100, 2)))
            
            return (p, r, f1)
            
        if batches % 500 == 0:
            logger.info("")
            logger.info("Epoches: {}".format(epoches))
            logger.info("Batches: {}".format(batches))
            logger.info("Loss: {}".format(sum(losses) / len(losses)))
            losses = []
            
            p, r, f1 = valid(data_processor_valid, max_batches=10)
            if f1 > best_f1:
                best_f1 = f1
                logger.info("############# best performance now here ###############")

                ckpt_save_path = "./ckpt/model.ckpt.batch{}_{:.4f}".format(batches, f1)
                logger.info("Path of ckpt: {}".format(ckpt_save_path))
                saver.save(sess, ckpt_save_path)
            
            