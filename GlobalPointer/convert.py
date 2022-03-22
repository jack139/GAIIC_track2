import os
import json

infile = '../data/train_data/train.txt'
outdir = './data'

dev_ratio = 0.1

'''
  {
    "text": "对儿童SARST细胞亚群的研究表明，与成人SARS相比，儿童细胞下降不明显，证明上述推测成立。",
    "entities": [
      {
        "start_idx": 3,
        "end_idx": 9,
        "type": "bod",
        "entity": "SARST细胞"
      },
      {
        "start_idx": 19,
        "end_idx": 24,
        "type": "dis",
        "entity": "成人SARS"
      }
    ]
  },

'''

D = []
text = ''
entities = []
all_idx = 0
start_idx = 0
entity = ''
etype = ''

for l in open(infile):
    c = l[0]
    label = l[2:-1] if l[-1]=='\n' else l[2:] # 去掉 \n
    text += c

    if len(l.strip())==0: # 一行text结束
        if entity!='':
            entities.append({
                "start_idx": start_idx,
                "end_idx": all_idx - 1,
                "type": etype,
                "entity": entity,
            })

        text = text.strip()

        # 检查
        for e in entities:
            if text[e['start_idx']:e['end_idx']+1]!=e['entity']:
                print('error:', text, e['start_idx'], e['end_idx'], e['entity'])

        # 加入数据集
        D.append({
            'text' : text,
            'entities' : entities,
        })
        text = ''
        entities = []
        all_idx = 0
        start_idx = 0
        entity = ''
        etype = ''

    else:
        if label[0]=='O':
            if entity!='':
                entities.append({
                    "start_idx": start_idx,
                    "end_idx": all_idx - 1,
                    "type": etype,
                    "entity": entity,
                })
            start_idx = 0
            entity = ''
            etype = ''
        elif label[0]=='B':
            if entity!='':
                entities.append({
                    "start_idx": start_idx,
                    "end_idx": all_idx - 1,
                    "type": etype,
                    "entity": entity,
                })                
            start_idx = all_idx
            entity = c
            etype = label.split('-')[1]
        elif label[0]=='I':
            entity += c
        else:
            print('unknown label: ', label)

        all_idx += 1

# 分割保存
total = len(D)
dev = int(total*dev_ratio)

json.dump(
    D[:-dev],
    open(os.path.join(outdir, 'train.json'), 'w', encoding='utf-8'),
    indent=4,
    ensure_ascii=False
)

json.dump(
    D[-dev:],
    open(os.path.join(outdir, 'dev.json'), 'w', encoding='utf-8'),
    indent=4,
    ensure_ascii=False
)

print(f'train: {total-dev}\tdev: {dev}')