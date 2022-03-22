import json


def load_data(filename, text_name='text'):
    """加载数据
    单条格式：(文本, 标签id)
    """
    max_len = 0    
    max_cnt = 0
    for l in json.load(open(filename)):
        if len(l[text_name])>128:
            print(len(l[text_name]), l[text_name][:10])
            max_cnt += 1
        if len(l[text_name])>max_len:
        	max_len = len(l[text_name])
    return max_len, max_cnt

if __name__ == '__main__':
	print(load_data('./data/train.json'))
	print(load_data('./data/dev.json'))