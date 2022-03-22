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

def load_data_raw(filename):
    """加载数据
    单条格式：文本
    """
    max_len = 0    
    max_cnt = 0
    for l in open(filename):
        l = l.strip()
        if len(l)>128:
            print(len(l), l[:10])
            max_cnt += 1
        if len(l)>max_len:
            max_len = len(l)
    return max_len, max_cnt


if __name__ == '__main__':
    print('train:', load_data('./data/train.json'))
    print('dev:', load_data('./data/dev.json'))
    print('test', load_data_raw('../data/preliminary_test_a/sample_per_line_preliminary_A.txt'))
