import re

import numpy as np
import tensorflow as tf


# by 恶搞大王
class NodeLookup(object):
    # 将类别ID转换为人类易读的标签
    # 自己阅读之后
    # 大体就是讲imagenet_2012_challenge_label_map_proto.pbtxt和imagenet_synset_to_human_label_map.txt中的
    # target_class 和 target_class_string所对应的物体的名称对应起来
    # 里面用了很多的TensorFlow自带的函数.可能理解有差误
    # ****************模板代码***************
    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        # 这里输入文件的真实地址
        if not label_lookup_path:
            label_lookup_path = 'models/imagenet_2012_challenge_label_map_proto.pbtxt'
        if not uid_lookup_path:
            uid_lookup_path = 'models/imagenet_synset_to_human_label_map.txt'
        # 调用加载函数
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        # 判断文件是否存在
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)
        # 开始从imagenet_2012_challenge_label_map_proto.pbtxt读取数据
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        # 正则表达式
        p = re.compile(r'[n\d]*[ \S,]*')
        # 遍历
        for line in proto_as_ascii_lines:
            # findall会返回满足条件的以列表形式返回全部能匹配的子串
            parsed_items = p.findall(line)
            # [0]代表的是target_class_string值
            uid = parsed_items[0]
            # 代表id真实指向的物体
            human_string = parsed_items[2]
            # 存储数据
            uid_to_human[uid] = human_string
        node_id_to_uid = {}
        # 开始从imagenet_synset_to_human_label_map.txt读取数据
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            # 读取target_class的数据
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            # 读取target_class_string的数据
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]
        node_id_to_name = {}
        # 把获取的target_class 和 target_class_string指向的真实物体对应起来
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


# ****************模板代码***************

# 用于pb文件的读写
# 读取训练好的Inception-v3模型来创建graph
with tf.gfile.FastGFile('models/classify_image_graph_def.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# 实例化
node_lookup = NodeLookup()

# 得到session对象
sess = tf.Session()

# 返回给定名称的tensor
# Inception-v3模型的最后一层softmax的输出
softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')


def identification(image_data):
    # 输入图像数据，得到softmax概率值（一个shape=(1,1008)的向量）
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)
    # 取出前5个概率最大的值（top-10)
    top_k = predictions.argsort()[-10:][::-1]
    results = []
    for node_id in top_k:
        human_string = node_lookup.id_to_string(node_id)
        score = predictions[node_id]
        # 生成Json
        results.append({'label': human_string, 'score': '{:.2}'.format(score)})
    return results
