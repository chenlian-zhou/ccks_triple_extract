# -*- coding: utf-8 -*-

# 数据相关的配置
event_type = "ccks2019"

train_file_path = "./data/%s.train" % event_type
dev_file_path = "./data/%s.test" % event_type
test_file_path = "./data/%s.test" % event_type

# 模型相关的配置
MAX_SEQ_LEN = 128   # 输入的文本最大长度
