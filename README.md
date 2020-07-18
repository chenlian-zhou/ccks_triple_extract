## 利用ALBERT+LSTM模型实现序列标注算法

### 数据集

1. 人民日报语料集，实体为人名、地名、组织机构名，数据集位于data/example.*;

2. 笔者自己标注的时间数据集，实体为事件，数据集位于data/time.*

### 模型

&emsp;&emsp;模型结构图如下：

![](https://github.com/percent4/ALBERT_BER_KERAS/blob/master/albert_bi_lstm.png)

&emsp;&emsp;训练效果如下图：

![](https://github.com/percent4/ALBERT_BER_KERAS/blob/master/example_loss_acc.png)

&emsp;&emsp;ALBERT+Bi-LSTM在人民日报测试集上的效果如下：

```
           precision    recall  f1-score   support

      ORG     0.9001    0.9112    0.9056      2185
      LOC     0.9383    0.8898    0.9134      3658
      PER     0.9543    0.9415    0.9479      1864

micro avg     0.9310    0.9084    0.9196      7707
macro avg     0.9313    0.9084    0.9195      7707
```

&emsp;&emsp;ALBERT+Bi-LSTM+CRF在人民日报测试集上的效果如下：

```
           precision    recall  f1-score   support

      ORG     0.7057    0.6902    0.6978      2185
      PER     0.8254    0.7988    0.8119      1864
      LOC     0.7361    0.5992    0.6606      3658

micro avg     0.7500    0.6733    0.7096      7707
macro avg     0.7490    0.6733    0.7078      7707
```
&emsp;&emsp;ALBERT+Bi-LSTM+CRF在CLUENER的dev数据集上的效果如下：

```
              precision    recall  f1-score   support

        book     0.9343    0.8421    0.8858       152
    position     0.9549    0.8965    0.9248       425
  government     0.9372    0.9180    0.9275       244
        game     0.6968    0.6725    0.6844       287
organization     0.8836    0.8605    0.8719       344
     company     0.8659    0.7760    0.8184       366
     address     0.8394    0.8187    0.8289       364
       movie     0.9217    0.7067    0.8000       150
        name     0.8771    0.8071    0.8406       451
       scene     0.9939    0.8191    0.8981       199

   micro avg     0.8817    0.8172    0.8482      2982
   macro avg     0.8835    0.8172    0.8482      2982
```

### 预测

&emsp;&emsp;人民日报NER的模型预测例子：

```
Please enter an sentence: 昨天进行的女单半决赛中，陈梦4-2击败了队友王曼昱，伊藤美诚则以4-0横扫了中国选手丁宁。
{'LOC': ['中国'], 'PER': ['陈梦', '王曼昱', '伊藤美诚', '丁宁']}
Please enter an sentence: 报道还提到，德国卫生部长延斯·施潘在会上也表示，如果不能率先开发出且使用疫苗，那么60%至70%的人可能会被感染新冠病毒。
{'ORG': ['德国卫生部'], 'PER': ['延斯·施潘']}
Please enter an sentence: “隔离结束回来，发现公司不见了”，网上的段子，真发生在了昆山达鑫电子有限公司员工身上。
{'ORG': ['昆山达鑫电子有限公司']}
Please enter an sentence: 真人版的《花木兰》由新西兰导演妮基·卡罗执导，由刘亦菲、甄子丹、郑佩佩、巩俐、李连杰等加盟，几乎是全亚洲整容。
{'LOC': ['新西兰', '亚洲'], 'PER': ['妮基·卡罗', '刘亦菲', '甄子丹', '郑佩佩', '巩俐', '李连杰']}
```

&emsp;&emsp;CLUENER的模型预测例子：

```
Please enter an sentence: 据中山外侨局消息，近日，秘鲁国会议员、祖籍中山市开发区的玛利亚·洪大女士在秘鲁国会大厦亲切会见了中山市人民政府副市长冯煜荣一行，对中山市友好代表团的来访表示热烈的欢迎。
{'address': ['中山市开发区', '秘鲁国会大厦'],
 'government': ['中山外侨局', '秘鲁国会', '中山市人民政府'],
 'name': ['玛利亚·洪大', '冯煜荣'],
 'position': ['议员', '副市长']}
 Please enter an sentence: “隔离结束回来，发现公司不见了”，网上的段子，真发生在了昆山达鑫电子有限公司员工身上。
{'company': ['昆山达鑫电子有限公司']}
Please enter an sentence: 由黄子韬、易烊千玺、胡冰卿、王子腾等一众青年演员主演的热血励志剧《热血同行》正在热播中。
{'game': ['《热血同行》'], 'name': ['黄子韬', '易烊千玺', '胡冰卿', '王子腾'], 'position': ['演员']}
Please enter an sentence: 近日，由作家出版社主办的韩作荣《天生我才——李白传》新书发布会在京举行
{'book': ['《天生我才——李白传》'], 'name': ['韩作荣'], 'organization': ['作家出版社']}

```