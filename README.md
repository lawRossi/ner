# 中文命名实体识别工具包

## 模型

实现了双向lstm+crf、bert、bert+crf模型。

## 特性

### 支持词典特征

可以使用实体词典来提升模型的效果。

### 数据处理工具

包含数据自动标注、数据增强、多种标注方式互相转换等数据处理功能。

## 安装
    
    python3 setup.py install
    pip3 install -r requirements.txt

## 模型训练
    修改run_bilstm_crf.sh和run_bert.sh的相关参数，然后执行脚本即可。

# 使用模型

```python
from ner.bert_ner.bert_ner import BertNERTagger
from ner.bilstm_crf.bilstm_crf_ner import BilstmCrfNerTagger

tagger = BertNERTagger.load_model("model/to/bert")   # 指定模型路径加载Bert命名实体识别模型
print(tagger.recognize_nes("输入句子"))           # 识别实体

tagger = BilstmCrfNerTagger.load_model("model/to/bilstm_crf")  # 指定模型路径加载Bilstm-crf命名实体识别模型
print(tagger.recognize_nes("输入句子"))           # 识别实体
```
