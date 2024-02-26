# 当代人工智能实验四：文本摘要
<hr/>

## 介绍
<hr/>
本实验实践了一个基于 **Encoder-Decoder 架构的序列到序列（seq2seq）模型**，进行了一个文本摘要的任务。


## 环境配置
<hr/>

- 本项目所使用的环境为python3.9
- 为了更快地安装本实验所需要一些库，可以执行下列代码换源后安装
- 安装requirements.txt

```
pip install --upgrade pip 
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
```

## 运行方式
<hr/>

由于Seq2seq模型训练时间似乎较长，本实验保存了训练好的模型，可以直接进行测试，也可以进行训练。

### 训练模型：
利用argparse包提供了一些可以调整的参数：
- --model， default='Seq2SeqBART'
- --lr， default=0.05
- --dropout， default=0.0
- --epoch， default=1
- --batch_size，  default=16


以下是两个代码训练运行的示例（需要将main中注释掉的train取消注释）：
```
python main.py --model Seq2SeqLSTM --lr 0.1
python main.py --model Seq2SeqBART --lr 0.05
```
直接运行main.py会加载保存下来的模型开始测试：
```
python main.py
```
