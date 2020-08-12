# bert_regression

工作中遇到一个用文本做回归的问题，原版bert本来有run_reg.py，但不知为什么后来就又移除了，这个版本的y是从0到1，我的需求是从0到几十万，所以需要改改。

在bert基础上，有几点修改：

1、extract_features.py中的read_examples函数读文件，改成自己的输入文件的格式，即“数值y\t文本x”的形式。

2、modeling.py文件的227行改成，用倒数第二层。

3、run_reg_v2.3.1.py在run_reg.py的基础上，做了如下修改：

   （1）去掉softmax：让y的范围从0到1变成从0到无穷大。
    
   （2）再加两个relu和全连接：增加模型的拟合能力。
    
   （3）用Xavier给w做初始化：更快、更好的效果。
    
run_reg.py来自：https://github.com/google-research/bert/pull/503/commits/f005e159ffb40591b7e16d257ab4abc4e137182a
