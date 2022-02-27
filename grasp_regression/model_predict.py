
import torch
import numpy


def read_data(name):
    # 这里是你读取需要速测的数据的代码

    return 0

the_model = torch.load('model.pth')	#这里输入模型的名称，这句话使用的前提是在训练阶段，你使用 torch。save('model.pth')去存储你的模型

data = read_data('data.txt')

predict = the_model(data)


#很简单，就三句话，第一句load模型，第二句load数据，第三句获得预测





