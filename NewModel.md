# 简单说明
1.  sampling_function负责采样数据，并生成label(程序中需要的label), CornerEdgeNet的实现为sample/corneredgenet.py
2.  models/CornerEdgeNet.py为主模型,模型内包含loss函数和模型
3.  models/CornerEdgeNet.py:_decode函数用于解码输出，生成预测结果（只在测试时使用，暂时没有使用)
4.  损失函数的实现位于models/_py_utils/losses.py, CornerEdgeNet_Loss为对应实现
5.  Model的生成根据配置文件动态生成，nnet/py_factory.py中为模型包装的实现，真实的模型实现现在models中
6.  CornerEdgeNet的配置文件还没有实现
# 训练过程
train -> sampling_function生成batch数据 -> py_factory(Network类的forward) -> models/CornerEdgeNet.py 中的模型实现和loss函数 -> backward
