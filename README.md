### UAV

# config 文件中数据为train_joint_new.npy/train_label_new.npy/test_joint_B_new.npy可使用train_new.py脚本对原始数据train_joint.npy/train_label.npy/test_joint_B.npy进行处理

# 对于infogcn日志文件中测试结果偏高的解释为测试集也为train_joint_new.npy/train_label_new.npy， 并未使用测试集A， 具体可见UAV-SAR/infogcn/main.py文件84-130行
# 对于74.690结果偏高，只有infogcn采用上述策略，其他模型均使用训练集训练，测试集A验证，具体可见各个模型config文件
