import numpy as np

train = np.load('train_joint.npy')
label = np.load('train_label.npy')
all_0 = 0
for i in range(train.shape[0]):
    valid_frame_num = np.sum(test_b[i].sum(0).sum(-1).sum(-1) != 0)
    if valid_frame_num == 0:
        print(i)
        all_0 = i
        
print(train.shape)
mask = np.ones(train.shape[0], dtype=bool)
mask[all_0] = False
new_arr = train[mask]
new_label = label[mask]

np.save('train_joint_new.npy', new_arr)
np.save('train_label_new.npy', new_label)