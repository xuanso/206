import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import optuna
from tqdm import tqdm  # 导入 tqdm

if __name__ == '__main__':
    np.random.seed(1028)
    # # test_B
    #
    # alpha = [4.0, 1.5, 1.0, 0.3, 1.0, 1.0]
    #
    # score_dir = './test_dir'
    # scores = []
    #
    # for mod in os.listdir(score_dir):
    #     pkl_path = os.path.join(score_dir, mod, 'epoch1_test_score.pkl')
    #     with open(pkl_path, 'rb') as f:
    #         a = list(pickle.load(f).items())
    #         b = []
    #         for i in a:
    #             b.append(i[1])
    #         scores.append(np.array(b))
    # scores = np.array(scores)
    #
    # pred_scores = np.zeros([4598, 155])
    # for i, _ in enumerate(alpha):
    #     pred_scores += alpha[i] * scores[i]
    #
    # all_zero = np.zeros(155)
    # all_zero[97] = 1
    # final_scores = np.insert(pred_scores, 3222, all_zero, axis=0)
    # print(final_scores.shape)
    # np.save('pred.npy', final_scores)

    # test_A

    # alpha = [4.0, 1.5, 1.0, 0.3, 1.0, 1.0]

    score_dir = './test_A_dir'
    scores = []

    for mod in os.listdir(score_dir):
        npy_path = os.path.join(score_dir, mod, 'epoch1_test_score.npy')
        pkl_path = os.path.join(score_dir, mod, 'epoch1_test_score.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                a = list(pickle.load(f).items())
                b = [i[1] for i in a]
                scores.append(np.array(b))
        elif os.path.exists(npy_path):  # 如果没有 .pkl，则检查 .npy
            scores.append(np.load(npy_path))
        else:
            print(f"Neither {pkl_path} nor {npy_path} exists for model {mod}.")
    scores = np.array(scores)

    #随机搜索

    # best_acc = 0
    # best_params = None
    # num_iterations = 500  # 随机搜索次数
    # label = np.load('./data/uav/province/test_A_label.npy')
    #
    # for _ in tqdm(range(num_iterations), desc="Random Searching"):
    #     # 生成 15 个在 [0, 2) 范围内的随机数，并将其转化为 0.1 的倍数
    #     alpha = np.random.randint(0, 20, size=16) * 0.1  # 生成 0.1 的倍数
    #
    #     # 计算加权得分
    #     pred_scores = np.zeros([2000, 155])
    #     for i in range(len(alpha)):
    #         pred_scores += alpha[i] * scores[i]
    #
    #     pred = pred_scores.argmax(axis=-1)
    #     label = np.load('./data/uav/province/test_A_label.npy')
    #     acc = accuracy_score(label, pred)
    #
    #     # 更新最佳参数和准确率
    #     if acc > best_acc:
    #         best_acc = acc
    #         best_params = alpha
    #
    # print('Best accuracy:', best_acc)
    # print('Best parameters:', best_params)

    #贝叶斯搜索
    n_trials = 300  # 试验次数

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    label = np.load('./data/uav/province/test_A_label.npy')

    def objective(trial):
        # 扩大 alpha 的搜索范围
        alpha = [trial.suggest_float(i, 0, 2) for i in range(17)]

        pred_scores = np.zeros([2000, 155])
        for i in range(len(alpha)):
            pred_scores += alpha[i] * scores[i]

        pred = pred_scores.argmax(axis=-1)
        return accuracy_score(label, pred)


    study = optuna.create_study(direction='maximize')

    # 使用 tqdm 添加进度条
    with tqdm(total=n_trials, desc="Bayesian Optimization") as pbar:
        for _ in range(n_trials):
            study.optimize(objective, n_trials=1)
            pbar.update(1)  # 更新进度条
    print('Best accuracy:', study.best_value)
