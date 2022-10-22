import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_s_curve
from option import Option


class SDataSet(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        s_curve, _ = make_s_curve(self.opt.num_sample, noise=0.1)  # 返回的是3D坐标
        s_curve = s_curve[:, [0, 2]] / 10.0
        self.dataset = s_curve
        print("shape of s:", self.dataset.shape)

    def __getitem__(self, item):
        data = self.dataset[item]
        data = torch.FloatTensor(data)
        data = data.to(self.opt.gpu_id)

        return data

    def __len__(self):
        return self.dataset.shape[0]

    def show_dataset(self):
        data = self.dataset.T
        plt.figure()  # 定义画纸
        plt.scatter(*data, color="red", edgecolor="white")  # 绘制
        plt.title("$q(\mathbf{x}_0)$")  # 标题
        plt.show()  # 显示


if __name__ == "__main__":
    opt = Option().load_opt()
    dataset = SDataSet(opt)
    dataset.show_dataset()  # 可视化