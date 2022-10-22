import argparse
import torch


class Option:
    def __init__(self):
        parser = argparse.ArgumentParser(description="hyperparam for toy diffusion model")
        parser.add_argument("--name", type=str, default="exp", help="本次训练的名称")
        parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
        parser.add_argument("--num_sample", type=int, default=10**4, help="训练样本的数量")
        parser.add_argument("--gpu_id", type=int, default=0, help="单GPU就可以了")
        parser.add_argument("--num_steps", type=int, default=100, help="扩散过程总步数")
        parser.add_argument("--num_uints", type=int, default=128, help="网络隐层的维数")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_epoch", type=int, default=4000)
        parser.add_argument("--use_ema", type=bool, default=False, help="是否使用参数平滑器")
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--fig_save_freq", type=int, default=100, help="每隔多少代保存一次当前模型预测结果")

        self.parser = parser.parse_args()
        self.add_parser()  # 增加其他需要的超参数，并进行一些必要的步骤

    def add_parser(self):
        # 检查GPU是否可用
        assert self.parser.gpu_id >= 0 and torch.cuda.is_available(), "GPU不可用"
        # 其他一些超参数
        betas = torch.linspace(-6, 6, self.parser.num_steps)
        betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
        alphas = 1 - betas
        alphas_overline = torch.cumprod(alphas, 0)
        alphas_overline_sqrt = torch.sqrt(alphas_overline)
        one_minus_alphas_overline_sqrt = torch.sqrt(1 - alphas_overline)
        # 检查尺寸
        assert betas.shape == alphas.shape == alphas_overline.shape \
               == alphas_overline_sqrt.shape == one_minus_alphas_overline_sqrt.shape
        print("all the same shape:", betas.shape)

        self.parser.betas = betas
        self.parser.alphas_overline_sqrt = alphas_overline_sqrt
        self.parser.one_minus_alphas_overline_sqrt = one_minus_alphas_overline_sqrt

    def load_opt(self):
        return self.parser