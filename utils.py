import torch
import matplotlib.pyplot as plt
from option import Option
from dataset import SDataSet


def q_x(x_0, t, opt):
    """基于x_0，t得到任意时刻的x_t"""
    eps = torch.randn_like(x_0)
    if isinstance(t, int):  # 说明当前是函数show_diffusion调用的，data[10000, 2]，t -> [10000, ]
        t = torch.ones([x_0.shape[0], ], dtype=torch.long) * t
    alphas_overline_t = opt.alphas_overline_sqrt[t].unsqueeze(-1).to(eps.device)  # [10000, 1]
    one_minus_alphas_overline_sqrt_t = opt.one_minus_alphas_overline_sqrt[t].unsqueeze(-1).to(eps.device)  # [10000, 1]

    return alphas_overline_t * x_0 + one_minus_alphas_overline_sqrt_t * eps


def show_diffusion(dataset, opt, num_shows=20):
    """可视化扩散过程"""
    data = dataset.dataset
    data = torch.FloatTensor(data)
    _, axs = plt.subplots(2, 10, figsize=(20, 3))

    for i in range(num_shows):
        j = i // 10
        k = i % 10
        q_i = q_x(data, i * opt.num_steps // num_shows, opt)
        axs[j, k].scatter(q_i[:, 0], q_i[:, 1], color="red", edgecolor="white")
        axs[j, k].set_axis_off()
        axs[j, k].set_title("$q(\mathbf{x}_{%d})$" % (i * opt.num_steps // num_shows))

    plt.show()


def p_sample(model, x_t, t, opt):
    """从p_\theta(x_{t-1}|x_t)采样"""
    eps_theta = model(x_t, t)

    coeff = (opt.betas[t] / opt.one_minus_alphas_overline_sqrt[t]).to(opt.gpu_id)
    mu_theta = (1 / (1-opt.betas[t]).sqrt()).to(opt.gpu_id) * (x_t - coeff * eps_theta)
    sigma_t = opt.betas[t].sqrt().to(opt.gpu_id)

    eps = torch.randn_like(x_t)
    sample = mu_theta + sigma_t * eps

    return sample  # gpu


def p_sample_loop(model, shape, opt):
    """从x_T恢复x_{t-1}, x_{t-2}, ...,x_0"""
    cur_x = torch.randn(shape).to(opt.gpu_id)
    x_seq = [cur_x]
    for t in reversed(range(opt.num_steps)):
        t = torch.LongTensor([t]).to(opt.gpu_id)
        cur_x = p_sample(model, cur_x, t, opt)  # gpu
        x_seq.append(cur_x)

    return x_seq


if __name__ == "__main__":
    opt = Option().load_opt()
    dataset = SDataSet(opt)
    show_diffusion(dataset, opt)