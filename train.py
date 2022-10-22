import torch
import utils
import matplotlib.pyplot as plt
import os
from option import Option
from dataset import SDataSet
from model import MLPDiffusion
from ema import EMA


if __name__ == "__main__":
    opt = Option().load_opt()
    dataset = SDataSet(opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    model = MLPDiffusion(opt).to(opt.gpu_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    if opt.use_ema:  # 参数平滑器
        ema = EMA(0.5)
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)

    print("start training ...")
    fig, axs = plt.subplots(1, 10, figsize=(28, 3))  # 保存中间的训练结果图示
    for epoch in range(opt.num_epoch):
        for idx, batch_x in enumerate(dataloader):
            # 对一个batchsize样本生成随机时刻t，尽量覆盖不同的t
            # 随机生成batchsize//2的时刻，剩下的时刻用其对立面的时刻n_steps-1-t
            t = torch.randint(0, opt.num_steps, size=(opt.batch_size // 2,))
            t = torch.cat([t, opt.num_steps - 1 - t], dim=0).to(opt.gpu_id)  # [batch_size, ]
            # 采样q(x_t|x_0)获得x_t
            eps = torch.randn_like(batch_x)
            alphas_overline_t = opt.alphas_overline_sqrt[t].unsqueeze(-1).to(opt.gpu_id)  # [batchSize, 1]
            one_minus_alphas_overline_sqrt_t = opt.one_minus_alphas_overline_sqrt[t].unsqueeze(-1).to(opt.gpu_id)  # [batchSize, 1]
            x_t = batch_x * alphas_overline_t + eps * one_minus_alphas_overline_sqrt_t
            # 网络预测 \epsilon_\theta
            eps_theta = model(x_t, t)
            # 目标函数
            loss = (eps - eps_theta).square().mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            if opt.use_ema:  # 参数平滑器
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param.data = ema(name, param.data)

        # 保存中间训练信息
        if epoch % opt.fig_save_freq == 0:
            print(loss)
            x_seq = utils.p_sample_loop(model, dataset.dataset.shape, opt)

            for i in range(1, 11):
                cur_x = x_seq[i * 10].detach().cpu().numpy()
                axs[i - 1].scatter(cur_x[:, 0], cur_x[:, 1], color="red", edgecolor="white")
                axs[i - 1].set_axis_off()
                axs[i - 1].set_title("$p_\\theta(\mathbf{x}_{%s})$" % str(opt.num_steps - i * 10))

            checkpoint_dir = os.path.join(opt.checkpoints_dir, opt.name)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            fig_name = os.path.join(checkpoint_dir, f"{epoch}.jpg")
            plt.savefig(fig_name)
            # 清除当前figure下subplots中所有axs
            plt.cla()
