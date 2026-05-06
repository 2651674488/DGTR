import torch
import torch.nn as nn
import torch.nn.functional as F


# \subsection{动态周期感知检索（Dynamic Period-Aware Retrieval）}
# Code anchor: DynamicPeriodEstimator class and Model.forward period estimation block.
class DynamicPeriodEstimator(nn.Module):
    """Estimate sample-wise multi-period candidates and weights."""

    def __init__(
        self,
        spectrum_k,
        tau_min,
        tau_max,
        tau_init,
        spectrum_mode="energy_cum",
        spectrum_cum_ratio=0.9,
    ):
        super().__init__()
        self.spectrum_k = int(spectrum_k)
        self.spectrum_mode = str(spectrum_mode)
        self.spectrum_cum_ratio = float(spectrum_cum_ratio)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

        tau_ratio = (float(tau_init) - self.tau_min) / (self.tau_max - self.tau_min)
        self.raw_tau = nn.Parameter(torch.logit(torch.tensor(tau_ratio)))
        self.base_harmonic_logit = nn.Parameter(torch.tensor(0.0))  # base周期权重

    def forward(self, x_input):
        # 从输入序列中提取候选周期，并给每个周期分配权重。
        # x_input: (B, C, S)
        B, _, S = x_input.shape
        centered = x_input - x_input.mean(dim=-1, keepdim=True)  # 中心化
        spectrum = torch.fft.rfft(centered, dim=-1)  #光谱 傅里叶变换
        amplitude = spectrum.abs().mean(dim=1)  #振幅 (B, F)在通道维求平均，把多通道频谱合成为一个代表性频谱，用于后续统一的周期候选提取。
        amplitude[:, 0] = 0.0

        valid_bins = amplitude.shape[-1] - 1
        k_slots = min(self.spectrum_k, valid_bins)
        valid_amp = amplitude[:, 1:]
        if self.spectrum_mode == "amplitude_topk":
            topk_vals, rel_idx = torch.topk(valid_amp, k=k_slots, dim=-1)
            topk_idx = rel_idx + 1
        else:  # energy_cum: sort by |X|^2, mask later bins after cumulative-energy prefix
            # 按 |X|^2（能量）降序排列；用累计能量达 spectrum_cum_ratio 的前 n90
            # 个 bin 作为“有效”候选，其余 k_slots 槽位先验置零（张量仍保持 (B, k_slots)）。
            power = valid_amp * valid_amp
            total = power.sum(dim=-1, keepdim=True)
            sorted_p, rel_idx = torch.topk(power, k=valid_bins, dim=-1)
            frac = sorted_p.cumsum(dim=-1) / total
            m = self.spectrum_cum_ratio
            n90 = (frac < m).to(torch.long).sum(dim=-1) + 1
            # 取能量最高的 k_slots 个 bin 的顺序，与 n90 对齐做掩码
            rel_top = rel_idx[:, :k_slots]
            topk_idx = rel_top + 1
            topk_vals = torch.gather(valid_amp, 1, rel_top)
            col = torch.arange(k_slots, device=amplitude.device).view(1, -1).expand(B, -1)
            mask_90 = col < n90.view(B, 1)
            topk_vals = topk_vals * mask_90.to(topk_vals.dtype)
        periods = (float(S) / topk_idx.float()).clamp(self.tau_min, self.tau_max)

        tau_base = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.raw_tau)
        tau_base = tau_base.expand(B, 1)
        tau_all = torch.cat([tau_base, periods], dim=1)

        base_score = self.base_harmonic_logit.expand(B, 1)
        prior_scores = torch.cat([base_score, topk_vals], dim=1)
        tau_weights = F.softmax(prior_scores, dim=-1)
        # tau_all（候选周期）、tau_weights（对应权重）。用于后续的查询构建。
        return tau_all, tau_weights


class ChannelBranchMix(nn.Module):
    """Per-channel multi-branch Conv2d fused by sample-wise branch_weight."""

    def __init__(self, branch_kernels):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=(2, kernel),
                    stride=1,
                    padding=(0, kernel // 2),
                    padding_mode="zeros",
                    bias=False,
                )
                for kernel in branch_kernels
            ]
        )
        self.num_branches = len(self.convs)

    def forward(self, x, branch_weight):
        # x: (B, 1, 2, S), branch_weight: (B, num_branches)
        B = x.shape[0]
        branch_outs = [conv(x).squeeze(2) for conv in self.convs]
        stacked = torch.stack(branch_outs, dim=1)
        mixed = (stacked * branch_weight.view(B, self.num_branches, 1, 1)).sum(dim=1)
        return mixed

class SAPMixer(nn.Module):
    # \subsection{自适应多分支时序融合（Adaptive Multi-Branch Temporal Fusion）}
    # Code anchor: multi-branch Conv2d retrieval with branch_weight gating.

    def __init__(self, d_series, c, period_len=24, branch_kernels=None, agg=True):
        super(SAPMixer, self).__init__()
        self.agg = agg
        self.period_len = period_len
        self.c = c
        self.branch_kernels = self._normalize_kernels(branch_kernels)
        self.num_branches = len(self.branch_kernels)

        self.q_norm = nn.LayerNorm(d_series)
        self.linear = nn.Linear(d_series, d_series)
        self.ds_convs = nn.ModuleList(
            ChannelBranchMix(self.branch_kernels) for _ in range(self.c)
        )


    def _normalize_kernels(self, branch_kernels):
        kernels = branch_kernels or [1 + 2 * (self.period_len // 2)]
        norm_kernels = []
        for kernel in kernels:
            k = int(kernel)
            if k % 2 == 0:
                k += 1
            norm_kernels.append(k)
        return norm_kernels

    def forward(self, x, q, branch_weight):
        _, C, S = x.shape
        # Step 1: Mapping（q 在时间上逐通道 LayerNorm 再进 linear）
        global_query = self.linear(self.q_norm(q))

        # Step 2: GTA mode, aggregate along temporal length.
        # if self.agg:
        #     weight = F.softmax(global_query, dim=-1)  # normalize over length S
        #     global_query = torch.sum(global_query * weight, dim=-1, keepdim=True)  # (B, C, 1)
        #     global_query = global_query.repeat(1, 1, S)  # (B, C, S)

        # Step 3: Fuse
        out = torch.stack([x, global_query], dim=2)  # (B, C, 2, S)

        conv_outs = []
        for channel in range(self.c):
            channel_in = out[:, channel, :, :].unsqueeze(1)  # (B, 1, 2, S)
            # ChannelBranchMix: all branches are fused inside with branch_weight
            mixed = self.ds_convs[channel](channel_in, branch_weight)  # (B, 1, S)
            conv_outs.append(mixed)
        conv_out = torch.cat(conv_outs, dim=1)  # (B, C, S)
        return conv_out


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle

        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.spectrum_k = int(getattr(configs, 'sapmixer_spectrum_k', 4))
        self.spectrum_mode = str(getattr(configs, "sapmixer_spectrum_mode", "energy_cum"))
        self.spectrum_cum_ratio = float(getattr(configs, "sapmixer_spectrum_cum_ratio", 0.9))
        self.use_multiscale = bool(int(getattr(configs, 'sapmixer_use_multiscale', 1)))
        branch_kernels = self._parse_branch_kernels(getattr(configs, 'sapmixer_branch_kernels', '3,7,15,31'))

        self.tau_min = float(getattr(configs, 'learnable_tau_min', 2.0))
        self.tau_max = float(getattr(configs, 'learnable_tau_max', 512.0))
        tau_init = float(getattr(configs, 'learnable_tau_init', -1.0))
        if tau_init <= 0:
            tau_init = float(self.cycle_len)

        self.phase_proj = nn.Linear(2, self.enc_in, bias=True)
        nn.init.xavier_uniform_(self.phase_proj.weight, gain=0.1)
        nn.init.zeros_(self.phase_proj.bias)
        self.period_estimator = DynamicPeriodEstimator(
            spectrum_k=self.spectrum_k,
            tau_min=self.tau_min,
            tau_max=self.tau_max,
            tau_init=tau_init,
            spectrum_mode=self.spectrum_mode,
            spectrum_cum_ratio=self.spectrum_cum_ratio,
        )

        self.sap_mixer = SAPMixer(
            d_series=self.seq_len,
            c=self.enc_in,
            branch_kernels=branch_kernels,
            agg=False,
        )

        self.log_Tm = nn.Parameter(torch.tensor(0.0))  # 初始 Tm=1

        self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len),
        )

    def _parse_branch_kernels(self, raw_branch_kernels):
        if isinstance(raw_branch_kernels, str):
            kernels = [item.strip() for item in raw_branch_kernels.split(',') if item.strip()]
        else:
            kernels = [int(x) for x in raw_branch_kernels]
        return [int(kernel) for kernel in kernels]

    def _build_multiscale_query(self, tau_all, tau_weight, seq_len, device):
        # \subsection{多尺度相位查询构建（Multi-Scale Phase Query Construction）}
        # Code anchor: build sinusoidal phase features for all candidate periods,
        # then project and merge by tau_weight to form query_input.
        B = tau_all.shape[0]
        time_index = torch.arange(seq_len, device=device, dtype=torch.float32).view(1, -1).expand(B, -1)
        theta = (2.0 * torch.pi * time_index.unsqueeze(1)) / tau_all.unsqueeze(-1)  # (B, K+1, S)
        phase_feats = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)  # (B, K+1, S, 2)

        proj = self.phase_proj(phase_feats.reshape(-1, seq_len, 2))
        proj = proj.reshape(B, tau_all.shape[1], seq_len, self.enc_in).permute(0, 1, 3, 2)  # (B, K+1, C, S)
        query_input = (proj * tau_weight.view(B, tau_all.shape[1], 1, 1)).sum(dim=1)  # (B, C, S)
        return query_input

    def _build_branch_weight_from_periods(self, tau_all, tau_weight):
        # Period-kernel matching: score_j = sum_i w_i * exp(-|log(tau_i)-log(k_j)| / T)
        Tm = torch.exp(self.log_Tm)
        kernel_tensor = tau_all.new_tensor(self.sap_mixer.branch_kernels)
        log_tau = torch.log(tau_all).unsqueeze(-1)  # (B, K+1, 1)
        log_kernel = torch.log(kernel_tensor).view(1, 1, -1)  # (1, 1, M)
        dist = (log_tau - log_kernel).abs()  # (B, K+1, M)
        match = torch.exp(-dist / Tm)  # (B, K+1, M)
        score = (tau_weight.unsqueeze(-1) * match).sum(dim=1)  # (B, M)
        return F.softmax(score, dim=-1)

    def forward(self, x):
        # RevIN normalize
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        seq_std = seq_var.sqrt()
        x = (x - seq_mean) / seq_std

        # (B, S, C) -> (B, C, S)
        x_input = x.permute(0, 2, 1)

        # \subsection{动态周期感知检索（Dynamic Period-Aware Retrieval）}
        # Code anchor: estimate candidate periods and sample-wise period weights.
        tau_all, tau_weight = self.period_estimator(x_input)
        if not self.use_multiscale:
            tau_weight = tau_weight.new_zeros(tau_weight.shape)
            tau_weight[:, 0] = 1.0

        # \subsection{多尺度相位查询构建（Multi-Scale Phase Query Construction）}
        # Code anchor: convert multi-period phase features into query_input.
        query_input = self._build_multiscale_query(
            tau_all=tau_all,
            tau_weight=tau_weight,
            seq_len=x_input.shape[-1],
            device=x_input.device,
        )

        # \subsection{自适应多分支时序融合（Adaptive Multi-Branch Temporal Fusion）}
        # Code anchor: period-kernel matching gates branch fusion in SAPMixer.
        branch_weight = self._build_branch_weight_from_periods(tau_all, tau_weight)
        global_information = self.sap_mixer(x_input, query_input, branch_weight=branch_weight)

        # Projection + MLP
        input_proj = self.input_proj(x_input + global_information)
        hidden = self.model(input_proj)
        output = self.output_proj(hidden + input_proj).permute(0, 2, 1)

        # RevIN de-normalize
        output = output * seq_std + seq_mean
        return output
