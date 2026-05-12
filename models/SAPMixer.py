import torch
import torch.nn as nn
import torch.nn.functional as F


# \subsection{动态周期感知检索（Dynamic Period-Aware Retrieval）}
# Code anchor: DynamicPeriodEstimator class and Model.forward period estimation block.
class DynamicPeriodEstimator(nn.Module):

    def __init__(
        self,
        spectrum_k,
        tau_min,
        tau_max,
        spectrum_cum_ratio=0.9,
    ):
        super().__init__()
        self.spectrum_k = int(spectrum_k)
        self.spectrum_cum_ratio = float(spectrum_cum_ratio)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

    def forward(self, x_input):
        # 从输入序列中提取候选周期，并给每个周期分配权重（每变量独立 FFT 谱）。
        # x_input: (B, C, S)
        # 返回 tau_all (B, C, K), tau_weights (B, C, K)；F = S//2+1，valid_bins = F-1，K = k_slots
        B, C, S = x_input.shape
        centered = x_input - x_input.mean(dim=-1, keepdim=True)  # (B, C, S)，中心化
        spectrum = torch.fft.rfft(centered, dim=-1)  # (B, C, F)，F = rfft 频率 bins
        amplitude = spectrum.abs()  # (B, C, F)
        amplitude[:, :, 0] = 0.0

        valid_bins = amplitude.shape[-1] - 1
        k_slots = min(self.spectrum_k, valid_bins)
        valid_amp = amplitude[:, :, 1:]  # (B, C, valid_bins)
        # 按 |X|^2（能量）降序；累计能量达 spectrum_cum_ratio 的前 n90 个 bin 为有效候选
        power = valid_amp * valid_amp  # (B, C, valid_bins)
        total = power.sum(dim=-1, keepdim=True)  # (B, C, 1)
        sorted_p, rel_idx = torch.topk(power, k=valid_bins, dim=-1)  # 均为 (B, C, valid_bins)
        frac = sorted_p.cumsum(dim=-1) / total  # (B, C, valid_bins)
        m = self.spectrum_cum_ratio
        n90 = (frac < m).to(torch.long).sum(dim=-1) + 1  # (B, C)
        rel_top = rel_idx[:, :, :k_slots]  # (B, C, k_slots)
        topk_idx = rel_top + 1  # (B, C, k_slots)，物理 bin 索引（不含 DC）
        topk_vals = torch.gather(valid_amp, 2, rel_top)  # (B, C, k_slots)
        col = torch.arange(k_slots, device=amplitude.device).view(1, 1, -1).expand(B, C, -1)  # (B, C, k_slots)
        mask_90 = col < n90.view(B, C, 1)  # (B, C, k_slots)
        topk_vals = topk_vals * mask_90.to(topk_vals.dtype)
        periods = (float(S) / topk_idx.float()).clamp(self.tau_min, self.tau_max)  # (B, C, k_slots)

        tau_weights = F.softmax(topk_vals, dim=-1)  # 仅由 FFT 幅值得到权重
        return periods, tau_weights


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
        branch_outs = [conv(x).squeeze(2) for conv in self.convs]  # 每项 (B, 1, S)
        stacked = torch.stack(branch_outs, dim=1)  # (B, num_branches, 1, S)
        mixed = (stacked * branch_weight.view(B, self.num_branches, 1, 1)).sum(dim=1)  # (B, 1, S)
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
        # x, q: (B, C, S)；branch_weight: (B, num_branches)
        _, C, S = x.shape
        # Step 1: Mapping（q 在时间上逐通道 LayerNorm 再进 linear）
        global_query = self.linear(self.q_norm(q))  # (B, C, S)

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

        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.spectrum_k = int(getattr(configs, 'sapmixer_spectrum_k', 4))
        self.spectrum_cum_ratio = float(getattr(configs, "sapmixer_spectrum_cum_ratio", 0.9))
        self.use_multiscale = bool(int(getattr(configs, 'sapmixer_use_multiscale', 1)))
        self.period_array = self._parse_period_array_list(configs.sapmixer_period_array)
        branch_kernels = self._parse_branch_kernels(self.period_array)

        self.phase_proj = nn.Linear(2, 1, bias=True)
        nn.init.xavier_uniform_(self.phase_proj.weight, gain=0.1)
        nn.init.zeros_(self.phase_proj.bias)
        self.query_channel_attn = nn.MultiheadAttention(
            embed_dim=self.seq_len,
            num_heads=1,
            batch_first=True,
        )
        self.query_channel_norm = nn.LayerNorm(self.seq_len)
        self.period_estimator = DynamicPeriodEstimator(
            spectrum_k=self.spectrum_k,
            tau_min=2.0,
            tau_max=512.0,
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

    def _parse_period_array_list(self, period_array):
        """Comma-separated string or sequence -> list of floats (for tensors / matching)."""
        if isinstance(period_array, str):
            return [float(x.strip()) for x in period_array.split(',') if x.strip()]
        return [float(x) for x in period_array]

    def _parse_branch_kernels(self, period_array):
        """Effective scales: round(period * 0.25), clamp to >=1, then force odd (for Conv padding)."""
        if isinstance(period_array, str):
            nums = self._parse_period_array_list(period_array)
        else:
            nums = [float(x) for x in period_array]
        out = []
        for v in nums:
            k = int(round(v * 0.25))
            k = max(k, 1)
            if k % 2 == 0:
                k += 1
            out.append(k)
        return out

    def _build_multiscale_query(self, tau_all, tau_weight, seq_len, device):
        # tau_all, tau_weight: (B, C, K)；S = seq_len -> query_per_var: (B, C, S)
        time_index = torch.arange(
            seq_len, device=device, dtype=torch.float32
        ).view(1, 1, 1, -1)  # (1, 1, 1, S)
        theta = (2.0 * torch.pi * time_index) / tau_all.unsqueeze(-1)  # (B, C, K, S)
        phase_feats = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)  # (B, C, K, S, 2)
        proj = self.phase_proj(phase_feats).squeeze(-1)  # (B, C, K, S)
        query_per_var = (proj * tau_weight.unsqueeze(-1)).sum(dim=2)  # (B, C, S)；tau_weight.unsqueeze(-1): (B,C,K,1)
        return query_per_var

    def _build_branch_weight_from_periods(self, tau_all, tau_weight):
        # Period-kernel matching: score_j = sum_i w_i * exp(-|log(tau_i)-log(k_j)| / T)
        # tau_all, tau_weight: (B, K)；M = len(period_array)
        Tm = torch.exp(self.log_Tm)  # 标量
        period_tensor = tau_all.new_tensor(self.period_array)  # (M,)
        log_tau = torch.log(tau_all).unsqueeze(-1)  # (B, K, 1)
        log_period = torch.log(period_tensor).view(1, 1, -1)  # (1, 1, M)
        dist = (log_tau - log_period).abs()  # (B, K, M)
        match = torch.exp(-dist / Tm)  # (B, K, M)
        score = (tau_weight.unsqueeze(-1) * match).sum(dim=1)  # (B, M)；tau_weight.unsqueeze(-1): (B, K, 1)
        return F.softmax(score, dim=-1)  # (B, M)

    def forward(self, x):
        # x: (B, S, C)；S=seq_len，C=enc_in
        # RevIN normalize（沿时间维）
        seq_mean = torch.mean(x, dim=1, keepdim=True)  # (B, 1, C)
        seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5  # (B, 1, C)
        seq_std = seq_var.sqrt()  # (B, 1, C)
        x = (x - seq_mean) / seq_std  # (B, S, C)

        x_input = x.permute(0, 2, 1)  # (B, C, S)

        # \subsection{动态周期感知检索（Dynamic Period-Aware Retrieval）}
        # Code anchor: estimate candidate periods and sample-wise period weights.
        tau_all, tau_weight = self.period_estimator(x_input)  # 均为 (B, C, K)，K 为 FFT top-k 槽位数
        if not self.use_multiscale:
            tau_weight = tau_weight.new_zeros(tau_weight.shape)
            tau_weight[:, :, 0] = 1.0

        query_per_var = self._build_multiscale_query(
            tau_all=tau_all,
            tau_weight=tau_weight,
            seq_len=x_input.shape[-1],
            device=x_input.device,
        )  # (B, C, S)；此处 S 即 seq_len，与 MultiheadAttention 的 embed_dim 一致

        # MultiheadAttention batch_first: (B, C, S) 视作 (batch, 通道数 tokens, embed_dim=S)
        attn_out, _ = self.query_channel_attn(
            query_per_var,
            query_per_var,
            query_per_var,
        )  # attn_out: (B, C, S)
        query_input = self.query_channel_norm(query_per_var + attn_out)  # (B, C, S)

        tau_all_shared = tau_all.mean(dim=1)  # (B, K)，跨通道平均
        tau_weight_shared = tau_weight.mean(dim=1)  # (B, K)
        branch_weight = self._build_branch_weight_from_periods(
            tau_all_shared,
            tau_weight_shared,
        )  # (B, M)，M = len(period_array)
        global_information = self.sap_mixer(x_input, query_input, branch_weight=branch_weight)  # (B, C, S)

        # Projection + MLP；d_model = configs.d_model
        input_proj = self.input_proj(x_input + global_information)  # (B, C, d_model)
        hidden = self.model(input_proj)  # (B, C, d_model)
        output = self.output_proj(hidden + input_proj).permute(0, 2, 1)  # 先 (B, C, pred_len) -> (B, pred_len, C)

        # RevIN de-normalize
        output = output * seq_std + seq_mean
        return output
