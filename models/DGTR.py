import torch
import torch.nn as nn
import torch.nn.functional as F


# \subsection{动态周期感知检索（Dynamic Period-Aware Retrieval）}
# Code anchor: DynamicPeriodEstimator class and Model.forward period estimation block.
class DynamicPeriodEstimator(nn.Module):
    """Estimate sample-wise multi-period candidates and weights."""

    def __init__(self, spectrum_k, tau_min, tau_max, tau_init, freq_smooth, gate_temp, stats_dim=3):
        super().__init__()
        self.spectrum_k = max(1, int(spectrum_k))
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.freq_smooth = min(max(float(freq_smooth), 0.0), 1.0)
        self.gate_temp = max(float(gate_temp), 1e-3)

        tau_init = max(min(float(tau_init), self.tau_max - 1e-4), self.tau_min + 1e-4)
        tau_ratio = (tau_init - self.tau_min) / (self.tau_max - self.tau_min)
        tau_ratio = min(max(tau_ratio, 1e-4), 1.0 - 1e-4)
        self.raw_tau = nn.Parameter(torch.logit(torch.tensor(tau_ratio)))
        self.base_harmonic_logit = nn.Parameter(torch.tensor(0.0))

        gate_hidden = max(16, stats_dim * 8)
        self.query_gate = nn.Sequential(
            nn.Linear(stats_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.spectrum_k + 1),
        )

    def forward(self, x_input):
        # x_input: (B, C, S)
        B, _, S = x_input.shape
        centered = x_input - x_input.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(centered, dim=-1)
        amplitude = spectrum.abs().mean(dim=1)  # (B, F)
        amplitude[:, 0] = 0.0

        valid_bins = max(0, amplitude.shape[-1] - 1)
        topk = min(self.spectrum_k, valid_bins)
        if topk > 0:
            topk_vals, topk_idx = torch.topk(amplitude[:, 1:], k=topk, dim=-1)
            topk_idx = topk_idx + 1
            if self.freq_smooth > 0:
                topk_vals = (1.0 - self.freq_smooth) * topk_vals + self.freq_smooth * topk_vals.mean(dim=0, keepdim=True)
            periods = (float(S) / topk_idx.float()).clamp(self.tau_min, self.tau_max)
        else:
            topk_vals = amplitude.new_zeros(B, 0)
            periods = amplitude.new_zeros(B, 0)

        if valid_bins > 0:
            valid_amp = amplitude[:, 1:]
            spec_stats = torch.stack(
                [
                    valid_amp.mean(dim=-1),
                    valid_amp.max(dim=-1).values,
                    valid_amp.std(dim=-1, unbiased=False),
                ],
                dim=-1,
            )
        else:
            spec_stats = amplitude.new_zeros(B, 3)

        tau_base = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.raw_tau)
        tau_base = tau_base.expand(B, 1)
        tau_all = torch.cat([tau_base, periods], dim=1)

        base_score = self.base_harmonic_logit.expand(B, 1)
        prior_scores = torch.cat([base_score, topk_vals], dim=1)
        tau_logits = self.query_gate(spec_stats)[:, :tau_all.shape[1]] + prior_scores
        tau_weights = F.softmax(tau_logits / self.gate_temp, dim=-1)
        return tau_all, tau_weights, spec_stats


class DGTR(nn.Module):
    # \subsection{自适应多分支时序融合（Adaptive Multi-Branch Temporal Fusion）}
    # Code anchor: multi-branch Conv2d retrieval with branch_weight gating.
    def __init__(self, d_series, c, CI=False, period_len=24, branch_kernels=None, agg=True):
        super(DGTR, self).__init__()
        self.agg = agg
        self.period_len = period_len
        self.c = c
        self.branch_kernels = self._normalize_kernels(branch_kernels)
        self.num_branches = len(self.branch_kernels)

        self.q_norm = nn.LayerNorm(d_series)
        self.linear = nn.Linear(d_series, d_series)
        self.CI = CI
        if self.CI:
            self.ds_convs = nn.ModuleList([
                nn.ModuleList([
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=(2, kernel),
                        stride=1,
                        padding=(0, kernel // 2),
                        padding_mode="zeros",
                        bias=False,
                    )
                    for kernel in self.branch_kernels
                ])
                for _ in range(self.c)
            ])
        else:
            self.branches = nn.ModuleList([
                nn.Conv2d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=(2, kernel),
                    stride=1,
                    padding=(0, kernel // 2),
                    padding_mode="zeros",
                    bias=False,
                )
                for kernel in self.branch_kernels
            ])

    def _normalize_kernels(self, branch_kernels):
        if not branch_kernels:
            branch_kernels = [1 + 2 * (self.period_len // 2)]
        norm_kernels = []
        for kernel in branch_kernels:
            k = max(1, int(kernel))
            if k % 2 == 0:
                k += 1
            norm_kernels.append(k)
        return norm_kernels

    def forward(self, x, q, branch_weight=None):
        B, C, S = x.shape
        # Step 1: Mapping（q 在时间上逐通道 LayerNorm 再进 linear）
        global_query = self.linear(self.q_norm(q))

        # Step 2: GTA mode, aggregate along temporal length.
        if self.agg:
            weight = F.softmax(global_query, dim=-1)  # normalize over length S
            global_query = torch.sum(global_query * weight, dim=-1, keepdim=True)  # (B, C, 1)
            global_query = global_query.repeat(1, 1, S)  # (B, C, S)

        # Step 3: Fuse
        out = torch.stack([x, global_query], dim=2)  # (B, C, 2, S)
        if branch_weight is None:
            branch_weight = out.new_full((B, self.num_branches), 1.0 / self.num_branches)
        else:
            branch_weight = branch_weight.clamp_min(1e-6)
            branch_weight = branch_weight / branch_weight.sum(dim=-1, keepdim=True)

        if self.CI:
            conv_outs = []
            for channel in range(self.c):
                channel_in = out[:, channel, :, :].unsqueeze(1)  # (B, 1, 2, S)
                branch_outs = [
                    self.ds_convs[channel][branch_id](channel_in).squeeze(2)  # (B, 1, S)
                    for branch_id in range(self.num_branches)
                ]
                stacked = torch.stack(branch_outs, dim=1)  # (B, N, 1, S)
                mixed = (stacked * branch_weight.view(B, self.num_branches, 1, 1)).sum(dim=1)  # (B, 1, S)
                conv_outs.append(mixed)
            conv_out = torch.cat(conv_outs, dim=1)  # (B, C, S)
        else:
            out = out.reshape(-1, 1, 2, S)  # (B*C, 1, 2, S)
            branch_outs = [branch(out).squeeze(2) for branch in self.branches]  # N * [(B*C, 1, S)]
            stacked = torch.stack(branch_outs, dim=1)  # (B*C, N, 1, S)
            expanded_weight = branch_weight.repeat_interleave(C, dim=0).view(B * C, self.num_branches, 1, 1)
            conv_out = (stacked * expanded_weight).sum(dim=1)  # (B*C, 1, S)
            conv_out = conv_out.reshape(-1, C, S)  # (B, C, S)
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
        self.use_revin = configs.use_revin
        self.individual = configs.individual

        self.spectrum_k = max(1, int(getattr(configs, 'dgtr_spectrum_k', 4)))
        self.freq_smooth = min(max(float(getattr(configs, 'dgtr_freq_smooth', 0.2)), 0.0), 1.0)
        self.gate_temp = max(float(getattr(configs, 'dgtr_gate_temp', 1.0)), 1e-3)
        self.use_multiscale = bool(int(getattr(configs, 'dgtr_use_multiscale', 1)))
        branch_kernels = self._parse_branch_kernels(getattr(configs, 'dgtr_branch_kernels', '3,7,15,31'))

        self.tau_min = float(getattr(configs, 'learnable_tau_min', 2.0))
        self.tau_max = float(getattr(configs, 'learnable_tau_max', 512.0))
        if self.tau_max <= self.tau_min:
            raise ValueError('learnable_tau_max must be greater than learnable_tau_min')
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
            freq_smooth=self.freq_smooth,
            gate_temp=self.gate_temp,
        )

        self.DGTR = DGTR(
            d_series=self.seq_len,
            c=self.enc_in,
            CI=self.individual,
            branch_kernels=branch_kernels,
            agg=False,
        )

        gate_hidden = max(16, self.enc_in)
        self.branch_gate = nn.Sequential(
            nn.Linear(3, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.DGTR.num_branches),
        )

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
        elif isinstance(raw_branch_kernels, (list, tuple)):
            kernels = list(raw_branch_kernels)
        else:
            kernels = []
        if not kernels:
            kernels = [3, 7, 15, 31]
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

    def forward(self, x, cycle_index=None):
        # RevIN normalize
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # (B, S, C) -> (B, C, S)
        x_input = x.permute(0, 2, 1)

        # \subsection{动态周期感知检索（Dynamic Period-Aware Retrieval）}
        # Code anchor: estimate candidate periods and sample-wise period weights.
        tau_all, tau_weight, spec_stats = self.period_estimator(x_input)
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
        # Code anchor: compute branch-wise gates from spectral stats, then
        # fuse multi-branch temporal retrieval outputs in DGTR.
        branch_weight = F.softmax(self.branch_gate(spec_stats) / self.gate_temp, dim=-1)
        global_information = self.DGTR(x_input, query_input, branch_weight=branch_weight)

        # Projection + MLP
        input_proj = self.input_proj(x_input + global_information)
        hidden = self.model(input_proj)
        output = self.output_proj(hidden + input_proj).permute(0, 2, 1)

        # RevIN de-normalize
        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean
        return output

