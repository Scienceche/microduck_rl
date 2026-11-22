#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2 双专家 Actor: 行走分支 + 站立/恢复分支 + soft gate.
StableMimic 架构(2026): 两个 expert 的输出按 gz 门控连续混合,
部署时单一 ONNX, robotd 无感知. 行走分支 PPO 冻结(只让 BC 锚定),
恢复分支正常 PPO 训练.

关键设计:
- gate 输入: projected_gravity[2] (61 维 obs 的一个切片)
  |gz+1| < 0.4 (接近直立) → gate≈0 → 行走专家主导
  |gz+1| > 0.4 (倾斜/倒地) → gate≈1 → 恢复专家主导
- 行走分支从 run8 model_3999 加载, PPO 冻结
- 恢复分支从 run6 model_3999 加载, 正常 PPO
- obs_normalizer 共享(已验证两模型 mean 差异 < 0.65)
"""
import torch
import torch.nn as nn
from rsl_rl.modules.mlp import MLP


class DualExpertActor(nn.Module):
    """Two-branch actor with a soft gravity gate.

    Output = (1 - gate) * walk_expert(obs) + gate * stand_expert(obs)
    where gate = sigmoid(k * |gz_norm + offset|) from the gravity obs slice.
    """

    def __init__(self, obs_dim: int = 61, action_dim: int = 14,
                 hidden_dims: tuple = (512, 256, 128),
                 gravity_slice: int = None,
                 gate_k: float = 8.0,
                 gate_offset: float = 0.4):
        super().__init__()
        self.walk_expert = MLP(obs_dim, action_dim, hidden_dims, "elu")
        self.stand_expert = MLP(obs_dim, action_dim, hidden_dims, "elu")
        # gate is a simple learnable scalar on |gz - target|
        self.gate_k = nn.Parameter(torch.tensor(gate_k))
        self.gate_offset = nn.Parameter(torch.tensor(gate_offset))
        self.gravity_slice = gravity_slice  # index of gz in flat obs
        # freeze walk expert's PPO gradients
        self._freeze_walk = True

    def freeze_walk_expert(self, freeze: bool = True):
        self._freeze_walk = freeze
        for p in self.walk_expert.parameters():
            p.requires_grad = not freeze

    def gate(self, flat_obs: torch.Tensor) -> torch.Tensor:
        """Soft gate from gravity z component. 0=walk, 1=stand/recovery."""
        if flat_obs.dim() == 1:
            flat_obs = flat_obs.unsqueeze(0)
        if flat_obs.shape[-1] > 1:
            gz = flat_obs[:, self.gravity_slice]
        else:
            # standalone gate test: input IS gz
            gz = flat_obs[:, 0]
        # tilt_metric: 0 when upright (gz≈-1), large when tilted
        tilt = (gz + 1.0).abs()
        return torch.sigmoid(self.gate_k * (tilt - self.gate_offset))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self._freeze_walk:
            with torch.no_grad():
                walk_out = self.walk_expert(obs)
        else:
            walk_out = self.walk_expert(obs)
        stand_out = self.stand_expert(obs)
        g = self.gate(obs)  # [batch, 1] via broadcast
        g = g.unsqueeze(-1)
        return (1 - g) * walk_out + g * stand_out

    def _load_sequential(self, seq: nn.Module, state_dict: dict, prefix: str):
        """Copy weights from checkpoint mlp.* keys into a Sequential."""
        idx = 0
        loaded = 0
        for name, module in seq.named_modules():
            if isinstance(module, nn.Linear):
                ckpt_key = f"mlp.{idx}"
                if ckpt_key + ".weight" in state_dict:
                    module.weight.data.copy_(state_dict[ckpt_key + ".weight"])
                    if module.bias is not None and ckpt_key + ".bias" in state_dict:
                        module.bias.data.copy_(state_dict[ckpt_key + ".bias"])
                    loaded += 1
                idx += 2  # Linear at even indices in Sequential
        return loaded

    def load_walk_expert(self, state_dict: dict):
        n = self._load_sequential(self.walk_expert, state_dict, "mlp")
        assert n >= 3, f"only loaded {n} layers for walk expert"

    def load_stand_expert(self, state_dict: dict):
        n = self._load_sequential(self.stand_expert, state_dict, "mlp")
        assert n >= 3, f"only loaded {n} layers for stand expert"


if __name__ == "__main__":
    # Smoke test
    model = DualExpertActor(gravity_slice=5)  # projected_gravity z is obs index 5 (after 3 ang_vel)
    x = torch.randn(4, 61)
    out = model(x)
    print("output shape:", out.shape, "(expect [4, 14])")
    gate = model.gate(x)
    print("gate values:", gate.detach().numpy())
    print("upright test (gz=-0.99):", model.gate(torch.tensor([[-0.99]*1])) .item())
    print("tilted test (gz=-0.3):", model.gate(torch.tensor([[-0.3]*1])).item())
