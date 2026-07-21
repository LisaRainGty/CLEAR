"""CLAIMARC 模型（严格对齐 §3.2.4–§3.2.6）。

- Shared BGE-large-zh encoder + LoRA（query/value 投影，r=16,α=32）。
- 可训练：LoRA 矩阵、编码器 LayerNorm、新增特殊 token 的 embedding 行（梯度掩码）、
  TwoStreamFusion、两个任务头。骨干其余权重冻结。
- TwoStreamFusion：N=2 层，pre-LayerNorm；流内 self-attn → 双向 cross-attn
  （两方向共享 Q/K/V 投影）→ SwiGLU FFN。8 头，d_head=128。
- 任务头：Pair-LRC（ESIM 4 元组）+ RetrEmb（256 维单位球）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d, hidden=None):
        super().__init__()
        hidden = hidden or 4 * d
        self.w1 = nn.Linear(d, hidden)
        self.w2 = nn.Linear(d, hidden)
        self.w3 = nn.Linear(hidden, d)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GELUFFN(nn.Module):
    def __init__(self, d, hidden=None):
        super().__init__()
        hidden = hidden or 4 * d
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, x):
        return self.net(x)


class FusionLayer(nn.Module):
    """单层 TwoStreamFusion（pre-LN）。

    ablation 旋钮：
      xattn_dir: both | c2e（仅 claim 改写 evidence）| e2c（仅 evidence 改写 claim）
      indep_proj: True → 两方向独立投影；False → 共享（默认，对称比较几何）
      ffn: swiglu | gelu
    """

    def __init__(self, d=1024, heads=8, dropout=0.1, xattn_dir="both",
                 indep_proj=False, ffn="swiglu"):
        super().__init__()
        self.xattn_dir = xattn_dir
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.cross_attn2 = (nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
                            if indep_proj else None)
        self.ln_s_c, self.ln_s_e = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ln_x_c, self.ln_x_e = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ln_f_c, self.ln_f_e = nn.LayerNorm(d), nn.LayerNorm(d)
        FFN = SwiGLU if ffn == "swiglu" else GELUFFN
        self.ffn_c, self.ffn_e = FFN(d), FFN(d)

    def forward(self, hc, he, c_pad, e_pad):
        sc, _ = self.self_attn(*([self.ln_s_c(hc)] * 3), key_padding_mask=c_pad, need_weights=False)
        hc = hc + sc
        se, _ = self.self_attn(*([self.ln_s_e(he)] * 3), key_padding_mask=e_pad, need_weights=False)
        he = he + se
        nc, ne = self.ln_x_c(hc), self.ln_x_e(he)
        xa2 = self.cross_attn2 or self.cross_attn
        if self.xattn_dir in ("both", "c2e"):
            xc, _ = self.cross_attn(nc, ne, ne, key_padding_mask=e_pad, need_weights=False)
            hc = hc + xc
        if self.xattn_dir in ("both", "e2c"):
            xe, _ = xa2(ne, nc, nc, key_padding_mask=c_pad, need_weights=False)
            he = he + xe
        hc = hc + self.ffn_c(self.ln_f_c(hc))
        he = he + self.ffn_e(self.ln_f_e(he))
        return hc, he


class CLAIMARC(nn.Module):
    def __init__(self, bge_path: str, vocab_size: int, n_special: int, n_fusion=2,
                 use_lora=True, ret_dim=256, fusion_dropout=0.1, lora_rank=16,
                 xattn_dir="both", indep_proj=False, ffn="swiglu", heads=8,
                 enc_train="lora", unfreeze_top=0, ret_disc=True,
                 head_4tuple=True, joint_encode=False):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(bge_path)
        self.encoder.resize_token_embeddings(vocab_size)
        d = self.encoder.config.hidden_size
        self.d = d
        # head_4tuple=False：分类/检索头仅用 [h_c, h_e] 拼接（消融 ESIM 差/积交互特征）。
        # joint_encode=True：先把 claim+evidence 拼成单序列由共享编码器统一编码、再拆回两流
        #   （跨流交互在编码阶段已发生），随后融合/头保持不变（消融"独立编码再融合"vs"统一编码再拆流"）。
        self.head_4tuple = head_4tuple
        self.joint_encode = joint_encode
        self.max_pos = int(getattr(self.encoder.config, "max_position_embeddings", 512) or 512)
        # enc_train: lora（默认，LoRA+LN+特殊embedding）| topk（解冻顶部 unfreeze_top 层）| full（全参微调）
        self.enc_train = enc_train
        use_lora = use_lora and (enc_train == "lora")
        self.use_lora = use_lora
        if use_lora:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(
                r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05,
                target_modules=["query", "value"], bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.encoder = get_peft_model(self.encoder, cfg)
        self._unfreeze_encoder_extras(vocab_size, n_special, use_lora,
                                      enc_train=enc_train, unfreeze_top=unfreeze_top)
        self.fusion = nn.ModuleList([
            FusionLayer(d, heads=heads, dropout=fusion_dropout, xattn_dir=xattn_dir,
                        indep_proj=indep_proj, ffn=ffn) for _ in range(n_fusion)])
        cls_in = (4 * d) if head_4tuple else (2 * d)
        self.lrc_ln = nn.LayerNorm(cls_in)
        self.lrc_drop = nn.Dropout(fusion_dropout)
        self.lrc = nn.Linear(cls_in, 1)
        # 检索头输入用 ESIM 4 元组（含 claim−evidence 差/积），使检索空间对齐"宣传-事实
        # 落差"这一虚假宣传判别信号（§3.2.6），强化 RACL 检索表征的标签一致性。
        self.ret_disc = ret_disc
        ret_in = (4 * d) if (ret_disc and head_4tuple) else (2 * d)
        self.ret = nn.Sequential(
            nn.Linear(ret_in, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, ret_dim)
        )

    def _unfreeze_encoder_extras(self, vocab_size, n_special, use_lora,
                                 enc_train="lora", unfreeze_top=0):
        """§3.2.4：解冻编码器 LayerNorm 与新增特殊 token 的 embedding 行（梯度掩码）。
        enc_train=full：解冻全部编码器参数；topk：仅解冻顶部 unfreeze_top 个 Transformer 层。
        训练完成后编码器冻结、以检索库扩展新域（§3.2.9），故训练期更大容量与部署侧冻结不冲突。"""
        if enc_train == "full":
            for p in self.encoder.parameters():
                p.requires_grad_(True)
            return
        emb = self.encoder.get_input_embeddings()
        emb.weight.requires_grad_(True)
        new_start = vocab_size - n_special
        mask = torch.zeros(vocab_size, 1)
        mask[new_start:] = 1.0
        self.register_buffer("_emb_grad_mask", mask, persistent=False)

        def _hook(grad):
            return grad * self._emb_grad_mask.to(grad.device, grad.dtype)
        emb.weight.register_hook(_hook)
        for name, p in self.encoder.named_parameters():
            if "LayerNorm" in name or "layer_norm" in name:
                p.requires_grad_(True)
        if enc_train == "topk" and unfreeze_top > 0:
            # 解冻顶部 unfreeze_top 个 encoder 层（BERT/BGE 命名 encoder.layer.{i}）
            import re
            layer_ids = set()
            for name, _ in self.encoder.named_parameters():
                m = re.search(r"\.layer\.(\d+)\.", name)
                if m:
                    layer_ids.add(int(m.group(1)))
            if layer_ids:
                top = sorted(layer_ids)[-unfreeze_top:]
                for name, p in self.encoder.named_parameters():
                    m = re.search(r"\.layer\.(\d+)\.", name)
                    if m and int(m.group(1)) in top:
                        p.requires_grad_(True)

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        return out.last_hidden_state

    def forward(self, c_ids, c_mask, e_ids, e_mask):
        if self.joint_encode:
            # 统一编码再拆流：claim+evidence 拼成单序列由共享编码器一次性编码，
            # 跨流自注意力在编码阶段即发生；随后按 claim 长度拆回两流，融合/头保持不变。
            joint_ids = torch.cat([c_ids, e_ids], dim=1)[:, : self.max_pos]
            joint_mask = torch.cat([c_mask, e_mask], dim=1)[:, : self.max_pos]
            H = self.encode(joint_ids, joint_mask)
            Lc = min(c_ids.size(1), H.size(1) - 1)
            hc, he = H[:, :Lc], H[:, Lc:]
            c_pad = joint_mask[:, :Lc] == 0
            e_pad = joint_mask[:, Lc:] == 0
        else:
            hc = self.encode(c_ids, c_mask)
            he = self.encode(e_ids, e_mask)
            c_pad = c_mask == 0
            e_pad = e_mask == 0
        for layer in self.fusion:
            hc, he = layer(hc, he, c_pad, e_pad)
        h_c = hc[:, 0]
        h_e = he[:, 0]
        if self.head_4tuple:
            z = torch.cat([h_c, h_e, h_c - h_e, h_c * h_e], dim=-1)
            ret_in = z if self.ret_disc else torch.cat([h_c, h_e], dim=-1)
        else:
            z = torch.cat([h_c, h_e], dim=-1)
            ret_in = z
        logit = self.lrc(self.lrc_drop(self.lrc_ln(z))).squeeze(-1)
        g = F.normalize(self.ret(ret_in), dim=-1)
        return logit, g

    def param_groups(self, lr_encoder=2e-5, lr_head=1e-4):
        """§3.2.8 差分学习率：编码器侧(LoRA/LayerNorm/特殊embedding)=2e-5；融合+头=1e-4。"""
        enc, head = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (enc if n.startswith("encoder.") else head).append(p)
        return [{"params": enc, "lr": lr_encoder}, {"params": head, "lr": lr_head}]
