import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
import torch.nn.functional as F


class SharedProjection(nn.Module):

    def __init__(self, hidden_size, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size)
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CrossTaskAttention(nn.Module):

    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        assert hidden_size % num_heads == 0, \
            f"hidden_size({hidden_size}) must be divisible by num_heads({num_heads})"

    def forward(self, shared_feat, task_feat):
        attn_output, _ = self.multihead_attn(
            query=shared_feat,
            key=task_feat,
            value=task_feat
        )
        return self.layer_norm(shared_feat + self.dropout(attn_output))

class AttnPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, features):
        # features: [B, S, H]
        attn_weights = torch.softmax(self.attn(features), dim=1)  # [B, S, 1]
        pooled = (features * attn_weights).sum(dim=1)  # [B, H]
        return pooled
class RhymePredictor(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.pool = AttnPool(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, features):
        # features: [B, S, H]
        pooled = self.pool(features)  # [B, H]
        return self.layer_norm(self.dropout(pooled))  # [B, H]


class MultiTaskBERT(BertPreTrainedModel):
    def __init__(self, config,
                 use_mlm=True,
                 use_tone=False,
                 use_rhyme=False,
                 shared_layers=3,
                 alpha=0.6, beta=0.2, gamma=0.2):
        super().__init__(config
        self.task_flags = {
            'mlm': use_mlm,
            'tone': use_tone,
            'rhyme': use_rhyme
        }

        self.bert = BertModel(config)

        self.shared_proj = SharedProjection(config.hidden_size, shared_layers)

        self.cross_attn = CrossTaskAttention(config.hidden_size)

        if self.task_flags['tone']:
            self.tone_emb = nn.Embedding(5, 32, padding_idx=4)
            self.tone_fc = nn.Linear(32, config.hidden_size)
            self.tone_fusion = nn.Linear(config.hidden_size * 2, config.hidden_size)

        if self.task_flags['rhyme']:
            self.rhyme_proj = nn.Linear(768, config.hidden_size)  
            self.rhyme_attn = RhymePredictor(config.hidden_size)
            self.rhyme_fusion = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid()
        )

        self.shared_head = nn.Linear(config.hidden_size, config.hidden_size)

        if self.task_flags['mlm']:
            self.mlm_head = nn.Sequential(
                self.shared_head,
                nn.Linear(config.hidden_size, config.vocab_size)
            )
        if self.task_flags['tone']:
            self.tone_head = nn.Sequential(
                self.shared_head,  
                nn.Linear(config.hidden_size, 4)
            )
        if self.task_flags['rhyme']:
            self.rhyme_head = nn.Sequential(
                self.shared_head, 
                nn.Linear(config.hidden_size, 106)
            )

        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.init_weights()
        if self.task_flags['tone']:
            assert self.tone_emb.num_embeddings >= 5, "平仄词表维度不足"

    def build_features(self, base_output, task_inputs):
        shared_feat = self.shared_proj(base_output)

        task_feat = shared_feat.clone()

        if self.task_flags['tone'] and 'tone_ids' in task_inputs:
            tone_ids = task_inputs['tone_ids']
            assert tone_ids.max() <= 4 and tone_ids.min() >= 0, f"平仄ID越界! 应为0-4, 实际为{tone_ids.min()}-{tone_ids.max()}"

            tone_emb = self.tone_fc(self.tone_emb(tone_ids))  # [B,S,H]
            
            concat_feat = torch.cat([task_feat, tone_emb], dim=-1)  # [B,S,2H]
            task_feat = self.tone_fusion(concat_feat)  # [B,S,H]
            task_feat = F.gelu(task_feat) + shared_feat

        if self.task_flags['rhyme'] and 'rhyme_ids' in task_inputs and hasattr(self, 'rhyme_proj'):
            rhyme_ids = task_inputs['rhyme_ids']

            rhyme_proj = self.rhyme_proj(rhyme_ids)  # [B,S,H]

            concat_feat = torch.cat([task_feat, rhyme_proj], dim=-1)  # [B,S,2H]
            task_feat = self.rhyme_fusion(concat_feat)  # [B,S,H]
            task_feat = F.gelu(task_feat) + shared_feat

        fused_feat = self.cross_attn(shared_feat, task_feat)

        combined = torch.cat([shared_feat, fused_feat], dim=-1)
        gate = self.gate(combined)
        return gate * shared_feat + (1 - gate) * fused_feat

    def forward(self,
                input_ids,
                attention_mask=None,
                tone_ids=None,
                rhyme_ids=None,  
                rhyme_label=None,  
                input_labels=None,
                tone_labels=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        base_output = outputs.last_hidden_state

        task_inputs = {}
        if self.task_flags['tone'] and tone_ids is not None:
            task_inputs['tone_ids'] = tone_ids
        if self.task_flags['rhyme'] and rhyme_ids is not None and hasattr(self, 'rhyme_proj'):
            task_inputs['rhyme_ids'] = rhyme_ids

        fused_features = self.build_features(base_output, task_inputs)

        outputs = {}
        if self.task_flags['mlm']:
            outputs['mlm_logits'] = self.mlm_head(fused_features)
        if self.task_flags['tone']:
            outputs['tone_logits'] = self.tone_head(fused_features)
        if self.task_flags['rhyme']:
            rhyme_feat = self.rhyme_attn(fused_features)  # [B,1,H]
            outputs['rhyme_logits'] = self.rhyme_head(rhyme_feat.squeeze(1))

        loss = 0
        if (input_labels is not None) or (tone_labels is not None) or (rhyme_label is not None):
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

            if self.task_flags['mlm'] and input_labels is not None:
                mlm_loss = loss_fct(
                    outputs['mlm_logits'].view(-1, self.config.vocab_size),
                    input_labels.view(-1)
                ) * self.alpha
                loss += mlm_loss

            if self.task_flags['tone'] and tone_labels is not None:
                tone_loss = loss_fct(
                    outputs['tone_logits'].view(-1, 4),
                    tone_labels.view(-1)
                ) * self.beta
                loss += tone_loss

            if self.task_flags['rhyme'] and rhyme_label is not None:
                # print("rhyme_logits shape:", outputs['rhyme_logits'].shape)
                # print("rhyme_label shape:", rhyme_label.shape)
                rhyme_loss = loss_fct(
                    outputs['rhyme_logits'],
                    rhyme_label.squeeze(1)                 ) * self.gamma
                loss += rhyme_loss

        return {
            'loss': loss if loss != 0 else None,
            **outputs
        }
