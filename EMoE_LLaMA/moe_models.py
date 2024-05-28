import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
    LlamaRMSNorm,
    LlamaForCausalLM
)



class MoELlamaConfig(LlamaConfig):
    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        # hyper-parameter of MoEfication
        split_start_layer=0,
        split_every_layer=2,
        topk=2,
        n_expert=16,
        mode='EMoE',  # other setting
        select='gate', # or 'up', 'inter'
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            pad_token_id,
            bos_token_id,
            eos_token_id,
            pretraining_tp,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            **kwargs,
        )
        self.split_start_layer = split_start_layer
        self.split_every_layer = split_every_layer
        self.topk = topk
        self.n_expert = n_expert
        self.select = select
        self.mode = mode


class MoELlamaMLP(LlamaMLP):
    
    def __init__(self, config: MoELlamaConfig):
        super().__init__(config)
        self.config = config
        
    def forward(self, x):
        
        if self.config.mode == 'EMoE':
            if self.config.select == 'gate':
                gate_output = self.gate_proj(x) # bs, seq_len, 4h
                expert_scores = gate_output.reshape(x.shape[0], x.shape[1], self.config.n_expert, -1)
                intermediate_states = self.act_fn(gate_output) * self.up_proj(x) # bs, seq_len, hidden_size
            elif self.config.select == 'up':
                gate_output = self.gate_proj(x)
                up_output = self.up_proj(x)
                expert_scores = up_output.reshape(x.shape[0], x.shape[1], self.config.n_expert, -1)
                intermediate_states = self.act_fn(gate_output) * up_output
            elif self.config.select == 'up_abs':
                gate_output = self.gate_proj(x)
                up_output = self.up_proj(x)
                expert_scores = torch.abs(up_output).reshape(x.shape[0], x.shape[1], self.config.n_expert, -1)
                intermediate_states = self.act_fn(gate_output) * up_output
            elif self.config.select == 'inter_abs':
                intermediate_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                expert_scores = torch.abs(intermediate_states).reshape(x.shape[0], x.shape[1], self.config.n_expert, -1)
            elif self.config.select == 'inter':
                intermediate_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                expert_scores = intermediate_states.reshape(x.shape[0], x.shape[1], self.config.n_expert, -1)
            else:
                raise NotImplementedError
        
            expert_scores = torch.mean(expert_scores, dim=-1) # bs, seq_len, number_of_experts
            expert_topk_indices = torch.topk(expert_scores, k=self.config.topk, dim=-1).indices 
            expert_topk_mask = torch.zeros_like(expert_scores)
            expert_topk_mask.scatter_(dim=-1, index=expert_topk_indices, value=1)
            expert_topk_mask = expert_topk_mask.repeat_interleave(self.config.intermediate_size // self.config.n_expert, dim=-1) # bs, seq_len, hidden_size

            # do block masking
            intermediate_states = intermediate_states * expert_topk_mask
            down_proj = self.down_proj(intermediate_states)
            
        return down_proj



class EMoELlamaMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.config = config # clustering FFNs weights to construct the experts
    def forward(self, x):
        gate_output = self.gate_proj(x) # bs, seq_len, 8/3h
        intermediate_states = self.act_fn(gate_output) * self.up_proj(x) # bs, seq_len, 8/3h
        # standard SwiGLU FFN:
        # return self.down_proj(intermediate_states)
        # EMoE:
        expert_scores = gate_output.reshape(x.shape[0], x.shape[1], self.config.n_expert, -1)
        # Avg-k gating: avergate the activation scores
        expert_scores = torch.mean(expert_scores, dim=-1) # bs, seq_len, number_of_experts
        expert_topk_indices = torch.topk(expert_scores, k=self.config.topk, dim=-1).indices 
        expert_topk_mask = torch.zeros_like(expert_scores) # bs, seq_len, number_of_experts
        # only keep the top-k experts
        expert_topk_mask.scatter_(dim=-1, index=expert_topk_indices, value=1)
        expert_topk_mask = expert_topk_mask.repeat_interleave(
            self.config.intermediate_size // self.config.n_expert, dim=-1) # bs, seq_len, 8/3h
        # mask the intermediate states not belonging to the top-k experts
        intermediate_states = intermediate_states * expert_topk_mask
        down_proj = self.down_proj(intermediate_states)
        return down_proj



class MoELlamaDecoderLayer(LlamaDecoderLayer):
    
    def __init__(self, config: MoELlamaConfig, l_idx):
        super().__init__(config)
        if l_idx >= config.split_start_layer and (l_idx - config.split_start_layer) % config.split_every_layer == 0:
            self.mlp = MoELlamaMLP(config)
        
        
class MoELlamadModel(LlamaModel):
    
    def __init__(self, config: MoELlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MoELlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        
        
class MoELlamaForCausalLM(LlamaForCausalLM):
    
    def __init__(self, config):
        super().__init__(config)
        self.model = MoELlamadModel(config)