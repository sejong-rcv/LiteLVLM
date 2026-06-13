import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    AutoTokenizer
)

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
        self.with_region = args.with_region

    def load_model(self, device_map=None):
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModelWithProjection.from_pretrained(
            self.vision_tower_name, device_map=device_map
        )
        self.vision_tower.requires_grad_(False)

        self.text_tokenizer = AutoTokenizer.from_pretrained(self.vision_tower_name)
        self.text_tower = CLIPTextModelWithProjection.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.text_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs, select_feature=None):
        if select_feature is None:
            select_feature = self.select_feature
            
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if select_feature == "patch":
            image_features = image_features[:, 1:]
        elif select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def register_attn_hooks(self):
        attn_outs = {}
        
        def _hook_q(module, input, output): 
            attn_outs["hq"] = output
        def _hook_k(module, input, output): 
            attn_outs["hk"] = output
        def _hook_v(module, input, output): 
            attn_outs["hv"] = output
            
        self_attn = self.vision_tower.vision_model.encoder.layers[-1].self_attn
        
        hq = self_attn.q_proj.register_forward_hook(_hook_q)
        hk = self_attn.k_proj.register_forward_hook(_hook_k)
        hv = self_attn.v_proj.register_forward_hook(_hook_v)
        return attn_outs, hq, hk, hv
        
    def ctx_aware_pruning(self, outs):
        cls_q = outs["hq"][:, :1, :] # [B, 1, D]
        img_k = outs["hk"][:, 1:, :] # [B, L-1, D]
        img_v = outs["hv"][:, 1:, :] # [B, L-1, D]
        
        _, _, dim = cls_q.shape
        
        q_norm = F.normalize(cls_q, dim=-1)
        k_norm = F.normalize(img_k, dim=-1)
        
        # scaled key-query dot-product attention
        attn_logits = torch.matmul(q_norm, k_norm.transpose(-2, -1)) / (dim ** 0.5)
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        attn_to_cls = attn_weights.transpose(1,2) * img_v 
        attn_toks = attn_to_cls.norm(dim=-1)
        
        _, ctx_tok_idxs = torch.sort(attn_toks.view(-1), descending=True)
        return ctx_tok_idxs
    
    def sim_aware_pruning(self, image_features, text_features, retain_tokens):
        with torch.cuda.amp.autocast(enabled=False):
            image_features = image_features.to(self.vision_tower.vision_model.post_layernorm.weight.dtype)
            layernorm_image_features = self.vision_tower.vision_model.post_layernorm(image_features)
            proj_image_features = self.vision_tower.visual_projection(layernorm_image_features)
            
        # compute visual-text similarities
        similarities = torch.matmul(
            proj_image_features, text_features.unsqueeze(-1)
        ).squeeze(-1) # [seq_len, D]
        
        num_adaptive_toks = retain_tokens // text_features.shape[0]
        _lowk_sim_tok_idxs = torch.topk(
            similarities, num_adaptive_toks, dim=-1, largest=False, sorted=True).indices
        lowk_sim_tok_idxs = torch.unique(_lowk_sim_tok_idxs.view(-1)).unsqueeze(0)
        return lowk_sim_tok_idxs
    
    @torch.no_grad()
    def forward(self, images, **kwargs):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else: # Inference batch size is always 1
            # LiteLVLM
            enc_img_stream = torch.cuda.Stream()
            enc_text_stream = torch.cuda.Stream()
            
            with torch.cuda.stream(enc_img_stream):
                attn_outs, hq, hk, hv = self.register_attn_hooks()
                try:
                    image_forward_outs = self.vision_tower(
                        images.to(device=self.device, dtype=self.dtype), output_hidden_states=True,
                    ) # image forward pass
                    image_features = self.feature_select(image_forward_outs, select_feature="cls_patch").to(images.dtype) 

                    # ranks context-aware tokens
                    ctx_tok_idxs = self.ctx_aware_pruning(attn_outs)
                finally: # prevent memory leaks
                    hq.remove()
                    hk.remove()
                    hv.remove()

            with torch.cuda.stream(enc_text_stream):
                questions = kwargs["questions_list"][0]
                input_text = self.text_tokenizer(
                    text=questions, return_tensors="pt", truncation=True, padding=True
                ).to(device=images.device) # text to token (input_ids)
                text_features = self.text_tower(**input_text, output_attentions=True)
                
                lowk_sim_tok_idxs = self.sim_aware_pruning(
                    image_features[:, 1:], text_features.text_embeds, retain_tokens=kwargs["num_retain_tokens"]
                )
            torch.cuda.synchronize()
            
            # adaptive token selection setup
            num_sim_topk = lowk_sim_tok_idxs.shape[1]
            m_ctx_tok_idxs = ~torch.isin(ctx_tok_idxs.view(-1), lowk_sim_tok_idxs.view(-1))
            l_ctx_tok_idxs = ctx_tok_idxs.view(-1)[m_ctx_tok_idxs].unsqueeze(0)
            
            num_left_tokens = kwargs["num_retain_tokens"] - num_sim_topk
            retain_tok_idxs = torch.cat(
                [l_ctx_tok_idxs[:, :num_left_tokens], lowk_sim_tok_idxs], dim=1
            ).to(torch.int64)
            
            assert retain_tok_idxs.shape[1] == kwargs["num_retain_tokens"]
            
            image_features = torch.gather(
                image_features[:, 1:], dim=1, index=retain_tok_idxs.unsqueeze(-1).expand(-1, -1, image_features.size(-1))
            )
        
        torch.cuda.empty_cache()
        if self.with_region:
            return image_features, image_forward_outs
        return image_features, None
        
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
