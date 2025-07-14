import torch
import torch.nn as nn
from einops import rearrange
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoConfig
import numpy as np

from utils.io_utils import load_json

class Latent_Token_predictor(nn.Module):
    def __init__(
        self,
        model_type,
        model_path,
        motion_codebook_size,
        **kwargs,
    ):
        super().__init__()
        self.max_length = 256
        self.m_codebook_size = motion_codebook_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
        if model_type == 't5':
            self.language_model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.model_config = AutoConfig.from_pretrained(model_path)
            self.lm_type = 'encdec'
        else:
            raise NotImplementedError(f"{model_type} not integrated")

        self.tokenizer.add_tokens(
            [f'[MID_{i}]' for i in range(self.m_codebook_size + 3)])
        self.tokenizer.add_tokens("[BETA]")
        
        self.motion_ids = []
        for i in range(self.m_codebook_size + 3):
            motion_id = self.tokenizer(f'[MID_{i}]', add_special_tokens=False).input_ids[0]
            if i == self.m_codebook_size:
                self.motion_start_id = motion_id
            if i == (self.m_codebook_size+1):
                self.motion_end_id = motion_id

            self.motion_ids.append(motion_id)

        self.beta_token_id = self.tokenizer("[BETA]", add_special_tokens=False).input_ids[0]
        
        self.language_model.resize_token_embeddings(len(self.tokenizer))
        in_dim = self.model_config.d_model
        out_dim = 10

        layer = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]

        self.projection = nn.Sequential(*layer)
        self.projection.train()
        for param in self.projection.parameters():
            param.requires_grad = True

    def generate_direct(self,
                    shape_texts, motion_texts, task, max_length: int = 256,
                    num_beams: int = 1,
                    do_sample: bool = True,
                    bad_words_ids = None):

        # Device
        self.device = self.language_model.device
        texts = shape_texts+ 'The person demonstrates a motion like '+motion_texts

        source_encoding = self.tokenizer(texts,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")
        
        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)

        outputs = self.language_model.generate(
            source_input_ids,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            bad_words_ids=bad_words_ids,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
        output_ids = outputs.sequences
        # outputs_string = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        motion_tokens = []
        if task == 't2sm' or task == 'mt2m' or task == 't2m':
            start_mask = (output_ids == self.motion_start_id)
            end_mask = (output_ids == self.motion_end_id)

            # the first index with motion_start_id/motion_end_id will be 1
            sum_start = start_mask.cumsum(1)
            sum_end = end_mask.flip(dims=[1]).cumsum(1).flip(dims=[1])

            has_start = sum_start[:, -1] > 0 
            has_end = sum_end[:, 0] > 0

            valid_tokens = (sum_start > 0) & (sum_end > 0) & has_start.unsqueeze(1) & has_end.unsqueeze(1)
            masked_idx = torch.where(valid_tokens, output_ids, torch.tensor(float('nan')))

            idx = 0
            for row in masked_idx:
                valid_values = row[torch.isnan(row) == False]
                if valid_values.numel() > 0:
                    motion_tokens.append(torch.tensor((valid_values-self.motion_ids[0]).tolist()[1:-1], dtype=int).to(self.device))
                else:
                    motion_tokens.append(torch.tensor([0], dtype=int).to(self.device))
                idx += 1

        last_layer_hidden_states = [state[-1] for state in outputs['decoder_hidden_states']]
        last_layer_hidden_states = torch.cat(last_layer_hidden_states, dim=1)

        feature = self.projection(last_layer_hidden_states)

        seg_token_mask = output_ids[:, 1:] == self.beta_token_id
        pred_embeddings = feature[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        pred_embeddings_ = []
        for i in range(len(seg_token_offset)-1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        return motion_tokens, pred_embeddings
    
