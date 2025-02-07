
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
import logging
import torch
from torch import nn

logger = logging.getLogger(__name__)


class CrossEncoderWEmbeds(nn.Module):
	"""
	Wrapper around BERT model which is used as a cross-encoder model.
	This first estimates contextualized embeddings for each input query-item pair
	and then computes score using dot-product of contextualized query and item.
	"""
	def __init__(self, model_name, config, **automodel_args):
		super(CrossEncoderWEmbeds, self).__init__()
		
		self.config = config
		# self.normalize_embeds = False
		# We need access to hidden_states to use contextualized representations
		# for computing final scores
		config.output_hidden_states = True

		self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, **automodel_args)
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		print("Loading Cross-Encoder model that uses dot-product of contextualized embeddings "
			  "for computing scores INSTEAD of a linear layer on top of CLS token embedding.")
		
		
	def forward(self, **features):
		
			
		input_ids = features["input_ids"]
		batch_size, seq_len = input_ids.shape
		output = self.model(**features)
	
		final_hidden = output.hidden_states[-1]
		
		all_cls_token_idxs = (input_ids == self.tokenizer.cls_token_id).nonzero()
		all_sep_token_idxs = (input_ids == self.tokenizer.sep_token_id).nonzero()
		
		assert 2*all_cls_token_idxs.shape[0] == all_sep_token_idxs.shape[0], f"all_cls_token_idxs.shape={all_cls_token_idxs.shape}, all_sep_token_idxs={all_sep_token_idxs.shape}, input_tokens={input_ids}"
		assert all_cls_token_idxs.shape[1] == all_sep_token_idxs.shape[1], f"all_cls_token_idxs.shape={all_cls_token_idxs.shape}, all_sep_token_idxs={all_sep_token_idxs.shape}"
		assert all_cls_token_idxs.shape[1] == 2, f"all_cls_token_idxs.shape={all_cls_token_idxs.shape}"
		
		# Query embedding correspond to [CLS] token embedding
		query_embed_tokens = all_cls_token_idxs
		
		# There are two [SEP] tokens in input_ids for each paired input and we want to extract indices of the first one
		passage_embed_tokens = all_sep_token_idxs[np.arange(0, all_sep_token_idxs.shape[0], 2)]
		
		assert (query_embed_tokens[:,0] == passage_embed_tokens[:,0]).all() # First col contains batch idxs so these should match
		
		query_embeds   = torch.stack([final_hidden[batch_idx, q_token_idx, :] for (batch_idx, q_token_idx) in query_embed_tokens])
		passage_embeds = torch.stack([final_hidden[batch_idx, p_token_idx, :] for (batch_idx, p_token_idx) in passage_embed_tokens])
		
		# if self.normalize_embeds:
		# query_embeds = torch.nn.functional.normalize(query_embeds,p=2, dim=1) # Normalize along embed_dim
		# passage_embeds = torch.nn.functional.normalize(passage_embeds,p=2, dim=1) # Normalize along embed_dim
		
		scores = torch.sum(query_embeds * passage_embeds, dim=1).unsqueeze(1)
		assert scores.shape == (batch_size, 1)
		
		# Update logits/scores
		output.logits = scores
		
		# print("logit scores w dot product", scores)
		##### To debug if we are correctly using hidden_states
		# final_hidden = output.hidden_states[-1]
		# default_pooled_output = self.model.bert.pooler(final_hidden)
		# logits = self.model.classifier(self.model.dropout(default_pooled_output))
		#
		# output_logits = output.logits
		#
		# print("comput_logits\n", logits)
		# print("output_logits\n", output_logits)
		#####
		
		# query_embeds = []
		# for (batch_idx, q_token_idx) in query_embed_tokens:
		# 	query_embeds += [final_hidden[batch_idx, q_token_idx, :]]
		# query_embeds = torch.stack(query_embeds)
		#
		# passage_embeds = []
		# for (batch_idx, p_token_idx) in passage_embed_tokens:
		# 	passage_embeds += [final_hidden[batch_idx, p_token_idx, :]]
		# passage_embeds = torch.stack(passage_embeds)

		# score_0 = query_embeds[0].T @ passage_embeds[0]
		# score_0 = query_embeds[10].T @ passage_embeds[10]
		# prod = query_embeds * passage_embeds
		# scores = torch.sum(query_embeds * passage_embeds, dim=1)
		# scores2 = torch.sum(torch.mul(query_embeds, passage_embeds), dim=1)
		# scores_np = np.sum(query_embeds.cpu().detach().numpy() * passage_embeds.cpu().detach().numpy(), axis=1)
		#
		# a = query_embeds[0]
		# b = passage_embeds[0]
		#
		#
		# c = a.cpu().detach().numpy()
		# d = b.cpu().detach().numpy()
		#
		# e = a.cpu().detach().numpy().astype(np.float16)
		# f = b.cpu().detach().numpy().astype(np.float16)
		#
		# aTb = a.T @ b
		# ab = a * b
		# cd = c * d
		# ef = e * f
		#
		# print(scores)
		# embed()
		return output
		
		
	
	def save_pretrained(self, save_directory):
		self.model.save_pretrained(save_directory)
	
	@property
	def device(self):
		return self.model.device
