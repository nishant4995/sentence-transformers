"""
This examples show how to train a Cross-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The query and the passage are passed simoultanously to a Transformer network. The network then returns
a score between 0 and 1 how relevant the passage is for a given query.

The resulting Cross-Encoder can then be used for passage re-ranking: You retrieve for example 100 passages
for a given query, for example with ElasticSearch, and pass the query+retrieved_passage to the CrossEncoder
for scoring. You sort the results then according to the output of the CrossEncoder.

This gives a significant boost compared to out-of-the-box ElasticSearch / BM25 ranking.

Running this script:
python train_cross-encoder.py
"""
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
import wandb
import torch
import numpy as np
import argparse
import random


def main(res_dir, seed, use_embed_ce_model, base_model_name, evaluation_steps, train_batch_size, lr):
	
	#### Just some code to print debug information to stdout
	logging.basicConfig(format='%(asctime)s - %(message)s',
						datefmt='%Y-%m-%d %H:%M:%S',
						level=logging.INFO,
						handlers=[LoggingHandler()])
	#### /print debug information to stdout

	#First, we define the transformer model we want to fine-tune
	model_name = base_model_name
	# train_batch_size = 32
	num_epochs = 1
	model_save_path = f'{res_dir}/output/training_ms-marco_cross-encoder-'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	
	
	# We train the network with as a binary label task
	# Given [query, passage] is the label 0 = irrelevant or 1 = relevant?
	# We use a positive-to-negative ratio: For 1 positive sample (label 1) we include 4 negative samples (label 0)
	# in our training setup. For the negative samples, we use the triplets provided by MS Marco that
	# specify (query, positive sample, negative sample).
	pos_neg_ration = 4
	
	# Maximal number of training samples we want to use
	max_train_samples = 2e7
	
	#We set num_labels=1, which predicts a continous score between 0 and 1
	model = CrossEncoder(model_name, num_labels=1, max_length=512, use_embed_ce_model=use_embed_ce_model)
	
	### Now we read the MS Marco dataset
	data_folder = 'msmarco-data'
	os.makedirs(data_folder, exist_ok=True)
	
	
	#### Read the corpus files, that contain all the passages. Store them in the corpus dict
	corpus = {}
	collection_filepath = os.path.join(data_folder, 'collection.tsv')
	if not os.path.exists(collection_filepath):
		tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
		if not os.path.exists(tar_filepath):
			logging.info("Download collection.tar.gz")
			util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)
	
		with tarfile.open(tar_filepath, "r:gz") as tar:
			tar.extractall(path=data_folder)
	
	with open(collection_filepath, 'r', encoding='utf8') as fIn:
		for line in fIn:
			pid, passage = line.strip().split("\t")
			corpus[pid] = passage
	
	
	### Read the train queries, store in queries dict
	queries = {}
	queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
	if not os.path.exists(queries_filepath):
		tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
		if not os.path.exists(tar_filepath):
			logging.info("Download queries.tar.gz")
			util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)
	
		with tarfile.open(tar_filepath, "r:gz") as tar:
			tar.extractall(path=data_folder)
	
	
	with open(queries_filepath, 'r', encoding='utf8') as fIn:
		for line in fIn:
			qid, query = line.strip().split("\t")
			queries[qid] = query
	
	
	
	### Now we create our training & dev data
	train_samples = []
	dev_samples = {}
	
	# We use 200 random queries from the train set for evaluation during training
	# Each query has at least one relevant and up to 200 irrelevant (negative) passages
	num_dev_queries = 200
	num_max_dev_negatives = 200
	
	# msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz and msmarco-qidpidtriples.rnd-shuf.train.tsv.gz is a randomly
	# shuffled version of qidpidtriples.train.full.2.tsv.gz from the MS Marco website
	# We extracted in the train-eval split 500 random queries that can be used for evaluation during training
	train_eval_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz')
	if not os.path.exists(train_eval_filepath):
		logging.info("Download "+os.path.basename(train_eval_filepath))
		util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz', train_eval_filepath)
	
	with gzip.open(train_eval_filepath, 'rt') as fIn:
		for line in fIn:
			qid, pos_id, neg_id = line.strip().split()
	
			if qid not in dev_samples and len(dev_samples) < num_dev_queries:
				dev_samples[qid] = {'query': queries[qid], 'positive': set(), 'negative': set()}
	
			if qid in dev_samples:
				dev_samples[qid]['positive'].add(corpus[pos_id])
	
				if len(dev_samples[qid]['negative']) < num_max_dev_negatives:
					dev_samples[qid]['negative'].add(corpus[neg_id])
	
	
	# Read our training file
	train_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train.tsv.gz')
	if not os.path.exists(train_filepath):
		logging.info("Download "+os.path.basename(train_filepath))
		util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz', train_filepath)
	
	cnt = 0
	with gzip.open(train_filepath, 'rt') as fIn:
		for line in tqdm.tqdm(fIn, unit_scale=True):
			qid, pos_id, neg_id = line.strip().split()
	
			if qid in dev_samples:
				continue
	
			query = queries[qid]
			if (cnt % (pos_neg_ration+1)) == 0:
				passage = corpus[pos_id]
				label = 1
			else:
				passage = corpus[neg_id]
				label = 0
	
			train_samples.append(InputExample(texts=[query, passage], label=label))
			cnt += 1
	
			if cnt >= max_train_samples:
				break
	
	# We create a DataLoader to load our train samples
	# train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
	# We create a DataLoader to load our train samples
	def seed_worker(worker_id):
		worker_seed = torch.initial_seed() % 2**32
		np.random.seed(worker_seed)
		random.seed(worker_seed)
		
	rng = torch.Generator()
	rng.manual_seed(seed)
	train_dataloader = DataLoader(
		train_samples,
		shuffle=True,
		batch_size=train_batch_size,
		drop_last=True,
		generator=rng,
		worker_init_fn=seed_worker
	)
	
	# We add an evaluator, which evaluates the performance during training
	# It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
	evaluator = CERerankingEvaluator(dev_samples, name='train-eval')
	
	# Configure the training
	warmup_steps = 5000
	logging.info("Warmup-steps: {}".format(warmup_steps))
	
	
	# Train the model
	model.fit(train_dataloader=train_dataloader,
			  evaluator=evaluator,
			  epochs=num_epochs,
			  evaluation_steps=evaluation_steps,
			  warmup_steps=warmup_steps,
			  output_path=model_save_path,
			  optimizer_params={'lr': lr},
			  use_amp=True)
	
	#Save latest model
	model.save(model_save_path+'-latest')


if __name__ == "__main__":
	
	
	parser = argparse.ArgumentParser( description='Train CE model')
	parser.add_argument("--base_model_name", type=str, default='nreimers/MiniLM-L6-H384-uncased',
						help="base model for finetuning. Some options are microsoft/MiniLM-L12-H384-uncased, nreimers/MiniLM-L6-H384-uncased, 'distilroberta-base'")
	parser.add_argument("--use_embed_ce_model", type=int, choices=[0, 1], required=True, help="0 - Use default CLS-token based pooling, and 1 - Use Embed based scoring for Cross-Encoder ")
	parser.add_argument("--res_dir", type=str, required=True, help="Base Res dir")
	# parser.add_argument("--loss_fnc_name", type=str, default='mse', help="Loss function to use")
	parser.add_argument("--evaluation_steps", type=int, default=5000, help="Model will be evauated after this number of steps")
	parser.add_argument("--train_batch_size", type=int, default=32, help="train_batch_size")
	parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
	parser.add_argument("--disable_wandb", type=int, choices=[0,1], default=0, help="1-Disable Wanbd, 0-Use wandb")
	parser.add_argument("--seed", type=int, default=0, help="Random seed")
	
	args = parser.parse_args()
	seed = args.seed
	use_embed_ce_model = args.use_embed_ce_model
	res_dir = args.res_dir
	base_model_name = args.base_model_name
	
	evaluation_steps = args.evaluation_steps
	train_batch_size = args.train_batch_size
	lr = args.lr
	disable_wandb = args.disable_wandb
	
	wandb.init(
		project="9_BEIR_CrossEnc",
		dir=res_dir,
		config=args.__dict__,
		mode="disabled" if disable_wandb else "online"
	)
	
	main(
		res_dir=res_dir,
		seed=seed,
		use_embed_ce_model=use_embed_ce_model,
		train_batch_size=train_batch_size,
		lr=lr,
		base_model_name=base_model_name,
		evaluation_steps=evaluation_steps,
	)
