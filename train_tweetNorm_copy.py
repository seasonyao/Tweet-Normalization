#!/usr/bin/python
# -*- coding: UTF-8 -*-

#get_command_line_arguments
import sys
import getopt

#python get_command_line_arguments.py --username=xxx

argv = sys.argv[1:]
model_name = ""

opts, args = getopt.getopt(argv, "", ["model_name="])

for opt, arg in opts:
    if opt in ("--model_name"):
        model_name = arg     
#---------------------------------------------------------------------------------------------
# #bert2bert

# import nlp
# import logging
# from datasets import load_metric
# from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
# from datasets import Dataset
# import pandas as pd

# logging.basicConfig(level=logging.INFO)

# model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# # CLS token will work as BOS token
# tokenizer.bos_token = tokenizer.cls_token

# # SEP token will work as EOS token
# tokenizer.eos_token = tokenizer.sep_token

# # load train and validation data
# train_dataset = pd.read_json("WNUT2015_dataset/train_data.json", orient="records")
# val_dataset = pd.read_json("WNUT2015_dataset/test_truth.json", orient="records")

# make_sentence = lambda x : " ".join(x)

# train_dataset['input_sentence'] = train_dataset['input'].apply(make_sentence)
# train_dataset['output_sentence'] = train_dataset['output'].apply(make_sentence)
# val_dataset['input_sentence'] = val_dataset['input'].apply(make_sentence)
# val_dataset['output_sentence'] = val_dataset['output'].apply(make_sentence)

# train_dataset = Dataset.from_pandas(train_dataset)
# val_dataset = Dataset.from_pandas(val_dataset)

# # load rouge for validation
# rouge = load_metric('rouge', experiment_id=0)
# bleu = load_metric('bleu', experiment_id=0)

# # set decoding params
# model.config.decoder_start_token_id = tokenizer.bos_token_id
# model.config.eos_token_id = tokenizer.eos_token_id
# model.config.max_length = 64
# model.config.min_length = 0
# model.config.no_repeat_ngram_size = 3
# model.early_stopping = True
# model.length_penalty = 2.0
# model.num_beams = 4


# # map data correctly
# def map_to_encoder_decoder_inputs(batch):
#     # Tokenizer will automatically set [BOS] <text> [EOS]
#     # cut off at BERT max length 512
#     inputs = tokenizer(batch["input_sentence"], padding="max_length", truncation=True, max_length=64)
#     # inputs = tokenizer(batch["input_sentence"], padding="max_length")
#     # force summarization <= 128
#     outputs = tokenizer(batch["output_sentence"], padding="max_length", truncation=True, max_length=64)
#     # outputs = tokenizer(batch["output_sentence"], padding="max_length")

#     batch["input_ids"] = inputs.input_ids
#     batch["attention_mask"] = inputs.attention_mask

#     batch["decoder_input_ids"] = outputs.input_ids
#     batch["labels"] = outputs.input_ids.copy()
#     # mask loss for padding
#     batch["labels"] = [
#         [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
#     ]
#     batch["decoder_attention_mask"] = outputs.attention_mask

#     assert all([len(x) == 64 for x in inputs.input_ids])
#     assert all([len(x) == 64 for x in outputs.input_ids])
    
#     return batch


# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     # all unnecessary tokens are removed
#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
#     metrics_rouge = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1", "rouge2", "rouge3", "rouge4", "rougeL", "rougeLsum"])

#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [tokenizer.convert_ids_to_tokens(seq, **kwargs) for seq in sequences]
    
#     pred_tokens = batch_convert_ids_to_tokens(pred_ids, skip_special_tokens=True)
    
#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [[tokenizer.convert_ids_to_tokens(seq, **kwargs)] for seq in sequences]
    
#     label_tokens = batch_convert_ids_to_tokens(labels_ids, skip_special_tokens=True)
    
#     metrics_bleu = bleu.compute(predictions=pred_tokens, references=label_tokens)
    
#     return {
#         "rouge1_precision": round(metrics_rouge['rouge1'].mid.precision, 4),
#         "rouge1_recall": round(metrics_rouge['rouge1'].mid.recall, 4),
#         "rouge1_fmeasure": round(metrics_rouge['rouge1'].mid.fmeasure, 4),
#         "rouge2_precision": round(metrics_rouge['rouge2'].mid.precision, 4),
#         "rouge2_recall": round(metrics_rouge['rouge2'].mid.recall, 4),
#         "rouge2_fmeasure": round(metrics_rouge['rouge2'].mid.fmeasure, 4),
#         "rouge3_precision": round(metrics_rouge['rouge3'].mid.precision, 4),
#         "rouge3_recall": round(metrics_rouge['rouge3'].mid.recall, 4),
#         "rouge3_fmeasure": round(metrics_rouge['rouge3'].mid.fmeasure, 4),
#         "rouge4_precision": round(metrics_rouge['rouge4'].mid.precision, 4),
#         "rouge4_recall": round(metrics_rouge['rouge4'].mid.recall, 4),
#         "rouge4_fmeasure": round(metrics_rouge['rouge4'].mid.fmeasure, 4),
#         "rougeL_precision": round(metrics_rouge['rougeL'].mid.precision, 4),
#         "rougeL_recall": round(metrics_rouge['rougeL'].mid.recall, 4),
#         "rougeL_fmeasure": round(metrics_rouge['rougeL'].mid.fmeasure, 4),
#         "rougeLsum_precision": round(metrics_rouge['rougeLsum'].mid.precision, 4),
#         "rougeLsum_recall": round(metrics_rouge['rougeLsum'].mid.recall, 4),
#         "rougeLsum_fmeasure": round(metrics_rouge['rougeLsum'].mid.fmeasure, 4),
#         "bleu": round(metrics_bleu['bleu'], 4),
#     }

# # set batch size here
# batch_size = 16

# # make train dataset ready
# train_dataset = train_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# train_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# # same for validation dataset
# val_dataset = val_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# val_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# # set training arguments - these params are not really tuned, feel free to change
# training_args = TrainingArguments(
#     output_dir="./models/bert2bert/"+str(model_name)+"/",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     predict_from_generate=True,
#     evaluate_during_training=True,
#     do_train=True,
#     do_eval=True,
#     logging_steps=20,
#     save_steps=500,
#     eval_steps=300,
#     overwrite_output_dir=True,
#     warmup_steps=50,
#     save_total_limit=3,
#     num_train_epochs=30,
#     fp16=True,
#     run_name="bert2bert",
# )

# # instantiate trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )


# # start training
# trainer.train()



#---------------------------------------------------------------------------------------
#bert2bert_share

# import nlp
# from datasets import load_metric, Dataset
# import logging
# from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
# import pandas as pd

# logging.basicConfig(level=logging.INFO)

# model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "bert-base-cased", tie_encoder_decoder=True)
# tokenizer =  BertTokenizer.from_pretrained("bert-base-cased")

# # CLS token will work as BOS token
# tokenizer.bos_token = tokenizer.cls_token

# # SEP token will work as EOS token
# tokenizer.eos_token = tokenizer.sep_token

# # load train and validation data
# train_dataset = pd.read_json("WNUT2015_dataset/train_data.json", orient="records")
# val_dataset = pd.read_json("WNUT2015_dataset/test_truth.json", orient="records")

# make_sentence = lambda x : " ".join(x)

# train_dataset['input_sentence'] = train_dataset['input'].apply(make_sentence)
# train_dataset['output_sentence'] = train_dataset['output'].apply(make_sentence)
# val_dataset['input_sentence'] = val_dataset['input'].apply(make_sentence)
# val_dataset['output_sentence'] = val_dataset['output'].apply(make_sentence)

# train_dataset = Dataset.from_pandas(train_dataset)
# val_dataset = Dataset.from_pandas(val_dataset)

# # load rouge for validation
# rouge = load_metric("rouge", experiment_id=1)
# bleu = load_metric('bleu', experiment_id=1)

# # set decoding params
# model.config.decoder_start_token_id = tokenizer.bos_token_id
# model.config.eos_token_id = tokenizer.eos_token_id
# model.config.max_length = 64
# model.config.min_length = 0
# model.config.no_repeat_ngram_size = 3
# model.early_stopping = True
# model.length_penalty = 2.0
# model.num_beams = 4

# encoder_length = 64
# decoder_length = 64
# batch_size = 16


# # map data correctly
# def map_to_encoder_decoder_inputs(batch):
#     # Tokenizer will automatically set [BOS] <text> [EOS]
#     # cut off at Longformer at 2048
#     inputs = tokenizer(batch["input_sentence"], padding="max_length", truncation=True, max_length=encoder_length)
#     # force summarization <= 256
#     outputs = tokenizer(batch["output_sentence"], padding="max_length", truncation=True, max_length=decoder_length)

#     batch["input_ids"] = inputs.input_ids
#     batch["attention_mask"] = inputs.attention_mask
#     batch["decoder_input_ids"] = outputs.input_ids
#     batch["labels"] = outputs.input_ids.copy()
#     # mask loss for padding
#     batch["labels"] = [
#         [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
#     ]
#     batch["decoder_attention_mask"] = outputs.attention_mask

#     assert all([len(x) == encoder_length for x in inputs.input_ids])
#     assert all([len(x) == decoder_length for x in outputs.input_ids])

#     return batch


# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     # all unnecessary tokens are removed
#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     labels_ids[labels_ids == -100] = tokenizer.eos_token_id
#     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

#     metrics_rouge = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1", "rouge2", "rouge3", "rouge4", "rougeL", "rougeLsum"])

#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [tokenizer.convert_ids_to_tokens(seq, **kwargs) for seq in sequences]
    
#     pred_tokens = batch_convert_ids_to_tokens(pred_ids, skip_special_tokens=True)
    
#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [[tokenizer.convert_ids_to_tokens(seq, **kwargs)] for seq in sequences]
    
#     label_tokens = batch_convert_ids_to_tokens(labels_ids, skip_special_tokens=True)
    
#     metrics_bleu = bleu.compute(predictions=pred_tokens, references=label_tokens)
    
#     return {
#         "rouge1_precision": round(metrics_rouge['rouge1'].mid.precision, 4),
#         "rouge1_recall": round(metrics_rouge['rouge1'].mid.recall, 4),
#         "rouge1_fmeasure": round(metrics_rouge['rouge1'].mid.fmeasure, 4),
#         "rouge2_precision": round(metrics_rouge['rouge2'].mid.precision, 4),
#         "rouge2_recall": round(metrics_rouge['rouge2'].mid.recall, 4),
#         "rouge2_fmeasure": round(metrics_rouge['rouge2'].mid.fmeasure, 4),
#         "rouge3_precision": round(metrics_rouge['rouge3'].mid.precision, 4),
#         "rouge3_recall": round(metrics_rouge['rouge3'].mid.recall, 4),
#         "rouge3_fmeasure": round(metrics_rouge['rouge3'].mid.fmeasure, 4),
#         "rouge4_precision": round(metrics_rouge['rouge4'].mid.precision, 4),
#         "rouge4_recall": round(metrics_rouge['rouge4'].mid.recall, 4),
#         "rouge4_fmeasure": round(metrics_rouge['rouge4'].mid.fmeasure, 4),
#         "rougeL_precision": round(metrics_rouge['rougeL'].mid.precision, 4),
#         "rougeL_recall": round(metrics_rouge['rougeL'].mid.recall, 4),
#         "rougeL_fmeasure": round(metrics_rouge['rougeL'].mid.fmeasure, 4),
#         "rougeLsum_precision": round(metrics_rouge['rougeLsum'].mid.precision, 4),
#         "rougeLsum_recall": round(metrics_rouge['rougeLsum'].mid.recall, 4),
#         "rougeLsum_fmeasure": round(metrics_rouge['rougeLsum'].mid.fmeasure, 4),
#         "bleu": round(metrics_bleu['bleu'], 4),
#     }

# # make train dataset ready
# train_dataset = train_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# train_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_attention_mask", "decoder_input_ids", "labels"],
# )

# # same for validation dataset
# val_dataset = val_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# val_dataset.set_format(
#     type="torch", columns=["input_ids", "decoder_attention_mask", "attention_mask", "decoder_input_ids", "labels"],
# )

# # set training arguments - these params are not really tuned, feel free to change
# training_args = TrainingArguments(
#     output_dir="./models/bert2bert_share/"+str(model_name)+"/",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     predict_from_generate=True,
#     evaluate_during_training=True,
#     do_train=True,
#     do_eval=True,
#     logging_steps=20,
#     save_steps=500,
#     eval_steps=300,
#     overwrite_output_dir=True,
#     warmup_steps=50,
#     save_total_limit=3,
#     fp16=True,
#     num_train_epochs=30,
#     run_name="bert2bert_share",
# )

# # instantiate trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# # start training
# trainer.train()



#---------------------------------------------------------------------------------------
#bert2gpt2

# import nlp
# from datasets import load_metric, Dataset
# import logging
# from transformers import BertTokenizer, GPT2Tokenizer, EncoderDecoderModel, Trainer, TrainingArguments
# import pandas as pd

# logging.basicConfig(level=logging.INFO)

# model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "gpt2")
# # cache is currently not supported by EncoderDecoder framework
# model.decoder.config.use_cache = False
# bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# # CLS token will work as BOS token
# bert_tokenizer.bos_token = bert_tokenizer.cls_token

# # SEP token will work as EOS token
# bert_tokenizer.eos_token = bert_tokenizer.sep_token


# # make sure GPT2 appends EOS in begin and end
# def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
#     outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
#     return outputs


# GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
# gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
# gpt2_tokenizer.pad_token = gpt2_tokenizer.unk_token


# # set decoding params
# model.config.decoder_start_token_id = gpt2_tokenizer.bos_token_id
# model.config.eos_token_id = gpt2_tokenizer.eos_token_id
# model.config.max_length = 64
# model.config.min_length = 0
# model.config.no_repeat_ngram_size = 3
# model.early_stopping = True
# model.length_penalty = 2.0
# model.num_beams = 4

# # load train and validation data
# train_dataset = pd.read_json("WNUT2015_dataset/train_data.json", orient="records")
# val_dataset = pd.read_json("WNUT2015_dataset/test_truth.json", orient="records")

# make_sentence = lambda x : " ".join(x)

# train_dataset['input_sentence'] = train_dataset['input'].apply(make_sentence)
# train_dataset['output_sentence'] = train_dataset['output'].apply(make_sentence)
# val_dataset['input_sentence'] = val_dataset['input'].apply(make_sentence)
# val_dataset['output_sentence'] = val_dataset['output'].apply(make_sentence)

# train_dataset = Dataset.from_pandas(train_dataset)
# val_dataset = Dataset.from_pandas(val_dataset)

# # load rouge for validation
# rouge = load_metric('rouge', experiment_id=2)
# bleu = load_metric('bleu', experiment_id=2)

# encoder_length = 64
# decoder_length = 64
# batch_size = 16


# # map data correctly
# def map_to_encoder_decoder_inputs(batch):    # Tokenizer will automatically set [BOS] <text> [EOS] 
#     # use bert tokenizer here for encoder
#     inputs = bert_tokenizer(batch["input_sentence"], padding="max_length", truncation=True, max_length=encoder_length)
#     # force summarization <= 128
#     outputs = gpt2_tokenizer(batch["output_sentence"], padding="max_length", truncation=True, max_length=decoder_length)

#     batch["input_ids"] = inputs.input_ids
#     batch["attention_mask"] = inputs.attention_mask
#     batch["decoder_input_ids"] = outputs.input_ids
#     batch["labels"] = outputs.input_ids.copy()
#     batch["decoder_attention_mask"] = outputs.attention_mask

#     # complicated list comprehension here because pad_token_id alone is not good enough to know whether label should be excluded or not
#     batch["labels"] = [
#         [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(batch["decoder_attention_mask"], batch["labels"])]
#     ]
#     # mask loss for padding
# #     batch["labels"] = [
# #         [-100 if token == gpt2_tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
# #     ]

#     assert all([len(x) == encoder_length for x in inputs.input_ids])
#     assert all([len(x) == decoder_length for x in outputs.input_ids])

#     return batch


# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     # all unnecessary tokens are removed
#     pred_str = gpt2_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     labels_ids[labels_ids == -100] = gpt2_tokenizer.eos_token_id
#     label_str = gpt2_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

#     metrics_rouge = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1", "rouge2", "rouge3", "rouge4", "rougeL", "rougeLsum"])

#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [gpt2_tokenizer.convert_ids_to_tokens(seq, **kwargs) for seq in sequences]
    
#     pred_tokens = batch_convert_ids_to_tokens(pred_ids, skip_special_tokens=True)
    
#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [[gpt2_tokenizer.convert_ids_to_tokens(seq, **kwargs)] for seq in sequences]
    
#     label_tokens = batch_convert_ids_to_tokens(labels_ids, skip_special_tokens=True)
    
#     metrics_bleu = bleu.compute(predictions=pred_tokens, references=label_tokens)
    
#     return {
#         "rouge1_precision": round(metrics_rouge['rouge1'].mid.precision, 4),
#         "rouge1_recall": round(metrics_rouge['rouge1'].mid.recall, 4),
#         "rouge1_fmeasure": round(metrics_rouge['rouge1'].mid.fmeasure, 4),
#         "rouge2_precision": round(metrics_rouge['rouge2'].mid.precision, 4),
#         "rouge2_recall": round(metrics_rouge['rouge2'].mid.recall, 4),
#         "rouge2_fmeasure": round(metrics_rouge['rouge2'].mid.fmeasure, 4),
#         "rouge3_precision": round(metrics_rouge['rouge3'].mid.precision, 4),
#         "rouge3_recall": round(metrics_rouge['rouge3'].mid.recall, 4),
#         "rouge3_fmeasure": round(metrics_rouge['rouge3'].mid.fmeasure, 4),
#         "rouge4_precision": round(metrics_rouge['rouge4'].mid.precision, 4),
#         "rouge4_recall": round(metrics_rouge['rouge4'].mid.recall, 4),
#         "rouge4_fmeasure": round(metrics_rouge['rouge4'].mid.fmeasure, 4),
#         "rougeL_precision": round(metrics_rouge['rougeL'].mid.precision, 4),
#         "rougeL_recall": round(metrics_rouge['rougeL'].mid.recall, 4),
#         "rougeL_fmeasure": round(metrics_rouge['rougeL'].mid.fmeasure, 4),
#         "rougeLsum_precision": round(metrics_rouge['rougeLsum'].mid.precision, 4),
#         "rougeLsum_recall": round(metrics_rouge['rougeLsum'].mid.recall, 4),
#         "rougeLsum_fmeasure": round(metrics_rouge['rougeLsum'].mid.fmeasure, 4),
#         "bleu": round(metrics_bleu['bleu'], 4),
#     }


# # make train dataset ready
# train_dataset = train_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# train_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# # same for validation dataset
# val_dataset = val_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# val_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# # set training arguments - these params are not really tuned, feel free to change
# training_args = TrainingArguments(
#     output_dir="./models/bert2gpt2/"+str(model_name)+"/",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     predict_from_generate=True,
#     evaluate_during_training=True,
#     do_train=True,
#     do_eval=True,
#     logging_steps=20,
#     save_steps=500,
#     eval_steps=300,
#     overwrite_output_dir=True,
#     warmup_steps=50,
#     save_total_limit=3,
#     fp16=True,
#     num_train_epochs=30,
#     run_name="bert2gpt2",
# )

# # instantiate trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# # start training
# trainer.train()



#--------------------------------------------------------------------------------
#roberta2roberta

# import nlp
# from datasets import load_metric, Dataset
# import logging
# from transformers import RobertaTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
# import pandas as pd

# logging.basicConfig(level=logging.INFO)

# model = EncoderDecoderModel.from_encoder_decoder_pretrained("roberta-base", "roberta-base")
# tokenizer =  RobertaTokenizer.from_pretrained("roberta-base")

# # load train and validation data
# train_dataset = pd.read_json("WNUT2015_dataset/train_data.json", orient="records")
# val_dataset = pd.read_json("WNUT2015_dataset/test_truth.json", orient="records")


# make_sentence = lambda x : " ".join(x)

# train_dataset['input_sentence'] = train_dataset['input'].apply(make_sentence)
# train_dataset['output_sentence'] = train_dataset['output'].apply(make_sentence)
# val_dataset['input_sentence'] = val_dataset['input'].apply(make_sentence)
# val_dataset['output_sentence'] = val_dataset['output'].apply(make_sentence)

# train_dataset = Dataset.from_pandas(train_dataset)
# val_dataset = Dataset.from_pandas(val_dataset)

# # load rouge for validation
# rouge = load_metric("rouge", experiment_id=3)
# bleu = load_metric('bleu', experiment_id=3)

# # set decoding params
# model.config.decoder_start_token_id = tokenizer.bos_token_id
# model.config.eos_token_id = tokenizer.eos_token_id
# model.config.max_length = 64
# model.config.min_length = 0
# model.config.no_repeat_ngram_size = 3
# model.early_stopping = True
# model.length_penalty = 2.0
# model.num_beams = 4

# encoder_length = 64
# decoder_length = 64
# batch_size = 16


# # map data correctly
# def map_to_encoder_decoder_inputs(batch):
#     # Tokenizer will automatically set [BOS] <text> [EOS]
#     # cut off at Longformer at 2048
#     inputs = tokenizer(batch["input_sentence"], padding="max_length", truncation=True, max_length=encoder_length)
#     # force summarization <= 256
#     outputs = tokenizer(batch["output_sentence"], padding="max_length", truncation=True, max_length=decoder_length)
    

#     batch["input_ids"] = inputs.input_ids
#     batch["attention_mask"] = inputs.attention_mask
#     batch["decoder_input_ids"] = outputs.input_ids
#     batch["labels"] = outputs.input_ids.copy()
#     # mask loss for padding
#     batch["labels"] = [
#         [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
#     ]
#     batch["decoder_attention_mask"] = outputs.attention_mask

#     assert all([len(x) == encoder_length for x in inputs.input_ids])
#     assert all([len(x) == decoder_length for x in outputs.input_ids])

#     return batch


# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     # all unnecessary tokens are removed
#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     labels_ids[labels_ids == -100] = tokenizer.eos_token_id
#     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

#     metrics_rouge = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1", "rouge2", "rouge3", "rouge4", "rougeL", "rougeLsum"])

#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [tokenizer.convert_ids_to_tokens(seq, **kwargs) for seq in sequences]
    
#     pred_tokens = batch_convert_ids_to_tokens(pred_ids, skip_special_tokens=True)
    
#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [[tokenizer.convert_ids_to_tokens(seq, **kwargs)] for seq in sequences]
    
#     label_tokens = batch_convert_ids_to_tokens(labels_ids, skip_special_tokens=True)
    
#     metrics_bleu = bleu.compute(predictions=pred_tokens, references=label_tokens)
    
#     return {
#         "rouge1_precision": round(metrics_rouge['rouge1'].mid.precision, 4),
#         "rouge1_recall": round(metrics_rouge['rouge1'].mid.recall, 4),
#         "rouge1_fmeasure": round(metrics_rouge['rouge1'].mid.fmeasure, 4),
#         "rouge2_precision": round(metrics_rouge['rouge2'].mid.precision, 4),
#         "rouge2_recall": round(metrics_rouge['rouge2'].mid.recall, 4),
#         "rouge2_fmeasure": round(metrics_rouge['rouge2'].mid.fmeasure, 4),
#         "rouge3_precision": round(metrics_rouge['rouge3'].mid.precision, 4),
#         "rouge3_recall": round(metrics_rouge['rouge3'].mid.recall, 4),
#         "rouge3_fmeasure": round(metrics_rouge['rouge3'].mid.fmeasure, 4),
#         "rouge4_precision": round(metrics_rouge['rouge4'].mid.precision, 4),
#         "rouge4_recall": round(metrics_rouge['rouge4'].mid.recall, 4),
#         "rouge4_fmeasure": round(metrics_rouge['rouge4'].mid.fmeasure, 4),
#         "rougeL_precision": round(metrics_rouge['rougeL'].mid.precision, 4),
#         "rougeL_recall": round(metrics_rouge['rougeL'].mid.recall, 4),
#         "rougeL_fmeasure": round(metrics_rouge['rougeL'].mid.fmeasure, 4),
#         "rougeLsum_precision": round(metrics_rouge['rougeLsum'].mid.precision, 4),
#         "rougeLsum_recall": round(metrics_rouge['rougeLsum'].mid.recall, 4),
#         "rougeLsum_fmeasure": round(metrics_rouge['rougeLsum'].mid.fmeasure, 4),
#         "bleu": round(metrics_bleu['bleu'], 4),
#     }


# # make train dataset ready
# train_dataset = train_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# train_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_attention_mask", "decoder_input_ids", "labels"],
# )

# # same for validation dataset
# val_dataset = val_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# val_dataset.set_format(
#     type="torch", columns=["input_ids", "decoder_attention_mask", "attention_mask", "decoder_input_ids", "labels"],
# )

# # set training arguments - these params are not really tuned, feel free to change
# training_args = TrainingArguments(
#     output_dir="./models/roberta2roberta/"+str(model_name)+"/",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     predict_from_generate=True,
#     evaluate_during_training=True,
#     do_train=True,
#     do_eval=True,
#     logging_steps=20,
#     save_steps=500,
#     eval_steps=300,
#     overwrite_output_dir=True,
#     warmup_steps=50,
#     save_total_limit=3,
#     fp16=True,
#     num_train_epochs=30,
#     run_name="roberta2roberta",
# )


# # instantiate trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# # start training
# trainer.train()


#------------------------------------------------------------------------------
#roberta2roberta_share

# import nlp
# from datasets import load_metric, Dataset
# import logging
# from transformers import RobertaTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
# import pandas as pd

# logging.basicConfig(level=logging.INFO)

# model = EncoderDecoderModel.from_encoder_decoder_pretrained("roberta-base", "roberta-base", tie_encoder_decoder=True)
# tokenizer =  RobertaTokenizer.from_pretrained("roberta-base")

# # load train and validation data
# train_dataset = pd.read_json("WNUT2015_dataset/train_data.json", orient="records")
# val_dataset = pd.read_json("WNUT2015_dataset/test_truth.json", orient="records")

# make_sentence = lambda x : " ".join(x)

# train_dataset['input_sentence'] = train_dataset['input'].apply(make_sentence)
# train_dataset['output_sentence'] = train_dataset['output'].apply(make_sentence)
# val_dataset['input_sentence'] = val_dataset['input'].apply(make_sentence)
# val_dataset['output_sentence'] = val_dataset['output'].apply(make_sentence)

# train_dataset = Dataset.from_pandas(train_dataset)
# val_dataset = Dataset.from_pandas(val_dataset)

# # load rouge for validation
# rouge = load_metric("rouge", experiment_id=4)
# bleu = load_metric('bleu', experiment_id=4)

# # set decoding params
# model.config.decoder_start_token_id = tokenizer.bos_token_id
# model.config.eos_token_id = tokenizer.eos_token_id
# model.config.max_length = 64
# model.config.min_length = 0
# model.config.no_repeat_ngram_size = 3
# model.early_stopping = True
# model.length_penalty = 2.0
# model.num_beams = 4

# encoder_length = 64
# decoder_length = 64
# batch_size = 16


# # map data correctly
# def map_to_encoder_decoder_inputs(batch):
#     # Tokenizer will automatically set [BOS] <text> [EOS]
#     # cut off at Longformer at 2048
#     inputs = tokenizer(batch["input_sentence"], padding="max_length", truncation=True, max_length=encoder_length)
#     # force summarization <= 256
#     outputs = tokenizer(batch["output_sentence"], padding="max_length", truncation=True, max_length=decoder_length)

#     batch["input_ids"] = inputs.input_ids
#     batch["attention_mask"] = inputs.attention_mask
#     batch["decoder_input_ids"] = outputs.input_ids
#     batch["labels"] = outputs.input_ids.copy()
#     # mask loss for padding
#     batch["labels"] = [
#         [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
#     ]
#     batch["decoder_attention_mask"] = outputs.attention_mask

#     assert all([len(x) == encoder_length for x in inputs.input_ids])
#     assert all([len(x) == decoder_length for x in outputs.input_ids])

#     return batch


# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     # all unnecessary tokens are removed
#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     labels_ids[labels_ids == -100] = tokenizer.eos_token_id
#     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

#     metrics_rouge = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1", "rouge2", "rouge3", "rouge4", "rougeL", "rougeLsum"])

#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [tokenizer.convert_ids_to_tokens(seq, **kwargs) for seq in sequences]
    
#     pred_tokens = batch_convert_ids_to_tokens(pred_ids, skip_special_tokens=True)
    
#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [[tokenizer.convert_ids_to_tokens(seq, **kwargs)] for seq in sequences]
    
#     label_tokens = batch_convert_ids_to_tokens(labels_ids, skip_special_tokens=True)
    
#     metrics_bleu = bleu.compute(predictions=pred_tokens, references=label_tokens)
    
#     return {
#         "rouge1_precision": round(metrics_rouge['rouge1'].mid.precision, 4),
#         "rouge1_recall": round(metrics_rouge['rouge1'].mid.recall, 4),
#         "rouge1_fmeasure": round(metrics_rouge['rouge1'].mid.fmeasure, 4),
#         "rouge2_precision": round(metrics_rouge['rouge2'].mid.precision, 4),
#         "rouge2_recall": round(metrics_rouge['rouge2'].mid.recall, 4),
#         "rouge2_fmeasure": round(metrics_rouge['rouge2'].mid.fmeasure, 4),
#         "rouge3_precision": round(metrics_rouge['rouge3'].mid.precision, 4),
#         "rouge3_recall": round(metrics_rouge['rouge3'].mid.recall, 4),
#         "rouge3_fmeasure": round(metrics_rouge['rouge3'].mid.fmeasure, 4),
#         "rouge4_precision": round(metrics_rouge['rouge4'].mid.precision, 4),
#         "rouge4_recall": round(metrics_rouge['rouge4'].mid.recall, 4),
#         "rouge4_fmeasure": round(metrics_rouge['rouge4'].mid.fmeasure, 4),
#         "rougeL_precision": round(metrics_rouge['rougeL'].mid.precision, 4),
#         "rougeL_recall": round(metrics_rouge['rougeL'].mid.recall, 4),
#         "rougeL_fmeasure": round(metrics_rouge['rougeL'].mid.fmeasure, 4),
#         "rougeLsum_precision": round(metrics_rouge['rougeLsum'].mid.precision, 4),
#         "rougeLsum_recall": round(metrics_rouge['rougeLsum'].mid.recall, 4),
#         "rougeLsum_fmeasure": round(metrics_rouge['rougeLsum'].mid.fmeasure, 4),
#         "bleu": round(metrics_bleu['bleu'], 4),
#     }



# # make train dataset ready
# train_dataset = train_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# train_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_attention_mask", "decoder_input_ids", "labels"],
# )

# # same for validation dataset
# val_dataset = val_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# val_dataset.set_format(
#     type="torch", columns=["input_ids", "decoder_attention_mask", "attention_mask", "decoder_input_ids", "labels"],
# )

# # set training arguments - these params are not really tuned, feel free to change
# training_args = TrainingArguments(
#     output_dir="./models/roberta2roberta_share/"+str(model_name)+"/",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     predict_from_generate=True,
#     evaluate_during_training=True,
#     do_train=True,
#     do_eval=True,
#     logging_steps=20,
#     save_steps=500,
#     eval_steps=300,
#     overwrite_output_dir=True,
#     warmup_steps=50,
#     save_total_limit=3,
#     fp16=True,
#     num_train_epochs=30,
#     run_name="roberta2roberta_share",
# )

# # instantiate trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# # start training
# trainer.train()




#-----------------------------------------------------------------------------------
# #roberta2gpt2

# import nlp
# from datasets import load_metric, Dataset
# import logging
# from transformers import RobertaTokenizer, GPT2Tokenizer, EncoderDecoderModel, Trainer, TrainingArguments
# import pandas as pd

# logging.basicConfig(level=logging.INFO)

# model = EncoderDecoderModel.from_encoder_decoder_pretrained("roberta-base", "gpt2")
# # cache is currently not supported by EncoderDecoder framework
# model.decoder.config.use_cache = False
# roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# # make sure GPT2 appends EOS in begin and end
# def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
#     outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
#     return outputs


# GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
# gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
# gpt2_tokenizer.pad_token = gpt2_tokenizer.unk_token


# # set decoding params
# model.config.decoder_start_token_id = gpt2_tokenizer.bos_token_id
# model.config.eos_token_id = gpt2_tokenizer.eos_token_id
# model.config.max_length = 64
# model.config.min_length = 0
# model.config.no_repeat_ngram_size = 3
# model.early_stopping = True
# model.length_penalty = 2.0
# model.num_beams = 4

# # load train and validation data
# train_dataset = pd.read_json("WNUT2015_dataset/train_data.json", orient="records")
# val_dataset = pd.read_json("WNUT2015_dataset/test_truth.json", orient="records")

# make_sentence = lambda x : " ".join(x)

# train_dataset['input_sentence'] = train_dataset['input'].apply(make_sentence)
# train_dataset['output_sentence'] = train_dataset['output'].apply(make_sentence)
# val_dataset['input_sentence'] = val_dataset['input'].apply(make_sentence)
# val_dataset['output_sentence'] = val_dataset['output'].apply(make_sentence)

# train_dataset = Dataset.from_pandas(train_dataset)
# val_dataset = Dataset.from_pandas(val_dataset)

# # load rouge for validation
# rouge = load_metric('rouge', experiment_id=5)
# bleu = load_metric('bleu', experiment_id=5)

# encoder_length = 64
# decoder_length = 64
# batch_size = 16


# # map data correctly
# def map_to_encoder_decoder_inputs(batch):    # Tokenizer will automatically set [BOS] <text> [EOS] 
#     # use bert tokenizer here for encoder
#     inputs = roberta_tokenizer(batch["input_sentence"], padding="max_length", truncation=True, max_length=encoder_length)
#     # force summarization <= 128
#     outputs = gpt2_tokenizer(batch["output_sentence"], padding="max_length", truncation=True, max_length=decoder_length)

#     batch["input_ids"] = inputs.input_ids
#     batch["attention_mask"] = inputs.attention_mask
#     batch["decoder_input_ids"] = outputs.input_ids
#     batch["labels"] = outputs.input_ids.copy()
#     batch["decoder_attention_mask"] = outputs.attention_mask

#     # complicated list comprehension here because pad_token_id alone is not good enough to know whether label should be excluded or not
#     batch["labels"] = [
#         [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(batch["decoder_attention_mask"], batch["labels"])]
#     ]

#     assert all([len(x) == encoder_length for x in inputs.input_ids])
#     assert all([len(x) == decoder_length for x in outputs.input_ids])

#     return batch


# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     # all unnecessary tokens are removed
#     pred_str = gpt2_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     labels_ids[labels_ids == -100] = gpt2_tokenizer.eos_token_id
#     label_str = gpt2_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

#     metrics_rouge = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1", "rouge2", "rouge3", "rouge4", "rougeL", "rougeLsum"])

#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [gpt2_tokenizer.convert_ids_to_tokens(seq, **kwargs) for seq in sequences]
    
#     pred_tokens = batch_convert_ids_to_tokens(pred_ids, skip_special_tokens=True)
    
#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [[gpt2_tokenizer.convert_ids_to_tokens(seq, **kwargs)] for seq in sequences]
    
#     label_tokens = batch_convert_ids_to_tokens(labels_ids, skip_special_tokens=True)
    
#     metrics_bleu = bleu.compute(predictions=pred_tokens, references=label_tokens)
    
#     return {
#         "rouge1_precision": round(metrics_rouge['rouge1'].mid.precision, 4),
#         "rouge1_recall": round(metrics_rouge['rouge1'].mid.recall, 4),
#         "rouge1_fmeasure": round(metrics_rouge['rouge1'].mid.fmeasure, 4),
#         "rouge2_precision": round(metrics_rouge['rouge2'].mid.precision, 4),
#         "rouge2_recall": round(metrics_rouge['rouge2'].mid.recall, 4),
#         "rouge2_fmeasure": round(metrics_rouge['rouge2'].mid.fmeasure, 4),
#         "rouge3_precision": round(metrics_rouge['rouge3'].mid.precision, 4),
#         "rouge3_recall": round(metrics_rouge['rouge3'].mid.recall, 4),
#         "rouge3_fmeasure": round(metrics_rouge['rouge3'].mid.fmeasure, 4),
#         "rouge4_precision": round(metrics_rouge['rouge4'].mid.precision, 4),
#         "rouge4_recall": round(metrics_rouge['rouge4'].mid.recall, 4),
#         "rouge4_fmeasure": round(metrics_rouge['rouge4'].mid.fmeasure, 4),
#         "rougeL_precision": round(metrics_rouge['rougeL'].mid.precision, 4),
#         "rougeL_recall": round(metrics_rouge['rougeL'].mid.recall, 4),
#         "rougeL_fmeasure": round(metrics_rouge['rougeL'].mid.fmeasure, 4),
#         "rougeLsum_precision": round(metrics_rouge['rougeLsum'].mid.precision, 4),
#         "rougeLsum_recall": round(metrics_rouge['rougeLsum'].mid.recall, 4),
#         "rougeLsum_fmeasure": round(metrics_rouge['rougeLsum'].mid.fmeasure, 4),
#         "bleu": round(metrics_bleu['bleu'], 4),
#     }



# # make train dataset ready
# train_dataset = train_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# train_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# # same for validation dataset
# val_dataset = val_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# val_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# # set training arguments - these params are not really tuned, feel free to change
# training_args = TrainingArguments(
#     output_dir="./models/roberta2gpt2/"+str(model_name)+"/",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     predict_from_generate=True,
#     evaluate_during_training=True,
#     do_train=True,
#     do_eval=True,
#     logging_steps=20,
#     save_steps=500,
#     eval_steps=300,
#     overwrite_output_dir=True,
#     warmup_steps=50,
#     save_total_limit=3,
#     fp16=True,
#     num_train_epochs=30,
#     run_name="roberta2gpt2",
# )

# # instantiate trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# # start training
# trainer.train()



#-----------------------------------------------------------------------------------
#bertweet2gpt2

# import nlp
# from datasets import load_metric, Dataset
# import logging
# from transformers import GPT2Tokenizer, AutoTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
# import pandas as pd

# logging.basicConfig(level=logging.INFO)

# model = EncoderDecoderModel.from_encoder_decoder_pretrained("vinai/bertweet-base", "gpt2")
# # cache is currently not supported by EncoderDecoder framework
# model.decoder.config.use_cache = False
# bertweet_tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

# # make sure GPT2 appends EOS in begin and end
# def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
#     outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
#     return outputs

# GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
# gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
# gpt2_tokenizer.pad_token = gpt2_tokenizer.unk_token


# # set decoding params
# model.config.decoder_start_token_id = gpt2_tokenizer.bos_token_id
# model.config.eos_token_id = gpt2_tokenizer.eos_token_id
# model.config.max_length = 64
# model.config.min_length = 0
# model.config.no_repeat_ngram_size = 3
# model.early_stopping = True
# model.length_penalty = 2.0
# model.num_beams = 4

# # load train and validation data
# train_dataset = pd.read_json("WNUT2015_dataset/train_data.json", orient="records")
# val_dataset = pd.read_json("WNUT2015_dataset/test_truth.json", orient="records")

# make_sentence = lambda x : " ".join(x)

# train_dataset['input_sentence'] = train_dataset['input'].apply(make_sentence)
# train_dataset['output_sentence'] = train_dataset['output'].apply(make_sentence)
# val_dataset['input_sentence'] = val_dataset['input'].apply(make_sentence)
# val_dataset['output_sentence'] = val_dataset['output'].apply(make_sentence)

# train_dataset = Dataset.from_pandas(train_dataset)
# val_dataset = Dataset.from_pandas(val_dataset)

# # load rouge for validation
# rouge = load_metric('rouge', experiment_id=6)
# bleu = load_metric('bleu', experiment_id=6)

# encoder_length = 64
# decoder_length = 64
# batch_size = 16


# # map data correctly
# def map_to_encoder_decoder_inputs(batch):    # Tokenizer will automatically set [BOS] <text> [EOS] 
#     # use bert tokenizer here for encoder
#     inputs = bertweet_tokenizer(batch["input_sentence"], padding="max_length", truncation=True, max_length=encoder_length)
#     # force summarization <= 128
#     outputs = gpt2_tokenizer(batch["output_sentence"], padding="max_length", truncation=True, max_length=decoder_length)

#     batch["input_ids"] = inputs.input_ids
#     batch["attention_mask"] = inputs.attention_mask
#     batch["decoder_input_ids"] = outputs.input_ids
#     batch["labels"] = outputs.input_ids.copy()
#     batch["decoder_attention_mask"] = outputs.attention_mask

#     # complicated list comprehension here because pad_token_id alone is not good enough to know whether label should be excluded or not
#     batch["labels"] = [
#         [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(batch["decoder_attention_mask"], batch["labels"])]
#     ]

#     assert all([len(x) == encoder_length for x in inputs.input_ids])
#     assert all([len(x) == decoder_length for x in outputs.input_ids])

#     return batch


# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     # all unnecessary tokens are removed
#     pred_str = gpt2_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     labels_ids[labels_ids == -100] = gpt2_tokenizer.eos_token_id
#     label_str = gpt2_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

#     metrics_rouge = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1", "rouge2", "rouge3", "rouge4", "rougeL", "rougeLsum"])

#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [gpt2_tokenizer.convert_ids_to_tokens(seq, **kwargs) for seq in sequences]
    
#     pred_tokens = batch_convert_ids_to_tokens(pred_ids, skip_special_tokens=True)
    
#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [[gpt2_tokenizer.convert_ids_to_tokens(seq, **kwargs)] for seq in sequences]
    
#     label_tokens = batch_convert_ids_to_tokens(labels_ids, skip_special_tokens=True)
    
#     metrics_bleu = bleu.compute(predictions=pred_tokens, references=label_tokens)
    
#     return {
#         "rouge1_precision": round(metrics_rouge['rouge1'].mid.precision, 4),
#         "rouge1_recall": round(metrics_rouge['rouge1'].mid.recall, 4),
#         "rouge1_fmeasure": round(metrics_rouge['rouge1'].mid.fmeasure, 4),
#         "rouge2_precision": round(metrics_rouge['rouge2'].mid.precision, 4),
#         "rouge2_recall": round(metrics_rouge['rouge2'].mid.recall, 4),
#         "rouge2_fmeasure": round(metrics_rouge['rouge2'].mid.fmeasure, 4),
#         "rouge3_precision": round(metrics_rouge['rouge3'].mid.precision, 4),
#         "rouge3_recall": round(metrics_rouge['rouge3'].mid.recall, 4),
#         "rouge3_fmeasure": round(metrics_rouge['rouge3'].mid.fmeasure, 4),
#         "rouge4_precision": round(metrics_rouge['rouge4'].mid.precision, 4),
#         "rouge4_recall": round(metrics_rouge['rouge4'].mid.recall, 4),
#         "rouge4_fmeasure": round(metrics_rouge['rouge4'].mid.fmeasure, 4),
#         "rougeL_precision": round(metrics_rouge['rougeL'].mid.precision, 4),
#         "rougeL_recall": round(metrics_rouge['rougeL'].mid.recall, 4),
#         "rougeL_fmeasure": round(metrics_rouge['rougeL'].mid.fmeasure, 4),
#         "rougeLsum_precision": round(metrics_rouge['rougeLsum'].mid.precision, 4),
#         "rougeLsum_recall": round(metrics_rouge['rougeLsum'].mid.recall, 4),
#         "rougeLsum_fmeasure": round(metrics_rouge['rougeLsum'].mid.fmeasure, 4),
#         "bleu": round(metrics_bleu['bleu'], 4),
#     }



# # make train dataset ready
# train_dataset = train_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# train_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# # same for validation dataset
# val_dataset = val_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# val_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# # set training arguments - these params are not really tuned, feel free to change
# training_args = TrainingArguments(
#     output_dir="./models/bertweet2gpt2/",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     predict_from_generate=True,
#     evaluate_during_training=True,
#     do_train=True,
#     do_eval=True,
#     logging_steps=20,
#     save_steps=500,
#     eval_steps=300,
#     overwrite_output_dir=True,
#     warmup_steps=50,
#     save_total_limit=3,
#     fp16=True,
#     num_train_epochs=30,
#     run_name="bertweet2gpt2",
# )


# # instantiate trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# # start training
# trainer.train()


#--------------------------------------------------------------------------
#bertweet2bert

# import nlp
# from datasets import load_metric, Dataset
# import logging
# from transformers import BertTokenizer, AutoTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
# import pandas as pd

# logging.basicConfig(level=logging.INFO)

# model = EncoderDecoderModel.from_encoder_decoder_pretrained("vinai/bertweet-base", "bert-base-uncased")
# # cache is currently not supported by EncoderDecoder framework
# model.decoder.config.use_cache = False
# bertweet_tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

# bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# # CLS token will work as BOS token
# bert_tokenizer.bos_token = bert_tokenizer.cls_token

# # SEP token will work as EOS token
# bert_tokenizer.eos_token = bert_tokenizer.sep_token

# # set decoding params
# model.config.decoder_start_token_id = bert_tokenizer.bos_token_id
# model.config.eos_token_id = bert_tokenizer.eos_token_id
# model.config.max_length = 64
# model.config.min_length = 0
# model.config.no_repeat_ngram_size = 3
# model.early_stopping = True
# model.length_penalty = 2.0
# model.num_beams = 4

# # load train and validation data
# train_dataset = pd.read_json("WNUT2015_dataset/train_data.json", orient="records")
# val_dataset = pd.read_json("WNUT2015_dataset/test_truth.json", orient="records")

# make_sentence = lambda x : " ".join(x)

# train_dataset['input_sentence'] = train_dataset['input'].apply(make_sentence)
# train_dataset['output_sentence'] = train_dataset['output'].apply(make_sentence)
# val_dataset['input_sentence'] = val_dataset['input'].apply(make_sentence)
# val_dataset['output_sentence'] = val_dataset['output'].apply(make_sentence)

# train_dataset = Dataset.from_pandas(train_dataset)
# val_dataset = Dataset.from_pandas(val_dataset)

# # load rouge for validation
# rouge = load_metric('rouge', experiment_id=7)
# bleu = load_metric('bleu', experiment_id=7)

# encoder_length = 64
# decoder_length = 64
# batch_size = 16


# # map data correctly
# def map_to_encoder_decoder_inputs(batch):    # Tokenizer will automatically set [BOS] <text> [EOS] 
#     # use bert tokenizer here for encoder
#     inputs = bertweet_tokenizer(batch["input_sentence"], padding="max_length", truncation=True, max_length=encoder_length)
#     # force summarization <= 128
#     outputs = bert_tokenizer(batch["output_sentence"], padding="max_length", truncation=True, max_length=decoder_length)

#     batch["input_ids"] = inputs.input_ids
#     batch["attention_mask"] = inputs.attention_mask
#     batch["decoder_input_ids"] = outputs.input_ids
#     batch["labels"] = outputs.input_ids.copy()
#     batch["decoder_attention_mask"] = outputs.attention_mask

#     # complicated list comprehension here because pad_token_id alone is not good enough to know whether label should be excluded or not
#     batch["labels"] = [
#         [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(batch["decoder_attention_mask"], batch["labels"])]
#     ]

#     assert all([len(x) == encoder_length for x in inputs.input_ids])
#     assert all([len(x) == decoder_length for x in outputs.input_ids])

#     return batch


# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     # all unnecessary tokens are removed
#     pred_str = bert_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     labels_ids[labels_ids == -100] = bert_tokenizer.eos_token_id
#     label_str = bert_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

#     metrics_rouge = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1", "rouge2", "rouge3", "rouge4", "rougeL", "rougeLsum"])

#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [bert_tokenizer.convert_ids_to_tokens(seq, **kwargs) for seq in sequences]
    
#     pred_tokens = batch_convert_ids_to_tokens(pred_ids, skip_special_tokens=True)
    
#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [[bert_tokenizer.convert_ids_to_tokens(seq, **kwargs)] for seq in sequences]
    
#     label_tokens = batch_convert_ids_to_tokens(labels_ids, skip_special_tokens=True)
    
#     metrics_bleu = bleu.compute(predictions=pred_tokens, references=label_tokens)
    
#     return {
#         "rouge1_precision": round(metrics_rouge['rouge1'].mid.precision, 4),
#         "rouge1_recall": round(metrics_rouge['rouge1'].mid.recall, 4),
#         "rouge1_fmeasure": round(metrics_rouge['rouge1'].mid.fmeasure, 4),
#         "rouge2_precision": round(metrics_rouge['rouge2'].mid.precision, 4),
#         "rouge2_recall": round(metrics_rouge['rouge2'].mid.recall, 4),
#         "rouge2_fmeasure": round(metrics_rouge['rouge2'].mid.fmeasure, 4),
#         "rouge3_precision": round(metrics_rouge['rouge3'].mid.precision, 4),
#         "rouge3_recall": round(metrics_rouge['rouge3'].mid.recall, 4),
#         "rouge3_fmeasure": round(metrics_rouge['rouge3'].mid.fmeasure, 4),
#         "rouge4_precision": round(metrics_rouge['rouge4'].mid.precision, 4),
#         "rouge4_recall": round(metrics_rouge['rouge4'].mid.recall, 4),
#         "rouge4_fmeasure": round(metrics_rouge['rouge4'].mid.fmeasure, 4),
#         "rougeL_precision": round(metrics_rouge['rougeL'].mid.precision, 4),
#         "rougeL_recall": round(metrics_rouge['rougeL'].mid.recall, 4),
#         "rougeL_fmeasure": round(metrics_rouge['rougeL'].mid.fmeasure, 4),
#         "rougeLsum_precision": round(metrics_rouge['rougeLsum'].mid.precision, 4),
#         "rougeLsum_recall": round(metrics_rouge['rougeLsum'].mid.recall, 4),
#         "rougeLsum_fmeasure": round(metrics_rouge['rougeLsum'].mid.fmeasure, 4),
#         "bleu": round(metrics_bleu['bleu'], 4),
#     }



# # make train dataset ready
# train_dataset = train_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# train_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# # same for validation dataset
# val_dataset = val_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# val_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# # set training arguments - these params are not really tuned, feel free to change
# training_args = TrainingArguments(
#     output_dir="./models/bertweet2bert/"+str(model_name)+"/",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     predict_from_generate=True,
#     evaluate_during_training=True,
#     do_train=True,
#     do_eval=True,
#     logging_steps=20,
#     save_steps=500,
#     eval_steps=300,
#     overwrite_output_dir=True,
#     warmup_steps=50,
#     save_total_limit=3,
#     fp16=True,
#     num_train_epochs=30,
#     run_name="bertweet2bert",
# )

# # instantiate trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# # start training
# trainer.train()




#--------------------------------------------------------------------------
#bertweet2bertweet

# import nlp
# import logging
# from datasets import load_metric, Dataset
# from transformers import AutoTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
# import pandas as pd

# logging.basicConfig(level=logging.INFO)

# model = EncoderDecoderModel.from_encoder_decoder_pretrained("vinai/bertweet-base", "vinai/bertweet-base")
# tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

# # load train and validation data
# train_dataset = pd.read_json("WNUT2015_dataset/train_data.json", orient="records")
# val_dataset = pd.read_json("WNUT2015_dataset/test_truth.json", orient="records")

# make_sentence = lambda x : " ".join(x)

# train_dataset['input_sentence'] = train_dataset['input'].apply(make_sentence)
# train_dataset['output_sentence'] = train_dataset['output'].apply(make_sentence)
# val_dataset['input_sentence'] = val_dataset['input'].apply(make_sentence)
# val_dataset['output_sentence'] = val_dataset['output'].apply(make_sentence)

# train_dataset = Dataset.from_pandas(train_dataset)
# val_dataset = Dataset.from_pandas(val_dataset)

# # load rouge for validation
# #rouge = nlp.load_metric("rouge")
# rouge = load_metric('rouge', experiment_id=8)
# bleu = load_metric('bleu', experiment_id=8)

# # set decoding params
# model.config.decoder_start_token_id = tokenizer.bos_token_id
# model.config.eos_token_id = tokenizer.eos_token_id
# model.config.max_length = 64
# model.config.min_length = 0
# model.config.no_repeat_ngram_size = 3
# model.early_stopping = True
# model.length_penalty = 2.0
# model.num_beams = 4


# # map data correctly
# def map_to_encoder_decoder_inputs(batch):
#     # Tokenizer will automatically set [BOS] <text> [EOS]
#     # cut off at BERT max length 512
#     inputs = tokenizer(batch["input_sentence"], padding="max_length", truncation=True, max_length=64)
#     # inputs = tokenizer(batch["input_sentence"], padding="max_length")
#     # force summarization <= 128
#     outputs = tokenizer(batch["output_sentence"], padding="max_length", truncation=True, max_length=64)
#     # outputs = tokenizer(batch["output_sentence"], padding="max_length")

#     batch["input_ids"] = inputs.input_ids
#     batch["attention_mask"] = inputs.attention_mask

#     batch["decoder_input_ids"] = outputs.input_ids
#     batch["labels"] = outputs.input_ids.copy()
#     # mask loss for padding
#     batch["labels"] = [
#         [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
#     ]
#     batch["decoder_attention_mask"] = outputs.attention_mask

#     assert all([len(x) == 64 for x in inputs.input_ids])
#     assert all([len(x) == 64 for x in outputs.input_ids])
    
#     return batch


# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     # all unnecessary tokens are removed
#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
#     metrics_rouge = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1", "rouge2", "rouge3", "rouge4", "rougeL", "rougeLsum"])

#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [tokenizer.convert_ids_to_tokens(seq, **kwargs) for seq in sequences]
    
#     pred_tokens = batch_convert_ids_to_tokens(pred_ids, skip_special_tokens=True)
    
#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [[tokenizer.convert_ids_to_tokens(seq, **kwargs)] for seq in sequences]
    
#     label_tokens = batch_convert_ids_to_tokens(labels_ids, skip_special_tokens=True)
    
#     metrics_bleu = bleu.compute(predictions=pred_tokens, references=label_tokens)
    
#     return {
#         "rouge1_precision": round(metrics_rouge['rouge1'].mid.precision, 4),
#         "rouge1_recall": round(metrics_rouge['rouge1'].mid.recall, 4),
#         "rouge1_fmeasure": round(metrics_rouge['rouge1'].mid.fmeasure, 4),
#         "rouge2_precision": round(metrics_rouge['rouge2'].mid.precision, 4),
#         "rouge2_recall": round(metrics_rouge['rouge2'].mid.recall, 4),
#         "rouge2_fmeasure": round(metrics_rouge['rouge2'].mid.fmeasure, 4),
#         "rouge3_precision": round(metrics_rouge['rouge3'].mid.precision, 4),
#         "rouge3_recall": round(metrics_rouge['rouge3'].mid.recall, 4),
#         "rouge3_fmeasure": round(metrics_rouge['rouge3'].mid.fmeasure, 4),
#         "rouge4_precision": round(metrics_rouge['rouge4'].mid.precision, 4),
#         "rouge4_recall": round(metrics_rouge['rouge4'].mid.recall, 4),
#         "rouge4_fmeasure": round(metrics_rouge['rouge4'].mid.fmeasure, 4),
#         "rougeL_precision": round(metrics_rouge['rougeL'].mid.precision, 4),
#         "rougeL_recall": round(metrics_rouge['rougeL'].mid.recall, 4),
#         "rougeL_fmeasure": round(metrics_rouge['rougeL'].mid.fmeasure, 4),
#         "rougeLsum_precision": round(metrics_rouge['rougeLsum'].mid.precision, 4),
#         "rougeLsum_recall": round(metrics_rouge['rougeLsum'].mid.recall, 4),
#         "rougeLsum_fmeasure": round(metrics_rouge['rougeLsum'].mid.fmeasure, 4),
#         "bleu": round(metrics_bleu['bleu'], 4),
#     }



# # set batch size here
# batch_size = 16

# # make train dataset ready
# train_dataset = train_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# train_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# # same for validation dataset
# val_dataset = val_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# val_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# # set training arguments - these params are not really tuned, feel free to change
# training_args = TrainingArguments(
#     output_dir="./models/bertweet2bertweet/"+str(model_name)+"/",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     predict_from_generate=True,
#     evaluate_during_training=True,
#     do_train=True,
#     do_eval=True,
#     logging_steps=20,
#     save_steps=500,
#     eval_steps=300,
#     overwrite_output_dir=True,
#     warmup_steps=50,
#     save_total_limit=3,
#     num_train_epochs=30,
#     run_name="bertweet2bertweet",
# )


# # instantiate trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )


# # start training
# trainer.train()


#----------------------------------------------------------------------------
#bertweet2bertweet_share


# import nlp
# from datasets import load_metric, Dataset
# import logging
# from transformers import AutoTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
# import pandas as pd

# logging.basicConfig(level=logging.INFO)

# model = EncoderDecoderModel.from_encoder_decoder_pretrained("vinai/bertweet-base", "vinai/bertweet-base", tie_encoder_decoder=True)
# tokenizer =  AutoTokenizer.from_pretrained("vinai/bertweet-base")

# # load train and validation data
# train_dataset = pd.read_json("WNUT2015_dataset/train_data.json", orient="records")
# val_dataset = pd.read_json("WNUT2015_dataset/test_truth.json", orient="records")

# make_sentence = lambda x : " ".join(x)

# train_dataset['input_sentence'] = train_dataset['input'].apply(make_sentence)
# train_dataset['output_sentence'] = train_dataset['output'].apply(make_sentence)
# val_dataset['input_sentence'] = val_dataset['input'].apply(make_sentence)
# val_dataset['output_sentence'] = val_dataset['output'].apply(make_sentence)

# train_dataset = Dataset.from_pandas(train_dataset)
# val_dataset = Dataset.from_pandas(val_dataset)

# # load rouge for validation
# rouge = load_metric("rouge", experiment_id=9)
# bleu = load_metric('bleu', experiment_id=9)

# # set decoding params
# model.config.decoder_start_token_id = tokenizer.bos_token_id
# model.config.eos_token_id = tokenizer.eos_token_id
# model.config.max_length = 64
# model.config.min_length = 0
# model.config.no_repeat_ngram_size = 3
# model.early_stopping = True
# model.length_penalty = 2.0
# model.num_beams = 4

# encoder_length = 64
# decoder_length = 64
# batch_size = 16


# # map data correctly
# def map_to_encoder_decoder_inputs(batch):
#     # Tokenizer will automatically set [BOS] <text> [EOS]
#     # cut off at Longformer at 2048
#     inputs = tokenizer(batch["input_sentence"], padding="max_length", truncation=True, max_length=encoder_length)
#     # force summarization <= 256
#     outputs = tokenizer(batch["output_sentence"], padding="max_length", truncation=True, max_length=decoder_length)

#     batch["input_ids"] = inputs.input_ids
#     batch["attention_mask"] = inputs.attention_mask
#     batch["decoder_input_ids"] = outputs.input_ids
#     batch["labels"] = outputs.input_ids.copy()
#     # mask loss for padding
#     batch["labels"] = [
#         [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
#     ]
#     batch["decoder_attention_mask"] = outputs.attention_mask

#     assert all([len(x) == encoder_length for x in inputs.input_ids])
#     assert all([len(x) == decoder_length for x in outputs.input_ids])

#     return batch


# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     # all unnecessary tokens are removed
#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     labels_ids[labels_ids == -100] = tokenizer.eos_token_id
#     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

#     metrics_rouge = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1", "rouge2", "rouge3", "rouge4", "rougeL", "rougeLsum"])

#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [tokenizer.convert_ids_to_tokens(seq, **kwargs) for seq in sequences]
    
#     pred_tokens = batch_convert_ids_to_tokens(pred_ids, skip_special_tokens=True)
    
#     def batch_convert_ids_to_tokens(sequences, **kwargs):
#         return [[tokenizer.convert_ids_to_tokens(seq, **kwargs)] for seq in sequences]
    
#     label_tokens = batch_convert_ids_to_tokens(labels_ids, skip_special_tokens=True)
    
#     metrics_bleu = bleu.compute(predictions=pred_tokens, references=label_tokens)
    
#     return {
#         "rouge1_precision": round(metrics_rouge['rouge1'].mid.precision, 4),
#         "rouge1_recall": round(metrics_rouge['rouge1'].mid.recall, 4),
#         "rouge1_fmeasure": round(metrics_rouge['rouge1'].mid.fmeasure, 4),
#         "rouge2_precision": round(metrics_rouge['rouge2'].mid.precision, 4),
#         "rouge2_recall": round(metrics_rouge['rouge2'].mid.recall, 4),
#         "rouge2_fmeasure": round(metrics_rouge['rouge2'].mid.fmeasure, 4),
#         "rouge3_precision": round(metrics_rouge['rouge3'].mid.precision, 4),
#         "rouge3_recall": round(metrics_rouge['rouge3'].mid.recall, 4),
#         "rouge3_fmeasure": round(metrics_rouge['rouge3'].mid.fmeasure, 4),
#         "rouge4_precision": round(metrics_rouge['rouge4'].mid.precision, 4),
#         "rouge4_recall": round(metrics_rouge['rouge4'].mid.recall, 4),
#         "rouge4_fmeasure": round(metrics_rouge['rouge4'].mid.fmeasure, 4),
#         "rougeL_precision": round(metrics_rouge['rougeL'].mid.precision, 4),
#         "rougeL_recall": round(metrics_rouge['rougeL'].mid.recall, 4),
#         "rougeL_fmeasure": round(metrics_rouge['rougeL'].mid.fmeasure, 4),
#         "rougeLsum_precision": round(metrics_rouge['rougeLsum'].mid.precision, 4),
#         "rougeLsum_recall": round(metrics_rouge['rougeLsum'].mid.recall, 4),
#         "rougeLsum_fmeasure": round(metrics_rouge['rougeLsum'].mid.fmeasure, 4),
#         "bleu": round(metrics_bleu['bleu'], 4),
#     }



# # make train dataset ready
# train_dataset = train_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# train_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_attention_mask", "decoder_input_ids", "labels"],
# )

# # same for validation dataset
# val_dataset = val_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
# )
# val_dataset.set_format(
#     type="torch", columns=["input_ids", "decoder_attention_mask", "attention_mask", "decoder_input_ids", "labels"],
# )

# # set training arguments - these params are not really tuned, feel free to change
# training_args = TrainingArguments(
#     output_dir="./models/bertweet2bertweet_share/"+str(model_name)+"/",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     predict_from_generate=True,
#     evaluate_during_training=True,
#     do_train=True,
#     do_eval=True,
#     logging_steps=20,
#     save_steps=500,
#     eval_steps=300,
#     overwrite_output_dir=True,
#     warmup_steps=50,
#     save_total_limit=3,
#     fp16=True,
#     num_train_epochs=30,
#     run_name="bertweet2bertweet_share",
# )

# # instantiate trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# # start training
# trainer.train()