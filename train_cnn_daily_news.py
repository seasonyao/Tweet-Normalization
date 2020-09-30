#!/usr/bin/env python3
import wandb
wandb.init(project="cnn_daily_news", entity="zonghaiyao")

# import nlp
# from datasets import load_metric
# import logging
# from transformers import BertTokenizer, GPT2Tokenizer, EncoderDecoderModel, Trainer, TrainingArguments

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
# model.config.max_length = 142
# model.config.min_length = 56
# model.config.no_repeat_ngram_size = 3
# model.early_stopping = True
# model.length_penalty = 2.0
# model.num_beams = 4

# # load train and validation data
# train_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="train")
# val_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="validation[:5%]")

# # load rouge for validation
# #rouge = nlp.load_metric("rouge", experiment_id=0, config_name="bert2gpt2")
# rouge = load_metric('rouge', experiment_id=0)

# encoder_length = 512
# decoder_length = 128
# batch_size = 16


# # map data correctly
# def map_to_encoder_decoder_inputs(batch):    # Tokenizer will automatically set [BOS] <text> [EOS] 
#     # use bert tokenizer here for encoder
#     inputs = bert_tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_length)
#     # force summarization <= 128
#     outputs = gpt2_tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=decoder_length)

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

#     rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

#     return {
#         "rouge2_precision": round(rouge_output.precision, 4),
#         "rouge2_recall": round(rouge_output.recall, 4),
#         "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
#     }


# # make train dataset ready
# train_dataset = train_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
# )
# train_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# # same for validation dataset
# val_dataset = val_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
# )
# val_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# # set training arguments - these params are not really tuned, feel free to change
# training_args = TrainingArguments(
#     output_dir="./models/bert2gpt2/",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     predict_from_generate=True,
#     evaluate_during_training=True,
#     do_train=True,
#     do_eval=True,
#     logging_steps=1000,
#     save_steps=5000,
#     eval_steps=5000,
#     overwrite_output_dir=True,
#     warmup_steps=2000,
#     save_total_limit=10,
#     fp16=True,
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


#-------------------------------------------------------------------------------
# import nlp
# import logging
# from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments

# logging.basicConfig(level=logging.INFO)

# model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# # CLS token will work as BOS token
# tokenizer.bos_token = tokenizer.cls_token

# # SEP token will work as EOS token
# tokenizer.eos_token = tokenizer.sep_token

# # load train and validation data
# train_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="train")
# val_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="validation[:10%]")

# # load rouge for validation
# rouge = nlp.load_metric("rouge")


# # set decoding params
# model.config.decoder_start_token_id = tokenizer.bos_token_id
# model.config.eos_token_id = tokenizer.eos_token_id
# model.config.max_length = 142
# model.config.min_length = 56
# model.config.no_repeat_ngram_size = 3
# model.early_stopping = True
# model.length_penalty = 2.0
# model.num_beams = 4


# # map data correctly
# def map_to_encoder_decoder_inputs(batch):
#     # Tokenizer will automatically set [BOS] <text> [EOS]
#     # cut off at BERT max length 512
#     inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512)
#     # force summarization <= 128
#     outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=128)

#     batch["input_ids"] = inputs.input_ids
#     batch["attention_mask"] = inputs.attention_mask

#     batch["decoder_input_ids"] = outputs.input_ids
#     batch["labels"] = outputs.input_ids.copy()
#     # mask loss for padding
#     batch["labels"] = [
#         [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
#     ]
#     batch["decoder_attention_mask"] = outputs.attention_mask

#     assert all([len(x) == 512 for x in inputs.input_ids])
#     assert all([len(x) == 128 for x in outputs.input_ids])
    
#     return batch


# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     # all unnecessary tokens are removed
#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
#     rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
    
#     return {
#         "rouge2_precision": round(rouge_output.precision, 4),
#         "rouge2_recall": round(rouge_output.recall, 4),
#         "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
#     }


# # set batch size here
# batch_size = 16

# # make train dataset ready
# train_dataset = train_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
# )
# train_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# # same for validation dataset
# val_dataset = val_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
# )
# val_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )

# # set training arguments - these params are not really tuned, feel free to change
# training_args = TrainingArguments(
#     output_dir="./models/bert2bert/",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     predict_from_generate=True,
#     evaluate_during_training=True,
#     do_train=True,
#     do_eval=True,
#     logging_steps=1000,
#     save_steps=1000,
#     eval_steps=1000,
#     overwrite_output_dir=True,
#     warmup_steps=2000,
#     save_total_limit=10,
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

#-----------------------------------------------------------------------------
# import nlp
# from datasets import load_metric
# import logging
# from transformers import RobertaTokenizer, EncoderDecoderModel, Trainer, TrainingArguments

# logging.basicConfig(level=logging.INFO)

# model = EncoderDecoderModel.from_encoder_decoder_pretrained("roberta-base", "roberta-base")
# tokenizer =  RobertaTokenizer.from_pretrained("roberta-base")

# # load train and validation data
# train_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="train")
# val_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="validation[:5%]")

# # load rouge for validation
# rouge = load_metric("rouge", experiment_id=3)

# # set decoding params
# model.config.decoder_start_token_id = tokenizer.bos_token_id
# model.config.eos_token_id = tokenizer.eos_token_id
# model.config.max_length = 142
# model.config.min_length = 56
# model.config.no_repeat_ngram_size = 3
# model.early_stopping = True
# model.length_penalty = 2.0
# model.num_beams = 4

# encoder_length = 512
# decoder_length = 128
# batch_size = 16


# # map data correctly
# def map_to_encoder_decoder_inputs(batch):
#     # Tokenizer will automatically set [BOS] <text> [EOS]
#     # cut off at Longformer at 2048
#     inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_length)
#     # force summarization <= 256
#     outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=decoder_length)

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

#     rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

#     return {
#         "rouge2_precision": round(rouge_output.precision, 4),
#         "rouge2_recall": round(rouge_output.recall, 4),
#         "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
#     }


# # make train dataset ready
# train_dataset = train_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
# )
# train_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_attention_mask", "decoder_input_ids", "labels"],
# )

# # same for validation dataset
# val_dataset = val_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
# )
# val_dataset.set_format(
#     type="torch", columns=["input_ids", "decoder_attention_mask", "attention_mask", "decoder_input_ids", "labels"],
# )

# # set training arguments - these params are not really tuned, feel free to change
# training_args = TrainingArguments(
#     output_dir="./models/roberta2roberta/",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     predict_from_generate=True,
#     evaluate_during_training=True,
#     do_train=True,
#     do_eval=True,
#     logging_steps=1000,
#     save_steps=1000,
#     eval_steps=1000,
#     overwrite_output_dir=True,
#     warmup_steps=2000,
#     save_total_limit=3,
#     fp16=True,
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

# #---------------------------------------------------------------------
# import nlp
# from datasets import load_metric
# import logging
# from transformers import RobertaTokenizer, EncoderDecoderModel, Trainer, TrainingArguments

# logging.basicConfig(level=logging.INFO)

# model = EncoderDecoderModel.from_encoder_decoder_pretrained("roberta-base", "roberta-base", tie_encoder_decoder=True)
# tokenizer =  RobertaTokenizer.from_pretrained("roberta-base")

# # load train and validation data
# train_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="train")
# val_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="validation[:5%]")

# # load rouge for validation
# rouge = load_metric("rouge", experiment_id=2)

# # set decoding params
# model.config.decoder_start_token_id = tokenizer.bos_token_id
# model.config.eos_token_id = tokenizer.eos_token_id
# model.config.max_length = 142
# model.config.min_length = 56
# model.config.no_repeat_ngram_size = 3
# model.early_stopping = True
# model.length_penalty = 2.0
# model.num_beams = 4

# encoder_length = 512
# decoder_length = 128
# batch_size = 16


# # map data correctly
# def map_to_encoder_decoder_inputs(batch):
#     # Tokenizer will automatically set [BOS] <text> [EOS]
#     # cut off at Longformer at 2048
#     inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_length)
#     # force summarization <= 256
#     outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=decoder_length)

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

#     rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

#     return {
#         "rouge2_precision": round(rouge_output.precision, 4),
#         "rouge2_recall": round(rouge_output.recall, 4),
#         "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
#     }


# # make train dataset ready
# train_dataset = train_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
# )
# train_dataset.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_attention_mask", "decoder_input_ids", "labels"],
# )

# # same for validation dataset
# val_dataset = val_dataset.map(
#     map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
# )
# val_dataset.set_format(
#     type="torch", columns=["input_ids", "decoder_attention_mask", "attention_mask", "decoder_input_ids", "labels"],
# )

# # set training arguments - these params are not really tuned, feel free to change
# training_args = TrainingArguments(
#     output_dir="./models/roberta2roberta_share/",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     predict_from_generate=True,
#     evaluate_during_training=True,
#     do_train=True,
#     do_eval=True,
#     logging_steps=1000,
#     save_steps=1000,
#     eval_steps=1000,
#     overwrite_output_dir=True,
#     warmup_steps=2000,
#     save_total_limit=3,
#     fp16=True,
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
from datasets import load_metric
import nlp
import logging
from transformers import AutoTokenizer, EncoderDecoderModel, Trainer, TrainingArguments

logging.basicConfig(level=logging.INFO)

model = EncoderDecoderModel.from_encoder_decoder_pretrained("vinai/bertweet-base", "bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

# CLS token will work as BOS token
tokenizer.bos_token = tokenizer.cls_token

# SEP token will work as EOS token
tokenizer.eos_token = tokenizer.sep_token

# load train and validation data
train_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="train")
val_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="validation[:10%]")

# load rouge for validation
rouge = load_metric('rouge', experiment_id=1)


# set decoding params
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 142
model.config.min_length = 56
model.config.no_repeat_ngram_size = 3
model.early_stopping = True
model.length_penalty = 2.0
model.num_beams = 4


# map data correctly
def map_to_encoder_decoder_inputs(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512)
    # force summarization <= 128
    outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=128)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    batch["decoder_input_ids"] = outputs.input_ids
    batch["labels"] = outputs.input_ids.copy()
    # mask loss for padding
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
    ]
    batch["decoder_attention_mask"] = outputs.attention_mask

    assert all([len(x) == 512 for x in inputs.input_ids])
    assert all([len(x) == 128 for x in outputs.input_ids])
    
    return batch


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
    
    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


# set batch size here
batch_size = 16

# make train dataset ready
train_dataset = train_dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
)
train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# same for validation dataset
val_dataset = val_dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["article", "highlights"],
)
val_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# set training arguments - these params are not really tuned, feel free to change
training_args = TrainingArguments(
    output_dir="./models/bertweet2bert/",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_from_generate=True,
    evaluate_during_training=True,
    do_train=True,
    do_eval=True,
    logging_steps=1000,
    save_steps=5000,
    eval_steps=5000,
    overwrite_output_dir=True,
    warmup_steps=2000,
    save_total_limit=10,
)

# instantiate trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# start training
trainer.train()