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
import nlp
import logging
from datasets import load_metric
from transformers import EncoderDecoderModel, Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset
import pandas as pd
from tweetNormalizer import normalizeTweet

logging.basicConfig(level=logging.INFO)

#change 6 places

#bert-base-uncased, gpt2, roberta-base, vinai/bertweet-base
RUN_NAME="bertweet2bertweet_share"
ENCODER = "vinai/bertweet-base"
DECODER = "vinai/bertweet-base"
tie_ENCODER_DECODER=True
OUTPUT_DIR="./models/"+RUN_NAME+"/"+str(model_name)+"/"

batch_size = 16   # set batch size here
encoder_length = 128
decoder_length = 128

PATH_TO_TRAIN_DATA = "WNUT2015_dataset/train_data.json"
PATH_TO_VAL_DATA = "WNUT2015_dataset/test_truth.json"
is_normalizeTweet = True

rouge = load_metric('rouge', experiment_id=7)
bleu = load_metric('bleu', experiment_id=7)

#---------------------------------------------------------------------------------------------
# set EncoderDecoderModel
model = EncoderDecoderModel.from_encoder_decoder_pretrained(ENCODER, DECODER, tie_encoder_decoder = tie_ENCODER_DECODER)

# encoder tokenizer
encoder_tokenizer = AutoTokenizer.from_pretrained(ENCODER)

if ENCODER=="bert-base-uncased":
    # CLS token will work as BOS token, SEP token will work as EOS token
    encoder_tokenizer.bos_token = encoder_tokenizer.cls_token
    encoder_tokenizer.eos_token = encoder_tokenizer.sep_token
    
# decoder tokenizer
if DECODER=="gpt2":
    # make sure GPT2 appends EOS in begin and end
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return outputs
    
    AutoTokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
    decoder_tokenizer = AutoTokenizer.from_pretrained(DECODER)
    # set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
    decoder_tokenizer.pad_token = decoder_tokenizer.unk_token
else:   
    decoder_tokenizer = AutoTokenizer.from_pretrained(DECODER)

if DECODER=="bert-base-uncased":
    # CLS token will work as BOS token, SEP token will work as EOS token
    decoder_tokenizer.bos_token = decoder_tokenizer.cls_token
    decoder_tokenizer.eos_token = decoder_tokenizer.sep_token
    

# set decoding params
model.config.decoder_start_token_id = decoder_tokenizer.bos_token_id
model.config.eos_token_id = decoder_tokenizer.eos_token_id
model.config.max_length = decoder_length
model.config.min_length = 0
model.config.no_repeat_ngram_size = 3
model.early_stopping = True
model.length_penalty = 2.0
model.num_beams = 4

#-------------------------------------------------------------------------------------------
# load train and validation data
train_dataset = pd.read_json(PATH_TO_TRAIN_DATA, orient="records")
val_dataset = pd.read_json(PATH_TO_VAL_DATA, orient="records")

if is_normalizeTweet:
    make_sentence = lambda x : normalizeTweet(" ".join(x)).lower()
else:
    make_sentence = lambda x : " ".join(x).lower()

train_dataset['input_sentence'] = train_dataset['input'].apply(make_sentence)
train_dataset['output_sentence'] = train_dataset['output'].apply(make_sentence)
val_dataset['input_sentence'] = val_dataset['input'].apply(make_sentence)
val_dataset['output_sentence'] = val_dataset['output'].apply(make_sentence)

train_dataset = Dataset.from_pandas(train_dataset)
val_dataset = Dataset.from_pandas(val_dataset)

# map data correctly
def map_to_encoder_decoder_inputs(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    inputs = encoder_tokenizer(batch["input_sentence"], padding="max_length", truncation=True, max_length=encoder_length)
    outputs = decoder_tokenizer(batch["output_sentence"], padding="max_length", truncation=True, max_length=decoder_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    batch["decoder_input_ids"] = outputs.input_ids
    batch["labels"] = outputs.input_ids.copy()
    batch["decoder_attention_mask"] = outputs.attention_mask
    
    if DECODER=="gpt2":
        # complicated list comprehension here because pad_token_id alone is not good enough to know whether label should be excluded or not
        batch["labels"] = [
            [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(batch["decoder_attention_mask"], batch["labels"])]
        ]
    else:
        # mask loss for padding
        batch["labels"] = [
            [-100 if token == decoder_tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
        ]
    

    assert all([len(x) == encoder_length for x in inputs.input_ids])
    assert all([len(x) == decoder_length for x in outputs.input_ids])
    
    return batch

# make train dataset ready
train_dataset = train_dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
)
train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# same for validation dataset
val_dataset = val_dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
)
val_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)
        
#-----------------------------------------------------------------------------------
# load metrics for validation

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = decoder_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = decoder_tokenizer.eos_token_id
    label_str = decoder_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    metrics_rouge = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1", "rouge2", "rouge3", "rouge4", "rougeL", "rougeLsum"])

    def batch_convert_ids_to_tokens(sequences, **kwargs):
        return [decoder_tokenizer.convert_ids_to_tokens(seq, **kwargs) for seq in sequences]
    
    pred_tokens = batch_convert_ids_to_tokens(pred_ids, skip_special_tokens=True)
    
    def batch_convert_ids_to_tokens(sequences, **kwargs):
        return [[decoder_tokenizer.convert_ids_to_tokens(seq, **kwargs)] for seq in sequences]
    
    label_tokens = batch_convert_ids_to_tokens(labels_ids, skip_special_tokens=True)
    
    metrics_bleu = bleu.compute(predictions=pred_tokens, references=label_tokens)
    
    return {
        "rouge1_precision": round(metrics_rouge['rouge1'].mid.precision, 4),
        "rouge1_recall": round(metrics_rouge['rouge1'].mid.recall, 4),
        "rouge1_fmeasure": round(metrics_rouge['rouge1'].mid.fmeasure, 4),
        "rouge2_precision": round(metrics_rouge['rouge2'].mid.precision, 4),
        "rouge2_recall": round(metrics_rouge['rouge2'].mid.recall, 4),
        "rouge2_fmeasure": round(metrics_rouge['rouge2'].mid.fmeasure, 4),
        "rouge3_precision": round(metrics_rouge['rouge3'].mid.precision, 4),
        "rouge3_recall": round(metrics_rouge['rouge3'].mid.recall, 4),
        "rouge3_fmeasure": round(metrics_rouge['rouge3'].mid.fmeasure, 4),
        "rouge4_precision": round(metrics_rouge['rouge4'].mid.precision, 4),
        "rouge4_recall": round(metrics_rouge['rouge4'].mid.recall, 4),
        "rouge4_fmeasure": round(metrics_rouge['rouge4'].mid.fmeasure, 4),
        "rougeL_precision": round(metrics_rouge['rougeL'].mid.precision, 4),
        "rougeL_recall": round(metrics_rouge['rougeL'].mid.recall, 4),
        "rougeL_fmeasure": round(metrics_rouge['rougeL'].mid.fmeasure, 4),
        "rougeLsum_precision": round(metrics_rouge['rougeLsum'].mid.precision, 4),
        "rougeLsum_recall": round(metrics_rouge['rougeLsum'].mid.recall, 4),
        "rougeLsum_fmeasure": round(metrics_rouge['rougeLsum'].mid.fmeasure, 4),
        "bleu": round(metrics_bleu['bleu'], 4),
    }
        
#----------------------------------------------------------------------------------------
# begin train

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_from_generate=True,
    evaluate_during_training=True,
    do_train=True,
    do_eval=True,
    logging_steps=20,
    save_steps=500,
    eval_steps=300,
    overwrite_output_dir=True,
    warmup_steps=50,
    save_total_limit=3,
    num_train_epochs=30,
    fp16=True,
    run_name=RUN_NAME,
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
