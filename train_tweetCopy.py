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
from transformers import EncoderDecoderModel, Trainer, TrainingArguments, AutoTokenizer, set_seed, EncoderDecoderConfig
from datasets import Dataset
import pandas as pd
from tweetNormalizer import normalizeTweet
from nltk.tokenize import TweetTokenizer
from emoji import demojize
from collections import defaultdict
import numpy as np

nltk_tokenizer = TweetTokenizer()

logging.basicConfig(level=logging.INFO)
#----------------------------------------------------------------------------------------------
SEED = 42
SAME_INOUTPUT_RATE = 0
TWEET_COPY_TASK = 1
COPY_TASK_MAX_LEN = 128

NORM_TASK_MAX_LEN = 128 #128 means all here, since max token num is 128

FINE_TUNING_ON_TWEET_COPY = 0
TWEET_COPY_MODEL_PATH = './models/bert2bert_share/tweetcopy_len12/3/checkpoint-5000'

LR = 1e-4 #3e-4, 1e-4, 5e-5, 3e-5

#bert-base-uncased, gpt2, roberta-base, vinai/bertweet-base, google/electra-base-discriminator
RUN_NAME="bert2bert_share"
ENCODER = "bert-base-uncased"
DECODER = "bert-base-uncased"
tie_ENCODER_DECODER=True
OUTPUT_DIR="./models/"+RUN_NAME+"/"+str(model_name)+"/14"

# RUN_NAME="bert2bert_share_ios0_lr1e4_alignEmbed"
# RUN_NAME="tweetcopy_max_all_1e4_batch16_preCopy12"
RUN_NAME="tweetcopy_max_all_1e4_batch16"
RUN_NAME = "zonghaiyao tweetcopy " + RUN_NAME

batch_size = 16   # set batch size here
encoder_length = 128
decoder_length = 128

PATH_TO_TRAIN_DATA = "WNUT2015_dataset/train_data.json"
PATH_TO_VAL_DATA = "WNUT2015_dataset/test_truth.json"
is_alignEmbed = False

rouge = load_metric('rouge', experiment_id=14)
bleu = load_metric('bleu', experiment_id=14)

#---------------------------------------------------------------------------------------------
# set EncoderDecoderModel
if FINE_TUNING_ON_TWEET_COPY == 1:
    encoder_decoder_config = EncoderDecoderConfig.from_pretrained(TWEET_COPY_MODEL_PATH)
    model = EncoderDecoderModel.from_pretrained(TWEET_COPY_MODEL_PATH, config=encoder_decoder_config)
else:
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(ENCODER, DECODER, tie_encoder_decoder = tie_ENCODER_DECODER)

# encoder tokenizer
encoder_tokenizer = AutoTokenizer.from_pretrained(ENCODER)

if ENCODER=="bert-base-uncased" or "google/electra-base-discriminator":
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

#---------------------------------
#do some normalization ourselves
def norm(token):
    if token.lower().startswith("@"):
        return "username"
    elif token.lower().startswith('#'):
        return "hashtag"
    elif token.lower().startswith("http") or token.lower().startswith("www"):
        return "httpurl"
    else:
        return token.replace("’", "'").replace("…", "...")

def pre_pocessing_input(x):
    result = []
    token_count = 0
    for item in x:
        item = norm(item)
        #the reason for encode+decode is "?!!" need to be "? ! !"
        input_ids = encoder_tokenizer(item).input_ids
        item = encoder_tokenizer.decode(input_ids, skip_special_tokens=True)
        result.append(item)
        token_count += 1
        if TWEET_COPY_TASK == 1 and token_count == COPY_TASK_MAX_LEN:
            break
        if TWEET_COPY_TASK == 0 and token_count == NORM_TASK_MAX_LEN:
            break
    return result

def pre_pocessing_output(x):
    result = []
    token_count = 0
    for item in x:
        item = norm(item)
        #the reason for encode+decode is "?!!" need to be "? ! !"
        input_ids = decoder_tokenizer(item).input_ids
        item = decoder_tokenizer.decode(input_ids, skip_special_tokens=True)
        result.append(item)
        token_count += 1
        if TWEET_COPY_TASK == 1 and token_count == COPY_TASK_MAX_LEN:
            break
        if TWEET_COPY_TASK == 0 and token_count == NORM_TASK_MAX_LEN:
            break
    return result

train_dataset['input'] = train_dataset['input'].apply(pre_pocessing_input)
train_dataset['output'] = train_dataset['output'].apply(pre_pocessing_output)
val_dataset['input'] = val_dataset['input'].apply(pre_pocessing_input)
val_dataset['output'] = val_dataset['output'].apply(pre_pocessing_output)
#-----------------------------------------
#check if it is tweet-copy task
if TWEET_COPY_TASK == 1:
    train_dataset_copy = train_dataset.copy()
    
    make_input_output_same = lambda x: x['output'].copy()
    train_dataset['input'] = train_dataset.apply(make_input_output_same, axis=1)
    val_dataset['input'] = val_dataset.apply(make_input_output_same, axis=1)
    
    make_input_output_same = lambda x: x['input'].copy()
    train_dataset_copy['output'] = train_dataset_copy.apply(make_input_output_same, axis=1)
    
    train_dataset = train_dataset.append(train_dataset_copy, ignore_index=True)
    train_dataset = train_dataset.sample(frac=1, random_state=SEED).reset_index(drop=True)
#do some data augumentation
elif SAME_INOUTPUT_RATE > 0: 
    train_dataset_no_chage_data = train_dataset.copy()
    train_dataset_no_chage_data = train_dataset_no_chage_data.sample(frac=1, random_state=SEED)
    origin_input_size = train_dataset_no_chage_data.shape[0]
    train_dataset_no_chage_data = train_dataset_no_chage_data.iloc[:int(SAME_INOUTPUT_RATE * origin_input_size)]
    make_input_output_same = lambda x: x['output'].copy()
    train_dataset_no_chage_data['input'] = train_dataset_no_chage_data.apply(make_input_output_same, axis=1)
    train_dataset = train_dataset.append(train_dataset_no_chage_data, ignore_index=True)
    train_dataset = train_dataset.sample(frac=1, random_state=SEED).reset_index(drop=True)

#-------------------------------------------
#make sentence for token list
make_sentence = lambda x : " ".join(x).lower()

train_dataset['input_sentence'] = train_dataset['input'].apply(make_sentence)
train_dataset['output_sentence'] = train_dataset['output'].apply(make_sentence)
val_dataset['input_sentence'] = val_dataset['input'].apply(make_sentence)
val_dataset['output_sentence'] = val_dataset['output'].apply(make_sentence)

#------------------------------------------
#make is_labels_need_change for previous metrics
#also calculate token_align_ids here, only for input, so use encode_tokenizer
labels_word2id = {}
def make_special_input_output(x):
    assert len(x['input']) == len(x['output'])
    
    input_token = []
    output_token = []
    
    for i in range(len(x['input'])):
        input_token.append(x['input'][i].split())
        output_token.append(x['output'][i].split())

    #then we make is_labels_need_change
    result = {}
    is_labels_need_change = []
    #align_ids for bos token is 0
    align_ids = [0]
    for i in range(len(input_token)):
        if len(input_token[i]) == len(output_token[i]):
            for j in range(len(output_token[i])):
                #add token into labels_word2id dict
                if output_token[i][j] not in labels_word2id.keys():
                    labels_word2id[output_token[i][j]] = len(labels_word2id)
                #they are the same, no need change, and align_ids + 0(keep) * ids_num
                if output_token[i][j].lower() == input_token[i][j].lower():
                    is_labels_need_change.append([labels_word2id[output_token[i][j]], 0])
                    length = len(encoder_tokenizer(input_token[i][j].lower(), add_special_tokens=False).input_ids)
                    align_ids.extend(list(np.zeros(length, dtype = np.int8)))
                #they are diff, need change, and align_ids + 1(norm) * ids_num
                else:
                    is_labels_need_change.append([labels_word2id[output_token[i][j]], 1])
                    length = len(encoder_tokenizer(input_token[i][j].lower(), add_special_tokens=False).input_ids)
                    align_ids.extend(list(np.ones(length, dtype = np.int8)))
        else:
            for j in range(len(output_token[i])):
                #add token into labels_word2id dict
                if output_token[i][j] not in labels_word2id.keys():
                    labels_word2id[output_token[i][j]] = len(labels_word2id)
                #they are diff, need change
                is_labels_need_change.append([labels_word2id[output_token[i][j]], 1])
            for j in range(len(input_token[i])):
                #they are diff, align_ids + 1(norm) * ids_num
                length = len(encoder_tokenizer(input_token[i][j].lower(), add_special_tokens=False).input_ids)
                align_ids.extend(list(np.ones(length, dtype = np.int8)))
    
    #align_ids for eos token is 0
    align_ids.append(0)
    #pad 0 to max_encoder_length
    while len(align_ids) < encoder_length : align_ids.append(0)
    
    return is_labels_need_change, align_ids

train_dataset[['is_labels_need_change', 'align_ids']] = train_dataset.apply(make_special_input_output, axis=1, result_type="expand")
val_dataset[['is_labels_need_change', 'align_ids']] = val_dataset.apply(make_special_input_output, axis=1, result_type="expand")


train_dataset = Dataset.from_pandas(train_dataset)
val_dataset = Dataset.from_pandas(val_dataset)

labels_id2word = {v: k for k, v in labels_word2id.items()}
#----------------------------------------------------------------------------------------------------
# map data correctly
def map_to_encoder_decoder_inputs(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    inputs = encoder_tokenizer(batch["input_sentence"], padding="max_length", truncation=True, max_length=encoder_length)
    outputs = decoder_tokenizer(batch["output_sentence"], padding="max_length", truncation=True, max_length=decoder_length)

    batch["input_ids"] = inputs.input_ids
    if is_alignEmbed:
        batch["token_align_ids"] = batch["align_ids"]
    
    batch["is_labels_need_change"] = batch["is_labels_need_change"]
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

if is_alignEmbed:
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels", "is_labels_need_change", "token_align_ids"],
    )
else:
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels", "is_labels_need_change"],
    )

# same for validation dataset
val_dataset = val_dataset.map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size, remove_columns=["input_sentence", "output_sentence"],
)

if is_alignEmbed:
    val_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels", "is_labels_need_change", "token_align_ids"],
    )
else:
    val_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels", "is_labels_need_change"],
    )
        
#-----------------------------------------------------------------------------------
# load metrics for validation
# Calculate previous_metrics score of a corpus.
def previous_metrics(pred):
    global record
    pred_ids = pred.predictions
    is_labels_need_change = pred.is_labels_need_change

    def batch_convert_ids_to_tokens(sequences, **kwargs):
        return [decoder_tokenizer.decode(seq, **kwargs) for seq in sequences]

    pred_tokens = batch_convert_ids_to_tokens(pred_ids, skip_special_tokens=True)

    assert len(pred_tokens) == len(is_labels_need_change)
    norm_correct, norm_wrong, keep_correct, keep_wrong = 0.0, 0.0, 0.0, 0.0

    for pred_tokens, oracle_tokens in zip(pred_tokens, is_labels_need_change):
        pred_tokens = pred_tokens.split()
        sent_length = len(oracle_tokens)

        while len(pred_tokens) < len(oracle_tokens) : pred_tokens.append("<PAD>")        
        for i in range(sent_length):
            pred_token = pred_tokens[i]
            oracle_token = labels_id2word[int(oracle_tokens[i][0])]
            token_need_change = oracle_tokens[i][1]
            #norm-correct：需要改且改对的
            if token_need_change == 1 and oracle_token.lower() == pred_token.lower() and oracle_token.strip():
                norm_correct += 1
            #norm-wrong ：需要改但没改对的
            if token_need_change == 1 and oracle_token.lower() != pred_token.lower() and oracle_token.strip():
                norm_wrong += 1
            #keep-correct：不需要改且没有改的
            if token_need_change == 0 and oracle_token.lower() == pred_token.lower() and oracle_token.strip():
                keep_correct += 1
            #keep-wrong ：不需要改但是改了的
            if token_need_change == 0 and oracle_token.lower() != pred_token.lower() and oracle_token.strip():
                keep_wrong += 1
    #所有不需要修改token的正确率
    correct_of_all_need_keep = keep_correct / (keep_correct + keep_wrong)

    results = {}
    results["keep_correct"] = keep_correct
    results["keep_wrong"] = keep_wrong
    results["correct_of_all_need_keep"] = correct_of_all_need_keep
    
    if TWEET_COPY_TASK == 0:
        #所有需要修改token的正确率
        correct_of_all_need_norm = norm_correct / (norm_correct + norm_wrong)
        #输出正确token的正确率
        correct_token = (norm_correct + keep_correct) / (norm_correct + norm_wrong + keep_correct + keep_wrong)
        results["norm_correct"] = norm_correct
        results["norm_wrong"] = norm_wrong
        results["correct_of_all_need_norm"] = correct_of_all_need_norm
        results["correct_token"] = correct_token
    return results

def compute_metrics(pred):
    #Calculate previous_metrics score of a corpus
    previous_metrics_results = previous_metrics(pred)
    
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
    
    if TWEET_COPY_TASK == 1:
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
            "keep_correct": round(previous_metrics_results['keep_correct'], 4),
            "keep_wrong": round(previous_metrics_results['keep_wrong'], 4),
            "correct_of_all_need_keep": round(previous_metrics_results['correct_of_all_need_keep'], 4),
        }
    else:
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
            "norm_correct": round(previous_metrics_results['norm_correct'], 4),
            "norm_wrong": round(previous_metrics_results['norm_wrong'], 4),
            "keep_correct": round(previous_metrics_results['keep_correct'], 4),
            "keep_wrong": round(previous_metrics_results['keep_wrong'], 4),
            "correct_of_all_need_norm": round(previous_metrics_results['correct_of_all_need_norm'], 4),
            "correct_of_all_need_keep": round(previous_metrics_results['correct_of_all_need_keep'], 4),
            "correct_token": round(previous_metrics_results['correct_token'], 4),
        }
#----------------------------------------------------------------------------------------
# begin train
print("begin train!!!!!")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_from_generate=True,
    evaluate_during_training=True,
    do_train=True,
    do_eval=True,
#     logging_steps=1000,
#     save_steps=5000,
#     eval_steps=2000,
#     overwrite_output_dir=True,
#     warmup_steps=1000,
#     save_total_limit=3,
#     num_train_epochs=200,
    logging_steps=200,
    save_steps=1000,
    eval_steps=200,
    overwrite_output_dir=True,
    warmup_steps=100,
    save_total_limit=3,
    num_train_epochs=20,
    fp16=True,
    seed=SEED,
    learning_rate=LR,
    run_name=RUN_NAME,
)


# instantiate trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    my_trainer = True
)


# start training
trainer.train()
