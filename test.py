# #!/usr/bin/env python3
# import nlp
# from transformers import RobertaTokenizer, EncoderDecoderModel

# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# model = EncoderDecoderModel.from_pretrained("patrickvonplaten/roberta2roberta-cnn_dailymail-fp16")
# model.to("cuda")

# test_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="test")
# batch_size = 128


# # map data correctly
# def generate_summary(batch):
#     # Tokenizer will automatically set [BOS] <text> [EOS]
#     # cut off at BERT max length 512
#     inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
#     input_ids = inputs.input_ids.to("cuda")
#     attention_mask = inputs.attention_mask.to("cuda")

#     outputs = model.generate(input_ids, attention_mask=attention_mask)

#     # all special tokens including will be removed
#     output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

#     batch["pred"] = output_str

#     return batch


# results = test_dataset.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["article"])

# # load rouge for validation
# rouge = nlp.load_metric("rouge")

# pred_str = results["pred"]
# label_str = results["highlights"]

# rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

# print(rouge_output)

#-------------------------------------------------------------------------------

# #!/usr/bin/env python3
# import nlp
# from transformers import BertTokenizer, EncoderDecoderModel
# tokenizer = BertTokenizer.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")
# model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")
# model.to("cuda")
# test_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="test")
# #test_dataset = load_dataset('cnn_dailymail', "3.0.0", split='test', ignore_verifications=True)
# batch_size = 128
# # map data correctly
# def generate_summary(batch):
#     # Tokenizer will automatically set [BOS] <text> [EOS]
#     # cut off at BERT max length 512
#     inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
#     input_ids = inputs.input_ids.to("cuda")
#     attention_mask = inputs.attention_mask.to("cuda")
#     outputs = model.generate(input_ids, attention_mask=attention_mask)
#     # all special tokens including will be removed
#     output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     batch["pred"] = output_str
#     return batch
# results = test_dataset.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["article"])
# # load rouge for validation
# rouge = nlp.load_metric("rouge")
# pred_str = results["pred"]
# label_str = results["highlights"]
# rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
# print(rouge_output)

#-----------------------------------------------------------------------------
# #!/usr/bin/env python3
# import nlp
# from transformers import RobertaTokenizer, EncoderDecoderModel

# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# model = EncoderDecoderModel.from_pretrained("patrickvonplaten/roberta2roberta-share-cnn_dailymail-fp16")
# model.to("cuda")

# test_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="test")
# batch_size = 128


# # map data correctly
# def generate_summary(batch):
#     # Tokenizer will automatically set [BOS] <text> [EOS]
#     # cut off at BERT max length 512
#     inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
#     input_ids = inputs.input_ids.to("cuda")
#     attention_mask = inputs.attention_mask.to("cuda")

#     outputs = model.generate(input_ids, attention_mask=attention_mask)

#     # all special tokens including will be removed
#     output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

#     batch["pred"] = output_str

#     return batch


# results = test_dataset.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["article"])

# # load rouge for validation
# rouge = nlp.load_metric("rouge")

# pred_str = results["pred"]
# label_str = results["highlights"]

# rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

# print(rouge_output)


#-----------------------------------------------------------------------
#!/usr/bin/env python3
import nlp
from transformers import BertTokenizer, GPT2Tokenizer, EncoderDecoderModel

model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2gpt2-cnn_dailymail-fp16")
model.to("cuda")

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# CLS token will work as BOS token
bert_tokenizer.bos_token = bert_tokenizer.cls_token

# SEP token will work as EOS token
bert_tokenizer.eos_token = bert_tokenizer.sep_token


# make sure GPT2 appends EOS in begin and end
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
gpt2_tokenizer.pad_token = gpt2_tokenizer.unk_token


# set decoding params
model.config.decoder_start_token_id = gpt2_tokenizer.bos_token_id
model.config.eos_token_id = gpt2_tokenizer.eos_token_id
model.config.max_length = 142
model.config.min_length = 56
model.config.no_repeat_ngram_size = 3
model.early_stopping = True
model.length_penalty = 2.0
model.num_beams = 4

test_dataset = nlp.load_dataset("cnn_dailymail", "3.0.0", split="test")
batch_size = 64


# map data correctly
def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = bert_tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = gpt2_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch


results = test_dataset.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["article"])

# load rouge for validation
rouge = nlp.load_metric("rouge")

pred_str = results["pred"]
label_str = results["highlights"]

rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

print(rouge_output)