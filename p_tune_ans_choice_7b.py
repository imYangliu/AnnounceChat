#!/usr/bin/env python
# coding: utf-8

# ## 1.   Load pretrained model and tokenizer

# In[1]:


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PromptEncoderConfig, get_peft_model, TaskType

tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-chat-7b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-chat-7b", device_map="cuda",trust_remote_code=True, torch_dtype=torch.float16)
# model = model.eval()
config = PromptEncoderConfig(
    peft_type="P_TUNING",
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=10,
    token_dim=4096,
    # num_transformer_submodules=1,
    # num_attention_heads=12,
    # num_layers=12,
    encoder_reparameterization_type="MLP",
    encoder_hidden_size=2048,
)

model = get_peft_model(model, config)
print("模型加载后显存占用情况",torch.cuda.memory_summary(device=None, abbreviated=False))
# ## 2.   Prepare dataset

# In[2]:


import pandas as pd

# DataFrame to Json
df_short_ans = pd.read_excel('./data/2024-02-28-公告测评集.xls', header=0)
df_short_ans['answer'] = df_short_ans['answer'].astype(str)
# df_short_ans.to_json('./data/data_short_ans_train.json',orient='records')

prompt = '''以下的问题，请用一句话简要地回答\n
下面是回答的例子。
【问题】："你是一名人力资源专家，针对沈晓苏先生的退休及张小龙先生的推举，评估该决策对公司人力资源策略的影响。
公告：上海汽车集团股份有限公司 八届十三次监事会会议决议公告 本公司监事会及全体监事保证本公告内容不存在任何虚假记载、误导性陈述或者重大 遗漏，
并对其内容的真实性、准确性和完整性承担法律责任。 上海汽车集团股份有限公司第八届监事会第十三次会议通知于 2023 年 10 月 9 日通过传真、电子邮件等形式送达。
本次监事会会议 于 2023 年 10 月 11 日在上海市漕溪北路 400 号会议室召开。会议的 召集、召开符合《公司法》及《公司章程》的有关规定。本次会议应到监事 4 人，
实际出席会议监事 4 人，会议由监事张小龙先生主持。 经与会监事认真审议，表决通过了如下决议： 关于推举公司监事会召集人的议案。 公司第八届监事会主席沈晓苏先生因到龄退休，
已递交辞职申请，请求辞去公司第八届监事会主席、监事职务。 根据《公司章程》第一百四十七条的规定，推举公司监事张小龙先生担任公司第八届监事会召集人。
监事会对沈晓苏先生在任职期间为公司发展所作贡献给予充分的肯定并致以衷心的感谢。 （同意 4 票，反对 0 票，弃权 0 票） 特此公告。 上
海汽车集团股份有限公司 监事会 2023 年 10 月 12 日 """
\n
【回答】：显示了公司对资深员工的尊重和感激，有利于提高员工的归属感和忠诚度。
\n'''

for i, row in df_short_ans.iterrows():
    question = row['question']
    df_short_ans.loc[i, 'question'] = prompt  +  question

df_choice = pd.read_excel('./data/2024-02-28-公告测评集-选项.xls', header=0)
df_choice['prompt'] = ''
for i, row in df_choice.iterrows():
    prompt = '''下面是几条对上述公告相关的理解或问题答案，请选出一条或多条理解到位、答案准确的选项, 用一句话简要地回答。\n\n'''
    question = row['评测问题'] + '\n\n'
    choice = (
        "选项A:" + str(row['选项A']), "选项B:" + str(row['选项B']), "选项C:" + str(row['选项C']), "选项D:" + str(row['选项D'])
    )
    choice_str = "\n".join(choice)
    df_choice.loc[i, 'question'] = question + prompt + choice_str

df_choice_2 = df_choice[['领域分类', 'question', '答案']].copy()
df_choice_2.rename(columns={'领域分类': 'type', '答案': 'answer'}, inplace=True)
# df_choice_2.to_json('./data/data_choice_train.json',orient='records')

df_combine = pd.concat([df_short_ans, df_choice_2], axis=0, ignore_index=True)
df_combine.to_json('./data/data_combine_test.json',orient='records')


# ## 3.   Load dataset and encode

# In[3]:


import pandas as pd
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from typing import Dict
from datasets import Dataset, load_dataset
import numpy as np


def encode_fn(text, tokenizer, max_length,return_attention_mask=False):
    return tokenizer(text, max_length=max_length, padding="max_length", truncation=True,return_attention_mask=return_attention_mask)


def get_dataset(file: str, split: str, encode_fn: callable, encode_args: dict,  cache_dir: str='.cache') -> Dataset:
    """
    Load a dataset
    """
    eos_token = tokenizer.eos_token
    dataset = load_dataset('json', data_files=file, split=split, cache_dir=cache_dir)
    def merge_prompt_and_responses(sample: dict):
        # add an eos token note that end of sentence, using in generate.
        # encoded_prompt = tokenizer([e + eos_token for e in sample['question']], truncation=False, padding=True, return_attention_mask=True)
        # encoded_response = tokenizer([e + eos_token for e in sample['answer']], truncation=False, padding=True, return_attention_mask=False)
        encoded_prompt = tokenizer(sample['question'] + eos_token, truncation=False, padding=True, return_attention_mask=True)
        encoded_response = tokenizer(sample['answer'] + eos_token, truncation=False, padding=True, return_attention_mask=False)
        encoded_q_type = tokenizer(sample['type'] + eos_token, truncation=False, padding=True, return_attention_mask=True)
        # input_ids = [np.array(item + [eos_token_id], dtype=np.uint32) for item in encoded_prompt["input_ids"]]
        # labels = [np.array(item + [eos_token_id], dtype=np.uint32) for item in encoded_response["input_ids"]]
        # prompt = encode_fn(sample['question'] + '[EOS]', return_attention_mask=True)
        # answer = encode_fn(sample['answer'] + '[EOS]', return_attention_mask=False)
        # title = encode_fn(sample['title'] + '[EOS]', **encode_args)
        # print(type(encoded_prompt.input_ids),'\n',type(encoded_prompt.attention_mask),'\n',labels)
        return {
            'input_ids': encoded_prompt.input_ids,
            'q_type': encoded_q_type.input_ids,
            'labels': encoded_response.input_ids,       
            'attention_mask': encoded_prompt.attention_mask,
            'q_type_attention_mask' : encoded_q_type.attention_mask,
        }

    # dataset = dataset.map(merge_prompt_and_responses, batched=True, batch_size=1)
    dataset = dataset.map(merge_prompt_and_responses)
    return dataset

dataset = get_dataset(
    file='./data/data_combine_train.json', 
    split="train", 
    encode_fn=encode_fn, 
    encode_args={"tokenizer": tokenizer, "max_length": 1024}, 
    cache_dir=".cache"
)

print("数据加载后显存占用情况\n", torch.cuda.memory_summary(device=None, abbreviated=False))
# ## 4.   Train

# In[4]:


import time

args = TrainingArguments(
    output_dir='./train_result/',
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    # 不能自动找，否则会出现No executable batch size found, reached zero.
    auto_find_batch_size=False,  # 防止OOM
    # per_device_eval_batch_size=8,
    
    gradient_accumulation_steps=10,
    learning_rate=1e-3,
    logging_steps=10,
    num_train_epochs=10,
    log_level='info',
    save_steps=10,
    save_total_limit=20,
     # 对于多卡模式的修改
    dataloader_prefetch_factor=1,
    dataloader_num_workers = 2,
    # fp16=config.fp16,
    # logging_first_step=config.logging_first_step,
    warmup_steps=50,
    seed=42,
)

# trainer
trainer = Trainer(
    model = model,
    args = args,
    train_dataset = dataset,
    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm=False)
    # data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer)
)
print("训练前显存占用情况", torch.cuda.memory_summary(device=None, abbreviated=False))
# 训练模型
trainer.train()

loss_log = pd.DataFrame(trainer.state.log_history)
loss_log.to_csv(f"{'./logs'}/p_tune_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")

# trainer.save_model('./trained_model')


# In[5]:


model.save_pretrained('./trained_model')


# In[6]:


print(model)


# ## 5.   Evaluation

# In[16]:


from peft import PeftConfig, PeftModelForCausalLM
peft_config = PeftConfig.from_pretrained('./trained_model/')
model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-chat-7b", device_map="cuda",trust_remote_code=True, torch_dtype=torch.float16)
model = PeftModelForCausalLM.from_pretrained(model, './trained_model', config=peft_config)


# In[17]:


print(model)

