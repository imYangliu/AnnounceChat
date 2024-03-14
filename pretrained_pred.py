#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-chat-7b", trust_remote_code=True)
# 设置`torch_dtype=torch.float16`来将模型精度指定为torch.float16，否则可能会因为您的硬件原因造成显存不足的问题。
model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-chat-7b", device_map="cuda",trust_remote_code=True, torch_dtype=torch.float16)
# (可选) 如果在低资源设备上，可以通过bitsandbytes加载4-bit或8-bit量化的模型，进一步节省GPU显存.
  # 4-bit 量化的 InternLM 7B 大约会消耗 8GB 显存.
  # pip install -U bitsandbytes
  # 8-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_8bit=True)
  # 4-bit: model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, load_in_4bit=True)
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
# 模型输出：你好！有什么我可以帮助你的吗？
# response, history = model.chat(tokenizer, "请提供三个管理时间的建议。", history=history)


# ## 小范围样本实践
# 书生：<https://github.com/InternLM/InternLM/blob/main/README_zh-CN.md>
# 

# In[7]:


get_ipython().run_line_magic('pip', 'install xlrd')


# In[1]:


# 数据预处理
from tqdm import tqdm
import pandas as pd


# In[3]:


df = pd.read_excel('./answer.xlsx')
pred_answer_list = []

for i, row in tqdm(df.iterrows()):
    question_type =  row['领域分类']
    question =  row['评测问题']
    answer = row['答案']
    response, history = model.chat(tokenizer, question, history=[])
    pred_answer_list.append(response)


# In[10]:


get_ipython().run_line_magic('pip', 'install openpyxl')


# In[14]:


# 预测后结果处理
df.insert(loc=len(df.columns), column='pred_answer', value=pred_answer_list)
# df
df.to_excel('answer.xlsx',index=False)


# In[4]:


df2 = pd.read_excel('./data/2024-02-28-公告测评集-选项.xls')
pred_answer_choice_list = []

for i, row in tqdm(df2.iterrows()):
    question_type = row['领域分类']
    prompt = '''下面是几条对上述公告相关的理解或问题答案，请选出一条或多条理解到位、答案准确的选项。尽量用一句话概括。\n\n'''
    question = row['评测问题'] + '\n\n'
    choice = (
        "选项A:" + str(row['选项A']), "选项B:" + str(row['选项B']), "选项C:" + str(row['选项C']), "选项D:" + str(row['选项D'])
    )
    choice_str = "\n".join(choice)
    response, history = model.chat(tokenizer, question + prompt + choice_str , history=[])
    pred_answer_choice_list.append(response)


# In[39]:


# 预测后结果处理
df2 = pd.read_excel('./data/2024-02-28-公告测评集-选项.xls')
df2.insert(loc=len(df2.columns), column='pred_answer', value=pred_answer_choice_list)
df2.to_excel('answer_choice.xlsx',index=False)

