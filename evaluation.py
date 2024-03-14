import json
from peft import PeftConfig, PeftModelForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
peft_config = PeftConfig.from_pretrained('./trained_model/')
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-chat-7b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-chat-7b", device_map="cuda",trust_remote_code=True, torch_dtype=torch.float16)
model = PeftModelForCausalLM.from_pretrained(model, './trained_model', config=peft_config)
model = model.eval()


answer_list = []
# 打开 JSON 文件
with open('./data/data_combine_test.json', 'r') as file:
    # 逐行读取文件内容
    for line in file:
        # 解析每行为 JSON 对象
        json_list = json.loads(line)
        # 遍历 JSON 对象进行处理
        for item in tqdm(json_list):
            tp  = item['type']
            q  = item['question']
            a  = item['answer']
            response, history = model.chat(tokenizer, q , history=[])
            answer_list.append({
                'type':tp,
                'question':q,
                'answer':a,
                'pred':response
            })

df = pd.DataFrame(answer_list)
df.to_excel('./evaluation_result.xlsx')

# df = pd.read_excel('./data/2024-02-28-公告测评集.xls')
# pred_answer_list = []

# for i, row in tqdm(df.iterrows()):
#     question_type =  row['领域分类']
#     question =  row['评测问题']
#     answer = row['答案']
#     response, history = model.chat(tokenizer, question, history=[])
#     pred_answer_list.append(response)


# df2 = pd.read_excel('./data/2024-02-28-公告测评集-选项.xls')
# pred_answer_choice_list = []

# for i, row in tqdm(df2.iterrows()):
#     question_type = row['领域分类']
#     prompt = '''下面是几条对上述公告相关的理解或问题答案，请选出一条或多条理解到位、答案准确的选项。尽量用一句话概括。\n\n'''
#     question = row['评测问题'] + '\n\n'
#     choice = (
#         "选项A:" + str(row['选项A']), "选项B:" + str(row['选项B']), "选项C:" + str(row['选项C']), "选项D:" + str(row['选项D'])
#     )
#     choice_str = "\n".join(choice)
#     response, history = model.chat(tokenizer, question + prompt + choice_str , history=[])
#     pred_answer_choice_list.append(response)
