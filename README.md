# AnnounceChat

## 数据预处理

将简答题、选择题Excel数据集加入Prompt合并转为`json`文件保存，用作P-Tuning的训练数据集。

```python
import pandas as pd
 
# DataFrame to Json
# 加载简答题数据集，统一数据类型
df_short_ans = pd.read_excel('./data/2024-02-28-公告测评集.xls', header=0)
df_short_ans['answer'] = df_short_ans['answer'].astype(str)
 
# 加载选择题数据集
df_choice = pd.read_excel('./data/2024-02-28-公告测评集-选项.xls', header=0)
df_choice['prompt'] = ''
 
# 加入Prompt，合并问题和选项
for i, row in df_choice.iterrows():
    prompt = '''下面是几条对上述公告相关的理解或问题答案，请选出一条或多条理解到位、答案准确的选项。\n\n'''
    question = row['评测问题'] + '\n\n'
    choice = (
        "选项A:" + str(row['选项A']), "选项B:" + str(row['选项B']), "选项C:" + str(row['选项C']), "选项D:" + str(row['选项D'])
    )
    choice_str = "\n".join(choice)
    df_choice.loc[i, 'question'] = question + prompt + choice_str
 
df_choice_2 = df_choice[['领域分类', 'question', '答案']].copy()
df_choice_2.rename(columns={'领域分类': 'type', '答案': 'answer'}, inplace=True)
 
# 合并选择题、简答题数据集并保存为待训练数据集 
df_combine = pd.concat([df_short_ans, df_choice_2], axis=0, ignore_index=True)
df_combine.to_json('./data/data_combine_train.json',orient='records')

```

## 提示词设计

### 文本问答

经过我们的测试，仅将数据中的原本问题提供给模型，模型通常会输出较长的分析，不符合答案的要求。因此，我们的提示词采用两个设计：

- 明确地指定`以下的问题，请用一句话简要地回答`，以限制模型输出的长度。
- 以few-shot的形式，给定模型从数据集中选出的一个问题与回答的例子，使得模型能够更好地理解所需回答的形式和风格。

```python
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
```

### 选项选择

对于选择题，我们的提示词融合了背景信息和选择题的要求，实现限制模型的输出形式并帮助模型理解选择题的背景。

```python
import pandas as pd

# DataFrame to Json
df_short_ans = pd.read_excel('./data/2024-02-28-公告测评集.xls', header=0)
df_short_ans['answer'] = df_short_ans['answer'].astype(str)
# df_short_ans.to_json('./data/data_short_ans_train.json',orient='records')

df_choice = pd.read_excel('./data/2024-02-28-公告测评集-选项.xls', header=0)
df_choice['prompt'] = ''
for i, row in df_choice.iterrows():
    prompt = '''下面是几条对上述公告相关的理解或问题答案，请选出一条或多条理解到位、答案准确的选项。\n\n'''
    question = row['评测问题'] + '\n\n'
    choice = (
        "选项A:" + str(row['选项A']), "选项B:" + str(row['选项B']), "选项C:" + str(row['选项C']), "选项D:" + str(row['选项D'])
    )
    choice_str = "\n".join(choice)
    df_choice.loc[i, 'question'] = question + prompt + choice_str


```

## 高效微调

我们采用了P-Tuning这种高效微调技术，固定模型本身的参数并引入少量可学习参数，在训练集上对模型整体进行微调。

具体而言，P-Tuning是一种用于自然语言处理任务的微调方法，它通过在预训练的语言模型中引入可训练的prompt（提示词）参数来提高模型性能。这种方法的主要思想是，相比于直接修改模型内部的参数，通过微调一小部分专门设计的prompt参数，可以更有效地将模型的知识适应到特定的任务上。

P-Tuning通过为每个任务设计特定的prompt模板，提供了一种灵活的方式来适应不同的任务需求，增强了模型的泛化能力。与传统的微调方法相比，P-Tuning只需要优化相对较少的参数（prompt中的参数），从而减少了计算资源的需求，提高了训练效率。总的来说，这种微调方式比较适合我们所面临的场景。

首先，我们使用HuggingFace提供的`peft`实现P-tuning包装下的internLM模型：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PromptEncoderConfig, get_peft_model, TaskType

# 加载InternLM预训练模型和Tokenizer
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-chat-1_8b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-chat-1_8b", device_map="cuda",trust_remote_code=True, torch_dtype=torch.float16)
config = PromptEncoderConfig(
    peft_type="P_TUNING",
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=10,
    token_dim=2048,
    encoder_reparameterization_type="MLP",
    encoder_hidden_size=2048,
)

model = get_peft_model(model, config)
```

随后，我们准备训练数据集：

```python
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from typing import Dict
from datasets import Dataset, load_dataset
import numpy as np


def encode_fn(text, tokenizer, max_length,return_attention_mask=False):
    return tokenizer(text, max_length=max_length, padding="max_length", truncation=True,return_attention_mask=return_attention_mask)

# 加载并编码Json数据集
def get_dataset(file: str, split: str, encode_fn: callable, encode_args: dict,  cache_dir: str='.cache') -> Dataset:
    eos_token = tokenizer.eos_token
    dataset = load_dataset('json', data_files=file, split=split, cache_dir=cache_dir)

    def merge_prompt_and_responses(sample: dict):
        encoded_prompt = tokenizer(sample['question'] + eos_token, truncation=False, padding=True, return_attention_mask=True)
        encoded_response = tokenizer(sample['answer'] + eos_token, truncation=False, padding=True, return_attention_mask=False)
        encoded_q_type = tokenizer(sample['type'] + eos_token, truncation=False, padding=True, return_attention_mask=True)
        return {
            'input_ids': encoded_prompt.input_ids,
            'q_type': encoded_q_type.input_ids,
            'labels': encoded_response.input_ids,       
            'attention_mask': encoded_prompt.attention_mask,
            'q_type_attention_mask' : encoded_q_type.attention_mask,
        }
    
    dataset = dataset.map(merge_prompt_and_responses)
    return dataset


dataset = get_dataset(
    file='./data/data_combine_train.json', 
    split="train", 
    encode_fn=encode_fn, 
    encode_args={"tokenizer": tokenizer, "max_length": 1024}, 
    cache_dir=".cache"
)
```

最后，我们在准备好的训练集上对P-Tuning包装的模型进行微调：

```python
import time

# 设置Trainer参数
args = TrainingArguments(
    output_dir='./train_result/',
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    auto_find_batch_size=True,
    gradient_accumulation_steps=10,
    learning_rate=1e-3,
    logging_steps=10,
    num_train_epochs=10,
    log_level='info',
    save_steps=10,
    save_total_limit=20,
    # fp16=,
    # logging_first_step=,
    warmup_steps=50,
    seed=42,
)

trainer = Trainer(
    model = model,
    args = args,
    train_dataset = dataset,
    data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm=False)
)

# 进行训练
trainer.train()
 
loss_log = pd.DataFrame(trainer.state.log_history)
loss_log.to_csv(f"{'./logs'}/p_tune_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")

model.save_pretrained('./trained_model')

# print(model)

```

## 结果输出

使用`chat`函数测试微调后的模型，将Prompt和问题逐个放入模型中获得回答，再将结果保存为`evaluation_result.xlsx`文件。

```python
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
```

下面是一些示例的输出。

### 文本问答

|                             问题                             |                             答案                             |                          我们的输出                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 你是一名人力资源管理专家，评估上港集团股权激励计划设计的合理性及对激励对象的选择标准··· | 限制性股票授予的条件和解锁标准设置得当，有利于激励对象长期贡献。 | 上港集团股权激励计划的设计合理，激励对象的选择标准明确，有助于吸引和留住优秀人才，提升企业竞争力。 |
| 你是一名人力资源管理专家，东方财富回购股份用于员工持股计划或股权激励计划，这一措施对公司内部管理有何潜在影响？··· | 可以提升员工的归属感和忠诚度，增加他们对公司未来发展的信心。股权激励可以激发员工的工作热情和创新能力，提高公司整体的竞争力。 | 回购股份用于员工持股计划或股权激励计划，可以提升员工对公司的归属感和忠诚度，促进人才稳定，同时可能增加员工的工作积极性和创造力，有利于公司长期发展。 |
| 你是一名公司治理专家，根据上汽集团选举华恩德先生为职工代表董事的情况，请评估职工代表董事在公司治理中的作用··· |         直接参与公司重大决策，反映职工的意见和需求。         | 职工代表董事在公司治理中起着维护员工利益、增强企业民主管理、促进企业和谐发展的重要作用。 |

### 选项选择

|                             问题                             | 答案 |                          我们的输出                          |
| :----------------------------------------------------------: | :--: | :----------------------------------------------------------: |
| 你是一名人力资源管理专家，东方财富回购股份用于员工持股计划或股权激励计划，这一措施对公司内部管理有何潜在影响？根据选项来选择合适的分析··· | A、B | 选项A和选项B都正确。回购股份用于员工持股计划或股权激励计划可以提升员工的归属感和忠诚度，增加他们对公司未来发展的信心，同时股权激励也可以激发员工的工作热情和创新能力，提高公司整体的竞争力。 |
| 你是一名公司治理专家，根据上汽集团选举华恩德先生为职工代表董事的情况，请评估职工代表董事在公司治理中的作用。 |  B   |      选项B:直接参与公司重大决策，反映职工的意见和需求。      |
| 你是一名财务分析师，根据东方财富证券股份有限公司2023年度第十三期短期融资券的发行情况，请评估该融资活动的目的和潜在影响。 |  A   |          选项A:增加公司的流动性，满足短期资金需求。          |

