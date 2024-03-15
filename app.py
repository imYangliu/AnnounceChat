import gradio as gr
from peft import PeftConfig, PeftModelForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
peft_config = PeftConfig.from_pretrained('./trained_model/')
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-chat-7b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-chat-7b", device_map="cuda",trust_remote_code=True, torch_dtype=torch.float16)
model = PeftModelForCausalLM.from_pretrained(model, './trained_model', config=peft_config)
model = model.eval()


# 聊天功能
def chat(message, history):
    response, history = model.chat(tokenizer, message , history=history)
    return response

gr.ChatInterface(chat).launch()