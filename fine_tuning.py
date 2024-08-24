from unsloth import FastLanguageModel
import torch
from unsloth import apply_chat_template,standardize_sharegpt,to_sharegpt
import pandas as pd
from datasets import Dataset
from datasets import load_dataset

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported



max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)




with open("fineTuningDataImperfect.txt", "r", encoding="utf-8", errors="ignore") as fin:
    lines = fin.readlines()
temp = ""
for line in lines :
    temp = temp + line
blocks = temp.strip().split('@@')

# List to store the extracted data
data = []

# Parse each block
for block in blocks:
    if block.strip():  # Skip any empty blocks
        lines = block.strip().split('&&&')
        status = lines[1].strip().replace('[','').replace(']','').replace("'","").replace("'","")
        fact_line = lines[2].strip().replace('\n','')
        description_line =lines[3].strip() 
        input = "The fact is : ["+fact_line+"] and the description is ["+description_line+"]"
        data.append({
            'output': status,
            'input': input,
            'fact' : fact_line,
            'description':description_line
        })
print(block)
# Create a DataFrame
df = pd.DataFrame(data)
print(df.head(2))
dataset = Dataset.from_pandas(df)

dataset = to_sharegpt(
    dataset,
    merged_prompt = "The fact is : [{fact}] and the description is [{description}]",
    output_column_name = "output",
)

dataset = standardize_sharegpt(dataset)

chat_template_llama3 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

### Instruction:\n{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Response:\n{OUTPUT}<|eot_id|><|start_header_id|>user<|end_header_id|>

### Instruction:\n{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Response:\n{OUTPUT}<|eot_id|>"""
chat_template1 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

### Instruction:\n{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Response:\n{OUTPUT}<|eot_id|><|start_header_id|>user<|end_header_id|>

### Instruction:\n{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

### Response:\n{OUTPUT}<|eot_id|>"""
chat_template_gemma ="""{SYSTEM}

### Instruction:
{INPUT}

### Response:
{OUTPUT}

### Instruction:
{INPUT}

### Response:
{OUTPUT}
"""
system_messageBaseModel = 'Answer correct if you find the fact in the description and wrong if you do not '
system_message = "You are an expert at identifying if a given fact is correctly mentionned in a description. You answer shortly by saying correct if it is correctly mentionned or wrong if it is not mentionned or wrongly mentionned. You ignore every other informations in the description."
system_message2 = 'You are an expert at identifying if a given fact is correctly mentionned in a description. \n You start by identifying the part of the description that is about the fact or if it is not mentionned you say that the description does not talk about the given fact. \n Then you answer shortly by saying "Correct" if it is correctly mentionned or "Wrong" if it is not mentionned or wrongly mentionned. \n You ignore every other informations in the description that are not relevant to the task.'


dataset = apply_chat_template(
    dataset,
    tokenizer = tokenizer,
    chat_template = chat_template_llama3,
    default_system_message = system_messageBaseModel ,
)



train_test_split = dataset.train_test_split(test_size=0.2)  # 20% for evaluation
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']
print(len(train_dataset))




trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs = 1,
        warmup_steps = 5,
       # max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)


# Define the filename where the stats will be written
filename = "gpu_memory_stats.txt"

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

with open(filename, "w") as file:
    file.write(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.\n")
    file.write(f"{start_gpu_memory} GB of memory reserved.\n")


#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
trainer_stats = trainer.train()

#model.save_pretrained_gguf("model", tokenizer,)
#model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
#model.save_pretrained_gguf("model", tokenizer, quantization_method = "q5_k_m")
#model.save_pretrained_gguf("model", tokenizer, quantization_method = "quantized")



with open(filename, "a") as file:
    file.write(f"{trainer_stats.metrics['train_runtime']} seconds used for training.\n")
    file.write(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.\n")
    file.write(f"Peak reserved memory = {used_memory} GB.\n")
    file.write(f"Peak reserved memory for training = {used_memory_for_lora} GB.\n")
    file.write(f"Peak reserved memory % of max memory = {used_percentage} %.\n")
    file.write(f"Peak reserved memory for training % of max memory = {lora_percentage} %.\n")
