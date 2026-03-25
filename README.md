# Marketing-description-generator
This generates marketing descriptions for a completed project using Llama 7b model
# Code to read csv file into Colaboratory:
!pip install -U -q PyDrive2
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
%%capture
# Installs Unsloth, Xformers (Flash Attention) and all other packages!
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
from unsloth import FastLanguageModel
import torch
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
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
==((====))==  Unsloth 2024.8: Fast Llama patching. Transformers = 4.44.0.
   \\   /|    GPU: Tesla T4. Max memory: 14.748 GB. Platform = Linux.
O^O/ \_/ \    Pytorch: 2.3.1+cu121. CUDA = 7.5. CUDA Toolkit = 12.1.
\        /    Bfloat16 = FALSE. FA [Xformers = 0.0.26.post1. FA2 = False]
 "-____-"     Free Apache license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
model.safetensors: 100%
 5.70G/5.70G [00:53<00:00, 517MB/s]
generation_config.json: 100%
 198/198 [00:00<00:00, 12.4kB/s]
tokenizer_config.json: 100%
 50.6k/50.6k [00:00<00:00, 2.70MB/s]
tokenizer.json: 100%
 9.09M/9.09M [00:00<00:00, 38.1MB/s]
special_tokens_map.json: 100%
 350/350 [00:00<00:00, 26.4kB/s]
 We now add LoRA adapters so we only need to update 1 to 10% of all parameters!
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
Data Prep
We now use the Alpaca dataset from yahma, which is a filtered version of 52K of the original Alpaca dataset. You can replace this code section with your own data prep.

[NOTE] To train only on completions (ignoring the user's input) read TRL's docs here.

[NOTE] Remember to add the EOS_TOKEN to the tokenized output!! Otherwise you'll get infinite generations!

If you want to use the llama-3 template for ShareGPT datasets, try our conversational notebook.

For text completions like novel writing, try this notebook.
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass


link = 'https://drive.google.com/file/d/1yDTrTsnrIhTpyDqKX4d-Vv4MFiTCjcbt/view?usp=sharing'

id = link.split('//')[1].split('/')[3]
print (id) # Verify that you have everything after '='
#id='1IvgF88ccQONnlqC_oPu4EtU7fGm3RTWj'
downloaded = drive.CreateFile({'id':id})
downloaded.GetContentFile('demofile.json')
from os.path import dirname
from datasets import load_dataset
dataset = load_dataset('json', data_files='demofile.json')
dataset = dataset.map(formatting_prompts_func, batched = True,)
Train the model
Now let's use Huggingface TRL's SFTTrainer! More docs here: TRL SFT docs. We do 60 steps to speed things up, but you can set num_train_epochs=1 for a full run, and turn off max_steps=None. We also support TRL's DPOTrainer!
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset['train'],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        #num_train_epochs=2,
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
#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs = 1
   \\   /|    Num examples = 2,194 | Num Epochs = 1
O^O/ \_/ \    Batch size per device = 2 | Gradient Accumulation steps = 4
\        /    Total batch size = 8 | Total steps = 60
 "-____-"     Number of trainable parameters = 41,943,040
 [60/60 13:51, Epoch 0/1]
Step	Training Loss
1	3.593000
2	3.428500
3	3.377100
4	3.255500
5	3.222700
6	3.088000
7	2.824000
8	2.568900
9	2.320400
10	2.232400
11	2.084300
12	1.886000
13	1.838500
14	1.813500
15	1.858100
16	1.715100
17	1.722900
18	1.734700
19	1.676000
20	1.588300
21	1.750200
22	1.542900
23	1.680700
24	1.631800
25	1.843000
26	1.593600
27	1.504900
28	1.689100
29	1.480200
30	1.479900
31	1.510900
32	1.541900
33	1.732800
34	1.649700
35	1.453300
36	1.447400
37	1.447900
38	1.416500
39	1.434100
40	1.477500
41	1.430600
42	1.473400
43	1.460000
44	1.457700
45	1.445000
46	1.361200
47	1.672100
48	1.462800
49	1.477100
50	1.495400
51	1.420700
52	1.378200
53	1.388000
54	1.494500
55	1.506600
56	1.451200
57	1.361700
58	1.543700
59	1.302300
60	1.510200
#Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
851.2157 seconds used for training.
14.19 minutes used for training.
Peak reserved memory = 9.066 GB.
Peak reserved memory for training = 3.453 GB.
Peak reserved memory % of max memory = 61.473 %.
Peak reserved memory for training % of max memory = 23.413 %.
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
     "instruction: Given a following comma separated list of data as input create the marketing description of the data set input to you, summarise with all the key essential data provided and generate the output as marketing description of the projects. The comma separated list of data is: Project Code,Long Name,Client,PM,PD,GBP Contract Value,Region / Country,Notes,Links,Practice Area,Status,Propositions,Capabilities,Topics,PD Team,Sold Date,Open Date,Closed Date,Open Year,Sold Year,",  #input
    "input: LXPTNM, Access modelling for NPT, update of XPTN4,Norwegian Communications Authority (Nkom) (Europe),Matthew Starling,James Allen,251 319,[{Label:Norway TermID:e3e4041f-6302-43a5-9bdc-5c39564b6327}],Country of interest: Norway  Proposition: REG2_Regulatory Economic Costing,%23Name?,Regulation & Policy,Dormant,{ __type : TaxonomyFieldValue:%23Microsoft.SharePoint.Taxonomy   Label : Price Control ＆ Cost Modelling   TermID : b4362775-be9b-4395-9c1d-8657037d12c1 },[{ Label : Cost Modelling   TermID : 64dc4d99-d395-4b5c-b712-c0d4f6f91d91 } { Label : Forecasting ＆ Benchmarking   TermID : e6fd4d49-2aaf-47af-bbf0-133031b46dca } { Label : GIS ＆ Geographic Analysis   TermID : 01d06a75-2ab5-4049-a13d-c88fe233a890 }],[{ Label : Fibre (incl. FTTx  dark fibre  GPON  XG-PON  XGS-PON  DWDM)   TermID : 62868b38-d5d6-4215-99e1-cdd68520474b } { Label : Broadband   TermID : 90a59d2e-fa3a-4ad2-a5f6-6632fee72988 } { Label : Fixed access networks   TermID : 023bc9bc-01bd-4b10-99da-613f96f4467c } { Label : REG: LRIC   TermID : 59b3f98f-4867-4da4-8dff-58478a5b137a }],T03,07/05/2015,02/01/2015,24/04/2020,2015,2015,", #input
    "output: ", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)
# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving
if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# alpaca_prompt = You MUST copy from above!

inputs = tokenizer(
[
    alpaca_prompt.format(
        "What is a famous tall tower in Paris?", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)
if False:
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained("lora_model")
    # Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")
# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer,)
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")
