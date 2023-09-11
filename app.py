import gradio as gr
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 

### Select the language model
model_id = "TheBloke/CodeLlama-7B-Instruct-GPTQ"
# Configuration
runtimeFlag = "cuda:0" #Run on GPU (you can't run GPTQ on cpu)
cache_dir = None # by default, don't set a cache directory. This is automatically updated if you connect Google Drive.
scaling_factor = 1.0 # allows for a max sequence length of 16384*6 = 98304! Unfortunately, requires Colab Pro and a V100 or A100 to have sufficient RAM.


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # rope_scaling = {"type": "dynamic", "factor": scaling_factor}
    )
tokenizer = AutoTokenizer.from_pretrained(model_id)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# max_doc_length = 50
max_context = int(model.config.max_position_embeddings*scaling_factor)
max_doc_length = int(0.75 * max_context)  # max doc length is 75% of the context length
max_doc_words = int(max_doc_length)

def generate_response(user_input,  temperature=0.01, top_p=0.9, logprobs=False):
    torch.cuda.empty_cache()
    max_context = 16384

    max_prompt_len = int(0.85 * max_context)
    max_gen_len = int(0.10 * max_prompt_len)

    # Initialize the conversation with the user's input
    dialog = [
        {
            "role": "user",
            "content": user_input,
        }
    ]

    # Tokenize the dialog
    dialog_tokens = [tokenizer(
        f"{B_INST} {(dialog[0]['content']).strip()} {E_INST}",
        return_tensors="pt",
        add_special_tokens=True
    ).input_ids.to(runtimeFlag)]

    # Append the tokens to prompt_tokens
    prompt_tokens = dialog_tokens

    for i in range(2, len(dialog), 2):
        user_tokens = tokenizer(
            f"{B_INST} {(dialog[i+1]['content']).strip()} {E_INST}",
            return_tensors="pt",
            add_special_tokens=True
        ).input_ids.to(runtimeFlag)
        assistant_w_eos = dialog[i]['content'].strip() + tokenizer.eos_token
        assistant_tokens = tokenizer(
            assistant_w_eos,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to(runtimeFlag)
        tokens = torch.cat([assistant_tokens, user_tokens], dim=-1)
        dialog_tokens.append(tokens)

    prompt_tokens.append(torch.cat(dialog_tokens, dim=-1))




    input_ids = prompt_tokens[0]
    if len(input_ids[0]) > max_prompt_len:
        return "\n\n **The language model's input limit has been reached. Clear the chat and start afresh!**"



    # print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    temperature = 0.1
    generation_output = model.generate(
        input_ids=input_ids,
        do_sample=True,
        max_new_tokens=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    );

    new_tokens = generation_output[0][input_ids.shape[-1]:]
    new_assistant_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip();

    return new_assistant_response

description_html = """
<div style="text-align: center;">
    <p>Chat with LLama 7 Billion, the friendly AI assistant!</p>
    <p>Ask questions, have a conversation, or just say hello.</p>
    <p>Learn more about the LLama 7 Billion model on <a href="https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GPTQ" target="_blank">Hugging Face</a>.</p>
</div>
"""

examples = [
    "Tell me about the best way to cook biryani",
    "Tell me the equation for calculating the area of a circle",
    "Explain the concept of gradient descent in machine learning",
    "What is the theory of relativity in physics?",
    "Who is considered the father of modern mathematics?",
    "Tell me a fun fact about the universe.",
     "tell me who is better Cristiano Ronaldo or Lionel Messi",
]

demo = gr.ChatInterface(fn=generate_response, examples=examples,description=description_html, title="LLama 🦙")
demo.launch(debug=True)
