import transformers
import torch
import convert

device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device == "cuda:0":
    transformers.models.gptj.modeling_gptj.GPTJBlock = convert.GPTJBlock

def init():
    global model
    global tokenizer
    
    print("Tokenizer loading on cpu...")
    tokenizer = transformers.AutoTokenizer.from_pretrained("yvan237/cedille-GPT-J-6B-8bit")
    print("done")

    print("Model loading on cpu...")
    if device == "cuda:0":
        print("convert version...")
        model = convert.GPTJForCausalLM.from_pretrained("yvan237/cedille-GPT-J-6B-8bit")
    else:
        print("initial version...")
        model = transformers.GPTJForCausalLM.from_pretrained("yvan237/cedille-GPT-J-6B-8bit")
    print("done")
    
    if device == "cuda:0":
        print("Model passing on gpu...")
        model.to(device)
        

def inference(prompt):
    global model
    global tokenizer

    # Parse out your arguments
    if prompt == None:
        return {'message': "No prompt provided"}
    
    prompt = "Je suis un jeune étudiant en"
    # Tokenize input
    input_tokens = tokenizer(prompt, return_tensors='pt')
    input_tokens = {key: value.to(device) for key, value in input_tokens.items()}

    # Run the model
    output = model.generate(**input_tokens, min_length=14, max_length=14, do_sample=True)

    # Decode output token
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    result = {"output": output_text}

    # Return the results as a dictionary
    return result

 
# prompt = {key: value.to(device) for key, value in prompt.items()}

# out = gpt.generate(**prompt, min_length=128, max_length=128, do_sample=True)

# tokenizer.decode(out[0], skip_special_tokens=True)