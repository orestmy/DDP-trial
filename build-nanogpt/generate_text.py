import torch
from torch.nn import functional as F
import tiktoken
from data_and_model import GPT, GPTConfig
import os


def generate_text(model, device, prompt, max_length=400, num_return_sequences=5, seed=42):
    """
    Generates text from a pre-trained GPT model given a prompt.

    Args:
        model: The pre-trained GPT model.
        device: The device to run the model on (e.g., 'cuda' or 'cpu').
        prompt: The prompt string to start generation from.
        max_length (int): The maximum length of the generated sequence.
        num_return_sequences (int): The number of sequences to generate.
        seed (int): The random seed for generating text.
    """
    # model = GPT(config)
    model = model.to(device)
    model.eval()
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)
            # print(x.shape)
            # print(logits[0].shape)
            logits = logits[0][:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)
        print("="*50)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = GPTConfig(vocab_size=50304)
    model = GPT(config)
    model_path = os.path.join(os.path.dirname(__file__), 'model_steps_300.pt')
    state_dict = torch.load(model_path)

    # Remove the "_orig_mod." prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "")
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict)
    generate_text(model, device, "Hello, I am a language model")
