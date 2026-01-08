import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

SEED = 42
torch.manual_seed(SEED)


model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

out_greedy = model.generate(
    **inputs,
    max_length=50
)
txt_greedy = tokenizer.decode(out_greedy[0], skip_special_tokens=True)
print(txt_greedy)

#
# 5-c 
#

def generate_once(seed, repetition_penalty=None, temperature=0.7):
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

for s in [1, 2, 3, 4, 5]:
    print(f"SEED {s}")
    print(generate_once(s))
    print("-" * 40)

#
# 5-d
#

print("Sans pénalité:")
print(generate_once(1))

print("\nAvec pénalité:")
print(generate_once(1, repetition_penalty=2.0))

#
# 5-e
#

print("\nTempérature très basse (0.1)")
print(generate_once(1, temperature=0.1))

print("\n Température très élevée (2.0)")
print(generate_once(1, temperature=2.0))

#
# 5-f
#

out_beam = model.generate(
    **inputs,
    max_length=50,
    num_beams=5,
    early_stopping=True
)
txt_beam = tokenizer.decode(out_beam[0], skip_special_tokens=True)
print(txt_beam)

#
# 5-g
#

for beams in [10, 20]:
    start = time.time()
    out = model.generate(
        **inputs,
        max_length=50,
        num_beams=beams,
        early_stopping=True
    )
    elapsed = time.time() - start
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"\nBeam search num_beams={beams}")
    print(txt)
    print("Temps:", elapsed)