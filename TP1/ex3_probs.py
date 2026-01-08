import torch
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model.eval()

phrase = "Artificial intelligence is fascinating."
inputs = tokenizer(phrase, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # (1, seq_len, vocab_size)

probs = torch.softmax(logits, dim=-1)

print("Probabilités conditionnelles par token:")
input_ids = inputs["input_ids"][0]

for t in range(1, len(input_ids)):
    tok_id = input_ids[t].item()
    p = probs[0, t - 1, tok_id].item()
    tok_txt = tokenizer.decode([tok_id])
    print(t, repr(tok_txt), f"{p:.3e}")

#
# 4-b
#

log_probs = torch.log_softmax(logits, dim=-1)

total_logp = 0.0
n = 0

for t in range(1, len(input_ids)):
    tok_id = input_ids[t].item()
    lp = log_probs[0, t - 1, tok_id].item()
    total_logp += lp
    n += 1

avg_neg_logp = - total_logp / n
ppl = math.exp(avg_neg_logp)

print("\nPhrase:", phrase)
print("total_logp:", total_logp)
print("avg_neg_logp:", avg_neg_logp)
print("perplexity:", ppl)

#
# 4-c 
#

phrase_bad = "Artificial fascinating intelligence is."
inputs_bad = tokenizer(phrase_bad, return_tensors="pt")

with torch.no_grad():
    outputs_bad = model(**inputs_bad)
    logits_bad = outputs_bad.logits

log_probs_bad = torch.log_softmax(logits_bad, dim=-1)
input_ids_bad = inputs_bad["input_ids"][0]

total_logp_bad = 0.0
n_bad = 0

for t in range(1, len(input_ids_bad)):
    tok_id = input_ids_bad[t].item()
    lp = log_probs_bad[0, t - 1, tok_id].item()
    total_logp_bad += lp
    n_bad += 1

avg_neg_logp_bad = - total_logp_bad / n_bad
ppl_bad = math.exp(avg_neg_logp_bad)

print("\nPhrase:", phrase_bad)
print("total_logp:", total_logp_bad)
print("avg_neg_logp:", avg_neg_logp_bad)
print("perplexity:", ppl_bad)

#
# 4-d 
#

phrase_fr = "L'intelligence artificielle est fascinante."
inputs_fr = tokenizer(phrase_fr, return_tensors="pt")

with torch.no_grad():
    outputs_fr = model(**inputs_fr)
    logits_fr = outputs_fr.logits

log_probs_fr = torch.log_softmax(logits_fr, dim=-1)
input_ids_fr = inputs_fr["input_ids"][0]

total_logp_fr = 0.0
n_fr = 0

for t in range(1, len(input_ids_fr)):
    tok_id = input_ids_fr[t].item()
    lp = log_probs_fr[0, t - 1, tok_id].item()
    total_logp_fr += lp
    n_fr += 1

avg_neg_logp_fr = - total_logp_fr / n_fr
ppl_fr = math.exp(avg_neg_logp_fr)

print("\nPhrase:", phrase_fr)
print("total_logp:", total_logp_fr)
print("avg_neg_logp:", avg_neg_logp_fr)
print("perplexity:", ppl_fr)

#
# 4-e
#

prefix = "Artificial intelligence is"
inp = tokenizer(prefix, return_tensors="pt")

with torch.no_grad():
    out = model(**inp)
    logits2 = out.logits

# Dernier pas de temps
last_logits = logits2[0, -1, :]
last_probs = torch.softmax(last_logits, dim=-1)

topk = 10
vals, idx = torch.topk(last_probs, k=topk)

print("\nTop-10 tokens probables après le préfixe:")
for p, tid in zip(vals.tolist(), idx.tolist()):
    print(repr(tokenizer.decode([tid])), f"{p:.3e}")