from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
phrase = "Artificial intelligence is metamorphosing the world!"

tokens = tokenizer.tokenize(phrase)
print("Tokens:")
print(tokens)

#
# 2.b 
#

token_ids = tokenizer.encode(phrase)
print("Token IDs:", token_ids)

print("\nDétails par token:")
for tid in token_ids:
    txt = tokenizer.decode([tid])
    print(tid, repr(txt))

#
# 2.d
#

phrase2 = "GPT models use BPE tokenization to process unusual words like antidisestablishmentarianism."

tokens2 = tokenizer.tokenize(phrase2)
print("Tokens phrase 2:")
print(tokens2)

long_word_tokens = tokenizer.tokenize("antidisestablishmentarianism")

print("\nSous-tokens du mot 'antidisestablishmentarianism':")
print(long_word_tokens)
print("Nombre de sous-tokens :", len(long_word_tokens))