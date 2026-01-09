- **Name** : KJAOUJ Aymane
- **Commands** :
```
Python -m venv 8614
Source 8614/bin/activate
pip install -r requirement 
```

- **Question 2**: The `settings` object is a **dictionary** (`dict`). It contains the hyperparameters of the GPT-2 model. For the '124M' version, it has 5 keys: `n_vocab` (50257), `n_ctx` (1024), `n_embd` (768), `n_head` (12), and `n_layer` (12).

- **Question 3**: The `params` object is also a **dictionary** (`dict`). It contains the actual weight tensors for the model. Its top-level keys are `blocks` (a list of dictionaries for each transformer layer), `b` (bias), `g` (gain/scale), `wpe` (positional embeddings), and `wte` (token embeddings).

- **Question 4**: 
The `GPTModel.__init__` method expects a configuration dictionary `cfg` with specific keys: `vocab_size`, `emb_dim`, `context_length`, `drop_rate`, and `n_layers`. Examining the `settings` object, we see that the keys are different (`n_vocab` instead of `vocab_size`, `n_ctx` instead of `context_length`). Therefore, the `settings` variable is **not** in the right format and needs to be mapped. We also need to add keys like `n_heads` and `qkv_bias` which are used by the internal attention layers.

- **Question 5.1**: We use `df.sample(frac=1, random_state=123)` to **shuffle** the dataset randomly before splitting. This ensures that the training and test sets are representative of the overall data distribution and don't contain any ordering bias. The `random_state` ensures reproducibility.

- **Question 5.2**: The training set is highly **unbalanced**. 'ham' messages make up ~86.6% of the data, while 'spam' accounts for only ~13.4%. This imbalance can lead the model to be biased towards 'ham'. We address this using `class_weights` in the loss function.

- **Question 7**:
With a training set of 4,457 samples and a batch size of 16, we have a total of **279** batches.

- **Question 8.1 & 8.2**: The number of output classes is 2 (spam vs ham). The original head was configured for vocab size (50,257), while the new head is a linear layer with output dimension 2.

- **Question 8.3**: We freeze internal layers to leverage the pre-trained weights which already contain general language features. This reduces the number of parameters to train, making it faster and preventing overfitting on the small spam dataset.

- **Question 10**:
The trend shows the loss steadily decreasing. The accuracy improves, especially for the spam class, proving the fine-tuning is working.