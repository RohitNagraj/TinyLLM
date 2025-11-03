import sys
sys.path.append("../")
from llm.model import TinyLLM
from llm.tokenizer import TinyTokenizer
import torch
from config import *
from utils import read_data, train_test_split_data, get_data_batch

texts = read_data(dataset_path)
tokenizer = TinyTokenizer(train=True)

data = torch.tensor(tokenizer.encode(texts))
train_data, test_data = train_test_split_data(data, split_ratio=0.9)

model = TinyLLM(gpt_config.vocab_size, gpt_config.embed_dim, gpt_config.num_heads, gpt_config.num_layers, gpt_config.block_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
model.train()
for iter in range(num_epochs):
    xb, yb = get_data_batch(train_data, block_size, batch_size)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), path)
