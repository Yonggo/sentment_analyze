import json

import torch.nn as nn
import os

class SentimentNet(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.0):
        super(SentimentNet, self).__init__()
        #self.save_parameters(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).squeeze().zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).squeeze().zero_().to(device))
        return hidden

    # gleich wie Aufruf init_hidden(batch_size=1, device), weil da innerlich die squeeze() aufgerufen wird,
    # sodass Array mit Dimention=1 aufgelöst wird
    def init_hidden_for_predic(self, device):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, self.hidden_dim).zero_().to(device))
        return hidden

    def save_parameters(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers):
        params = {"vocab_size": vocab_size, "output_size": output_size, "embedding_dim": embedding_dim,
                  "hidden_dim": hidden_dim, "n_layers": n_layers}
        print("Started writing parameters to a file")
        with open("params.json", "w") as fp:
            json.dump(params, fp)  # encode dict into JSON
        print("Done writing parameters into params.json")

