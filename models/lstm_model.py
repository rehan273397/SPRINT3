
import torch
import torch.nn as nn


class PhishGuardLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        # Convert token IDs into dense word vectors
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        # Learn patterns and context across word sequences
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Classify the message
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):

        embedded = self.embedding(input_ids)

        _, (hidden, _) = self.lstm(embedded)

        final_hidden = hidden[-1]

        output = self.dropout(final_hidden)
        output = self.classifier(output)

        return output.squeeze(1)

