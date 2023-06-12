import torch
import torch.nn as nn


class LSTM_forecast(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int) -> None:
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self) -> torch.Tensor:
        return (torch.zeros(self.num_layers, 1, self.hidden_size),
                torch.zeros(self.num_layers, 1, self.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, self.hidden = self.lstm(x.view(len(x), 1, -1), self.hidden)
        predictions = self.fc(lstm_out.view(len(x), -1))
        return predictions[-1]