import torch
import torch.nn as nn


class KAN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(KAN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class KANGaussianKernel:
    pass


class KANWithBatchNorm(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, dropout_p=0.5):
        super(KANWithBatchNorm, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = torch.sin(self.bn1(self.fc1(x)))
        x = torch.sin(self.bn2(self.fc2(x)))
        x = torch.sin(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


    class KANPolynomialKernel(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, degree=2):
            self.degree = degree
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.fc3 = nn.Linear(hidden_size2, hidden_size3)
            self.fc4 = nn.Linear(hidden_size3, output_size)

        def polynomial_kernel(self, x):
            return torch.pow(1 + x, self.degree)

        def forward(self, x):
            x = self.polynomial_kernel(self.fc1(x))
            x = self.polynomial_kernel(self.fc2(x))
            x = self.polynomial_kernel(self.fc3(x))
            x = self.fc4(x)
            return x
    class KANGaussianKernel(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, sigma=1.0):
            super(KANGaussianKernel, self).__init__()
            self.sigma = sigma
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.fc3 = nn.Linear(hidden_size2, hidden_size3)
            self.fc4 = nn.Linear(hidden_size3, output_size)

        def gaussian_kernel(self, x):
            return torch.exp(-torch.pow(x, 2) / (2 * self.sigma ** 2))

        def forward(self, x):
            x = self.gaussian_kernel(self.fc1(x))
            x = self.gaussian_kernel(self.fc2(x))
            x = self.gaussian_kernel(self.fc3(x))
            x = self.fc4(x)
            return x

class KANWithRecurrentLayer(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, rnn_type='LSTM',
                 rnn_hidden_size=128, rnn_num_layers=1, bidirectional=False):
        super(KANWithRecurrentLayer, self).__init__()
        self.rnn_type = rnn_type
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.bidirectional = bidirectional

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_size2, rnn_hidden_size, rnn_num_layers, bidirectional=bidirectional, batch_first=True)
        else:
            self.rnn = nn.GRU(hidden_size2, rnn_hidden_size, rnn_num_layers, bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x, _ = self.rnn(x.unsqueeze(1))  # Unsqueeze to add sequence dimension
        x = x.squeeze(1)  # Squeeze to remove sequence dimension
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x