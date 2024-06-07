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
        super(KANPolynomialKernel, self).__init__()
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
                 rnn_num_layers=1, bidirectional=False):
        super(KANWithRecurrentLayer, self).__init__()
        self.rnn_type = rnn_type
        self.rnn_num_layers = rnn_num_layers
        self.bidirectional = bidirectional

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size3, output_size)
        self.fc4 = nn.Linear(hidden_size3, output_size)

        rnn_hidden_size = hidden_size3  # Set rnn_hidden_size to hidden_size3

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_size2, rnn_hidden_size, rnn_num_layers, bidirectional=bidirectional,
                               batch_first=True)
        else:
            self.rnn = nn.GRU(hidden_size2, rnn_hidden_size, rnn_num_layers, bidirectional=bidirectional,
                              batch_first=True)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x, _ = self.rnn(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class KANGaussianSigmoid(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, sigma=1.0):
        super(KANGaussianSigmoid, self).__init__()
        self.sigma = sigma
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def gaussian_kernel(self, x):
        return torch.exp(-torch.pow(x, 2) / (2 * self.sigma ** 2))

    def sigmoid_kernel(self, x):
        return torch.sigmoid(x)

    def forward(self, x):
        x = self.gaussian_kernel(self.fc1(x))
        x = self.sigmoid_kernel(self.fc2(x))
        x = self.gaussian_kernel(self.fc3(x))
        x = self.fc4(x)
        return x

    import torch


class KANGaussianPolynomialCosine(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size, sigma=1.0, degree=2):
        super(KANGaussianPolynomialCosine, self).__init__()
        self.sigma = sigma
        self.degree = degree
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def gaussian_kernel(self, x):
        return torch.exp(-torch.pow(x, 2) / (2 * self.sigma ** 2))

    def polynomial_kernel(self, x):
        return torch.pow(1 + x, self.degree)

    def cosine_kernel(self, x, y):
        print(f"Shape of x: {x.shape}")
        print(f"Shape of y: {y.shape}")
        x = x.unsqueeze(0)  # Add a dimension for batch_size
        y = y.unsqueeze(0)  # Add a dimension for batch_size
        x_norm = x.norm(dim=2, keepdim=True)  # Calculate norms
        y_norm = y.norm(dim=2, keepdim=True)
        similarity = torch.bmm(x, y.transpose(1, 2)) / (x_norm * y_norm.transpose(1, 2))
        similarity = similarity.squeeze(0)  # Remove the batch_size dimension
        return similarity
    def forward(self, x):
        x = self.gaussian_kernel(self.fc1(x))
        x = self.polynomial_kernel(self.fc2(x))
        x = self.cosine_kernel(self.fc3(x), x)
        x = self.fc4(x)
        return x



class KANReluExp(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(KANReluExp, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def relu_kernel(self, x):
        return torch.relu(x)

    def exponential_kernel(self, x):
        return torch.exp(x)

    def forward(self, x):
        x = self.relu_kernel(self.fc1(x))
        x = self.exponential_kernel(self.fc2(x))
        x = self.relu_kernel(self.fc3(x))
        x = self.fc4(x)
        return x


class KANReluTanh(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(KANReluTanh, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def relu_kernel(self, x):
        return torch.relu(x)

    def tanh_kernel(self, x):
        return torch.tanh(x)

    def forward(self, x):
        x = self.relu_kernel(self.fc1(x))
        x = self.tanh_kernel(self.fc2(x))
        x = self.relu_kernel(self.fc3(x))
        x = self.fc4(x)
        return x