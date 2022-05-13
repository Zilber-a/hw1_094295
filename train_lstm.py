import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import pickle
import time


class RnnDataset(Dataset):
    def __init__(self, features, target):
        self.features_list = features
        self.target = target

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        return self.features_list[idx], self.target[idx]


class LSTM_Model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=50, target_size=1, num_layers=4):
        super(LSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, target_size))

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        lstm_out = lstm_out[:, -1]
        out = self.linear(lstm_out)
        return out


# transformation of data before each dataloader iteration
def collate_batch(batch):
    features_list = []
    labels_list = []
    features_list.append(torch.ones([100, batch[0][0].shape[1]]))
    for (features, label) in batch:
        features_list.append(features)
        labels_list.append(label)

    labels_list = torch.tensor(labels_list, dtype=torch.int64).unsqueeze(1).float()
    # a patient is represented with its last 100 rows (and padded if needed)
    features_list = torch.flip(pad_sequence(features_list, batch_first=True, padding_value=0), [1])[1:, -100:, :]

    return features_list.float(), labels_list


# function which trains the model and outputs a learning curve
def train(train_dataloader, test_dataloader, num_epochs, model, lr, weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
    train_loss = []
    train_f1 = []
    test_loss = []
    test_f1 = []
    best_test_f1 = 0.0
    for epoch in range(num_epochs):
        # train by iterating over the training set
        model.train()
        total_loss = 0
        preds = []
        targets = []
        start = time.time()
        for batch_idx, (data_tensor, target_tensor) in enumerate(train_dataloader):
            data_tensor = data_tensor.to(device)
            target_tensor = target_tensor.to(device)
            out = model(data_tensor)
            loss = bce_loss(out, target_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds += [1 if x >= 0 else 0 for x in out]
            targets += target_tensor.squeeze(1).tolist()
        train_f1.append(f1_score(targets, preds))
        train_loss.append(total_loss / batch_idx)

        # test the current model on test set
        model.eval()
        total_loss = 0
        preds = []
        targets = []
        with torch.no_grad():
            for batch_idx, (data_tensor, target_tensor) in enumerate(test_dataloader):
                data_tensor = data_tensor.to(device)
                target_tensor = target_tensor.to(device)
                out = model(data_tensor)
                loss = bce_loss(out, target_tensor)
                total_loss += loss.item()

                preds += [1 if x >= 0 else 0 for x in out]
                targets += target_tensor.squeeze(1).tolist()
        test_f1.append(f1_score(targets, preds))
        test_loss.append(total_loss / batch_idx)
        print("epoch no.", epoch, "train loss: ", train_loss[-1], " test loss: ", total_loss / batch_idx,
              "train f1: ", train_f1[-1], "test f1: ", test_f1[-1], " time: ", time.time() - start)
        # if test F1 score is improved, save model
        if (best_test_f1 < test_f1[-1]):
            best_test_f1 = test_f1[-1]
            fig = plt.figure()
            plt.plot(range(len(train_loss)), train_loss, color='green', label='train loss')
            plt.plot(range(len(train_loss)), test_loss, color='red', label='test loss')
            fig.suptitle('Loss', fontsize=20)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(loc='best')
            plt.savefig("loss_plot.png")
            fig.clf()
            plt.close()

            fig = plt.figure()
            plt.plot(range(len(train_loss)), train_f1, color='green', label='train f1')
            plt.plot(range(len(train_loss)), test_f1, color='red', label='test f1')
            fig.suptitle('F1 score', fontsize=20)
            plt.xlabel("Epoch")
            plt.ylabel("F1")
            plt.legend(loc='best')
            plt.savefig("f1_plot.png")
            fig.clf()
            plt.close()

            torch.save(model.state_dict(), "lstm_model.pt")


def weights_tensor(train_targets):
    # ones = torch.count_nonzero(train_targets).item()
    # zeros = len(train_targets) - ones
    return torch.tensor([1])


if __name__ == '__main__':
    with open('tensor_features_train.pickle', 'rb') as handle:
        train_features = pickle.load(handle)
    with open('tensor_targets_train.pickle', 'rb') as handle:
        train_targets = pickle.load(handle)
    with open('tensor_features_test.pickle', 'rb') as handle:
        test_features = pickle.load(handle)
    with open('tensor_targets_test.pickle', 'rb') as handle:
        test_targets = pickle.load(handle)

    batch_size = 200
    num_epochs = 60
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    dataset_train = RnnDataset(train_features, train_targets)
    dataset_test = RnnDataset(test_features, test_targets)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    model = LSTM_Model(dataset_train[0][0].shape[1]).to(device).float()
    weights = weights_tensor(train_targets).to(device)

    train(dataloader_train, dataloader_test, num_epochs, model, 0.0005, weights)
