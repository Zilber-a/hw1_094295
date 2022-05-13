import sys
from preprocess_lstm import pre_process_data_lstm
from train_lstm import LSTM_Model, RnnDataset, collate_batch
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import pandas as pd

if __name__ == '__main__':
    path = sys.argv[1]
    tensors_list, target, ids = pre_process_data_lstm(path, train=False, save=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_Model(tensors_list[0].shape[1]).to(device).float()
    model.load_state_dict(torch.load("lstm_model.pt"))  # load trained model
    dataset = RnnDataset(tensors_list, target)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
    model.eval()
    preds = []
    print("predicting data")
    with torch.no_grad():
        for i, (data_tensor, target_tensor) in enumerate(dataloader):
            id = ids[i]
            data_tensor = data_tensor.to(device)
            out = model(data_tensor).item()
            # patient is predicted as positive if out >= 0
            if out >= 0:
                pred = 1
            else:
                pred = 0
            preds.append((id, pred))
    print("F1 score: ", f1_score(target, [x[1] for x in preds]))
    print("Precision score: ", precision_score(target, [x[1] for x in preds]))
    print("Recall score: ", recall_score(target, [x[1] for x in preds]))
    print("Accuracy score: ", accuracy_score(target, [x[1] for x in preds]))
    df = pd.DataFrame(preds, columns=['Id', 'SepsisLabel'])
    df.to_csv("prediction.csv", index=False)  # save results as csv

