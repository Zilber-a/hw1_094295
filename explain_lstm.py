import lime
import lime.lime_tabular
from train_lstm import collate_batch, LSTM_Model
import pickle
import numpy as np
import pandas as pd
import torch

NUM_F = 21
model = LSTM_Model(NUM_F).float()
model.load_state_dict(torch.load("lstm_model.pt"))

def predict_proba(array):
    probs = torch.sigmoid(model(torch.tensor(array).float())).detach().numpy()
    probs = np.concatenate((1-probs, probs), axis=1)
    return probs


if __name__ == '__main__':
    print("reading data")
    with open('tensor_features_train.pickle', 'rb') as handle:
        train_features = pickle.load(handle)
    with open('tensor_targets_train.pickle', 'rb') as handle:
        train_targets = pickle.load(handle)
    with open('tensor_features_test.pickle', 'rb') as handle:
        test_features = pickle.load(handle)
    with open('tensor_targets_test.pickle', 'rb') as handle:
        test_targets = pickle.load(handle)

    columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BUN', 'Calcium', 'Creatinine', 'Glucose',
               'Magnesium', 'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets', 'Age', 'Gender', 'Unit1', 'HospAdmTime']
    print("preparing data")
    train_features, train_targets = collate_batch([(train_features[i], train_targets[i]) for i in range(len(train_targets))])
    test_features, test_targets = collate_batch([(test_features[i], test_targets[i]) for i in range(len(test_targets))])

    # explain some examples using Lime
    print("creating an explainer")
    explainer = lime.lime_tabular.RecurrentTabularExplainer(train_features.numpy(), training_labels=train_targets.numpy(),
                                                            feature_names=columns, class_names=[0, 1])
    print("explaining examples")
    exp = explainer.explain_instance(test_features[5].numpy(), num_samples=7000, classifier_fn=predict_proba)
    exp.save_to_file('lstm01.html')
    exp = explainer.explain_instance(test_features[6].numpy(), num_samples=7000, classifier_fn=predict_proba)
    exp.save_to_file('lstm11.html')

