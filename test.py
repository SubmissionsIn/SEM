import torch
from network import Network
from metric import valid
import argparse
from dataloader import load_data
from sklearn.metrics import accuracy_score, f1_score, recall_score

test_train = 0.7
# Dataname = 'DHA'
# Dataname = 'CCV'
# Dataname = 'NUSWIDE'
Dataname = 'Caltech'
# Dataname = 'YoutubeVideo'

CL_Loss = ['InfoNCE', 'PSCL', 'RINCE']
Measure_M_N = ['CMI', 'JSD', 'MMD']
Reconstruction = ['AE', 'DAE', 'MAE']
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=1.0)
parser.add_argument("--contrastive_loss", default=CL_Loss[0])
parser.add_argument("--measurement", default=Measure_M_N[0])
parser.add_argument("--Recon", default=Reconstruction[0])
parser.add_argument("--bi_level_iteration", default=4)
parser.add_argument("--times_for_K", default=1)
parser.add_argument("--Lambda", default=1)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=100)
parser.add_argument("--con_epochs", default=100)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import numpy as np
dataset, dims, view, data_size, class_num = load_data(args.dataset)
model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
model = model.to(device)
checkpoint = torch.load('./models/' + args.dataset + '.pth')
model.load_state_dict(checkpoint)
print("Dataset:{}".format(args.dataset))
print("Datasize:" + str(data_size))
print("Loading model...")

valid(model, device, dataset, view, data_size, class_num, eval_h=False, eval_z=False, test=True)

if Dataname == 'YoutubeVideo':
    exit(0)

from torch.utils.data import DataLoader
from metric import inference
from sklearn import svm
from sklearn import model_selection

np.random.seed(80)


def SVM_Classification(x, y, seed=1, test_r=0.3):
    # data_train, data_test, tag_train, tag_test = model_selection.train_test_split(x, y, random_state=seed, test_size=test_r)
    data_train, data_test, tag_train, tag_test = model_selection.train_test_split(x, y, test_size=test_r)

    def classifier():
        clf = svm.SVC(C=1,
                      kernel='linear',
                      decision_function_shape='ovr')
        return clf

    clf = classifier()

    def train(clf, x_train, y_train):
        clf.fit(x_train, y_train.ravel())

    train(clf, data_train, tag_train)

    def print_accuracy(clf, x_train, y_train, x_test, y_test):
        y_pre = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pre)
        # f1 = f1_score(y_test, y_pre, average='macro')
        recall = recall_score(y_test, y_pre, average='macro')
        return acc, recall
    acc, recall = print_accuracy(clf, data_train, tag_train, data_test, tag_test)

    return acc, recall


test_loader = DataLoader(dataset, batch_size=256, shuffle=False)
X_vectors, pred_vectors, high_level_vectors, labels_vector, low_level_vectors = inference(test_loader, model, device, view, data_size)
ACC_H = []
ACC_ALLH = []
REC_ALLH = []
ALLH = np.concatenate(high_level_vectors, axis=1)
for seed in range(10):
    y = labels_vector
    acc, recall = SVM_Classification(ALLH, y, seed=seed, test_r=test_train)
    ACC_ALLH.append(acc/0.01)
    REC_ALLH.append(recall/0.01)

print(ACC_ALLH, np.mean(ACC_ALLH), np.std(ACC_ALLH))
print(REC_ALLH, np.mean(REC_ALLH), np.std(REC_ALLH))
