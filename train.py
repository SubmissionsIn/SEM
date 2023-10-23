import torch
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
from loss import Loss
from dataloader import load_data
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Dataname = 'DHA'
# Dataname = 'CCV'
# Dataname = 'NUSWIDE'
Dataname = 'Caltech'
# Dataname = 'YoutubeVideo'

CL_Loss = ['InfoNCE', 'PSCL', 'RINCE']  # three kinds of contrastive losses
Measure_M_N = ['CMI', 'JSD', 'MMD']     # Class Mutual Information (CMI), Jensen–Shannon Divergence (JSD), Maximum Mean Discrepancy (MMD)
sample_mmd = 2000                       # select partial samples to compute MMD as it has high complexity, otherwise might be out-of-memory
Reconstruction = ['AE', 'DAE', 'MAE']   # autoencoder (AE), denoising autoencoder (DAE), masked autoencoder (MAE)
per = 0.3                               # the ratio of masked samples to perform masked AE, e.g., 30%

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)     # 256
parser.add_argument("--temperature_f", default=1.0)            # 1.0
parser.add_argument("--contrastive_loss", default=CL_Loss[0])  # 0, 1, 2
parser.add_argument("--measurement", default=Measure_M_N[0])   # 0, 1, 2
parser.add_argument("--Recon", default=Reconstruction[0])      # 0, 1, 2
parser.add_argument("--bi_level_iteration", default=4)         # 4
parser.add_argument("--times_for_K", default=1)                # 0.5 1 2 4
parser.add_argument("--Lambda", default=1)                     # 0.001 0.01 0.1 1 10 100 1000
parser.add_argument("--learning_rate", default=0.0003)         # 0.0003
parser.add_argument("--weight_decay", default=0.)              # 0.
parser.add_argument("--workers", default=8)                    # 8
parser.add_argument("--mse_epochs", default=100)               # 100
parser.add_argument("--con_epochs", default=100)               # 100
parser.add_argument("--feature_dim", default=512)              # 512
parser.add_argument("--high_feature_dim", default=128)         # 128
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('SEM + ' + args.contrastive_loss + ' + ' + args.measurement + ' + ' + args.Recon)

if args.dataset == "DHA":
    args.con_epochs = 300
    args.bi_level_iteration = 1

if args.dataset == "CCV":
    args.con_epochs = 50
    args.bi_level_iteration = 4

if args.dataset == "YoutubeVideo":
    args.con_epochs = 25
    args.bi_level_iteration = 1

if args.dataset == "NUSWIDE":
    args.con_epochs = 25
    args.bi_level_iteration = 4

if args.dataset == "Caltech":
    args.con_epochs = 100
    args.bi_level_iteration = 4
    # or
    args.bi_level_iteration = 3

Total_con_epochs = args.con_epochs * args.bi_level_iteration


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


accs = []
nmis = []
aris = []
purs = []
ACC_tmp = 0

for Runs in range(1):   # 10
    print("ROUND:{}".format(Runs+1))

    t1 = time.time()
    # setup_seed(5)   # if we find that the initialization of networks is sensitive, we can set a seed for stable performance.
    dataset, dims, view, data_size, class_num = load_data(args.dataset)

    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            # drop_last=True,
            drop_last=False,
        )


    def Low_level_rec_train(epoch, rec='AE', p=0.3, mask_ones_full=[], mask_ones_not_full=[]):
        tot_loss = 0.
        criterion = torch.nn.MSELoss()
        Vones_full = []
        Vones_not_full = []
        flag_full = 0
        flag_not_full = 0
        for batch_idx, (xs, _, _) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)

            xnum = xs[0].shape[0]

            if rec == 'AE':
                optimizer.zero_grad()
                _, _, xrs, _, _ = model(xs)
            if rec == 'DAE':
                noise_x = []
                for v in range(view):
                    # print(xs[v])
                    noise = torch.randn(xs[v].shape).to(device)
                    # print(noise)
                    noise = noise + xs[v]
                    # print(noise)
                    noise_x.append(noise)
                optimizer.zero_grad()
                _, _, xrs, _, _ = model(noise_x)
            if rec == 'MAE':
                noise_x = []
                for v in range(view):

                    if xnum == args.batch_size and flag_full == 0 and epoch == 1:
                        # print(1)
                        num = xs[v].shape[0] * xs[v].shape[1]
                        ones = torch.ones([1, num]).to(device)
                        zeros_num = int(num * p)
                        for i in range(zeros_num):
                            ones[0, i] = 0
                        Vones_full.append(ones)
                    if xnum is not args.batch_size and flag_not_full == 0 and epoch == 1:
                        # print(1)
                        num = xs[v].shape[0] * xs[v].shape[1]
                        ones = torch.ones([1, num]).to(device)
                        zeros_num = int(num * p)
                        for i in range(zeros_num):
                            ones[0, i] = 0
                        Vones_not_full.append(ones)

                    if xnum == args.batch_size and epoch == 1:
                        noise = Vones_full[v][:, torch.randperm(Vones_full[v].size(1))]
                    if xnum is not args.batch_size and epoch == 1:
                        noise = Vones_not_full[v][:, torch.randperm(Vones_not_full[v].size(1))]

                    if xnum == args.batch_size and epoch is not 1:
                        noise = mask_ones_full[v][:, torch.randperm(mask_ones_full[v].size(1))]
                    if xnum is not args.batch_size and epoch is not 1:
                        noise = mask_ones_not_full[v][:, torch.randperm(mask_ones_not_full[v].size(1))]
                    noise = torch.reshape(noise, xs[v].shape)
                    noise = noise * xs[v]
                    noise_x.append(noise)

                if xnum == args.batch_size:
                    flag_full = 1
                else:
                    flag_not_full = 1

                optimizer.zero_grad()
                _, _, xrs, _, _ = model(noise_x)

            loss_list = []
            for v in range(view):
                loss_list.append(criterion(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        # print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
        return Vones_full, Vones_not_full

    def High_level_contrastive_train(epoch, nmi_matrix, Lambda=1.0, rec='AE', p=0.3, mask_ones_full=[], mask_ones_not_full=[]):
        tot_loss = 0.
        mes = torch.nn.MSELoss()
        record_loss_con = []
        Vones_full = []
        Vones_not_full = []
        flag_full = 0
        flag_not_full = 0

        for v in range(view):
            record_loss_con.append([])
            for w in range(view):
                record_loss_con[v].append([])

        # Sim = 0
        # cos = torch.nn.CosineSimilarity(dim=0)

        for batch_idx, (xs, _, _) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)

            optimizer.zero_grad()
            zs, qs, xrs, hs, re_h = model(xs)
            loss_list = []

            xnum = xs[0].shape[0]
            #------------------------
            # P = zs[0]
            # Q = zs[1]
            # for i in range(xnum):
            #     # print(cos(P[i], Q[i]))
            #     Sim += cos(P[i], Q[i]).item()
            #-------------------------
            if rec == 'DAE':
                noise_x = []
                for v in range(view):
                    # print(xs[v])
                    noise = torch.randn(xs[v].shape).to(device)
                    # print(noise)
                    noise = noise + xs[v]
                    # print(noise)
                    noise_x.append(noise)
                optimizer.zero_grad()
                _, _, xrs, _, _ = model(noise_x)
            if rec == 'MAE':
                noise_x = []
                for v in range(view):

                    if xnum == args.batch_size and flag_full == 0 and epoch == 1:
                        # print(1)
                        num = xs[v].shape[0] * xs[v].shape[1]
                        ones = torch.ones([1, num]).to(device)
                        zeros_num = int(num * p)
                        for i in range(zeros_num):
                            ones[0, i] = 0
                        Vones_full.append(ones)
                    if xnum is not args.batch_size and flag_not_full == 0 and epoch == 1:
                        # print(1)
                        num = xs[v].shape[0] * xs[v].shape[1]
                        ones = torch.ones([1, num]).to(device)
                        zeros_num = int(num * p)
                        for i in range(zeros_num):
                            ones[0, i] = 0
                        Vones_not_full.append(ones)

                    if xnum == args.batch_size and epoch == 1:
                        noise = Vones_full[v][:, torch.randperm(Vones_full[v].size(1))]
                    if xnum is not args.batch_size and epoch == 1:
                        noise = Vones_not_full[v][:, torch.randperm(Vones_not_full[v].size(1))]

                    if xnum == args.batch_size and epoch is not 1:
                        noise = mask_ones_full[v][:, torch.randperm(mask_ones_full[v].size(1))]
                    if xnum is not args.batch_size and epoch is not 1:
                        noise = mask_ones_not_full[v][:, torch.randperm(mask_ones_not_full[v].size(1))]

                    noise = torch.reshape(noise, xs[v].shape)
                    noise = noise * xs[v]
                    noise_x.append(noise)

                if xnum == args.batch_size:
                    flag_full = 1
                else:
                    flag_not_full = 1

                optimizer.zero_grad()
                _, _, xrs, _, _ = model(noise_x)

            for v in range(view):
                # for w in range(v + 1, view):
                for w in range(view):
                    # if v == w:
                    #     continue
                    if args.contrastive_loss == 'InfoNCE':
                        tmp = criterion.forward_feature_InfoNCE(zs[v], zs[w], batch_size=xnum)
                    if args.contrastive_loss == 'PSCL':
                        tmp = criterion.forward_feature_PSCL(zs[v], zs[w])
                    if args.contrastive_loss == 'RINCE':
                        tmp = criterion.forward_feature_RINCE(zs[v], zs[w], batch_size=xnum)

                    # loss_list.append(tmp)
                    loss_list.append(tmp * nmi_matrix[v][w])
                    record_loss_con[v][w].append(tmp)

                loss_list.append(Lambda * mes(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

        # print(Sim / 1400)  # 1400 is the data size of Caltech

        for v in range(view):
            for w in range(view):
                record_loss_con[v][w] = sum(record_loss_con[v][w])
                record_loss_con[v][w] = record_loss_con[v][w].item() / len(data_loader)

        return Vones_full, Vones_not_full, record_loss_con, _

    if not os.path.exists('./models'):
        os.makedirs('./models')

    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    # print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, device).to(device)
    
    print("Initialization......")
    epoch = 0
    while epoch < args.mse_epochs:
        epoch += 1
        if epoch == 1:
            mask_ones_full, mask_ones_not_full = Low_level_rec_train(epoch,
                                                                     rec=args.Recon,
                                                                     p=per,
                                                                     )
        else:
            Low_level_rec_train(epoch,
                                rec=args.Recon,
                                p=per,
                                mask_ones_full=mask_ones_full,
                                mask_ones_not_full=mask_ones_not_full,
                                )

    acc, nmi, ari, pur, nmi_matrix_1, _ = valid(model, device, dataset, view, data_size, class_num,
                                                eval_h=True, eval_z=False, times_for_K=args.times_for_K,
                                                Measure=args.measurement, test=False, sample_num=sample_mmd)

    print("Self-Weighted Multi-view Contrastive Learning with Reconstruction Regularization...")
    Iteration = 1
    print("Iteration " + str(Iteration) + ":")
    epoch = 0
    record_loss_con = []
    record_cos = []
    while epoch < Total_con_epochs:
        epoch += 1
        if epoch == 1:
            mask_ones_full, mask_ones_not_full, record_loss_con_, record_cos_ = High_level_contrastive_train(epoch,
                                                                                             nmi_matrix_1,
                                                                                             args.Lambda,
                                                                                             rec=args.Recon,
                                                                                             p=per)
        else:
            _, _, record_loss_con_, record_cos_ = High_level_contrastive_train(epoch,
                                                                  nmi_matrix_1,
                                                                  args.Lambda,
                                                                  rec=args.Recon,
                                                                  p=per,
                                                                  mask_ones_full=mask_ones_full,
                                                                  mask_ones_not_full=mask_ones_not_full,
                                                                  )

        record_loss_con.append(record_loss_con_)
        record_cos.append(record_cos_)
        if epoch % args.con_epochs == 0:
            if epoch == args.mse_epochs + Total_con_epochs:
                break

            # print(nmi_matrix_1)

            acc, nmi, ari, pur, _, nmi_matrix_2 = valid(model, device, dataset, view, data_size, class_num,
                                                        eval_h=False, eval_z=True, times_for_K=args.times_for_K,
                                                        Measure=args.measurement, test=False, sample_num=sample_mmd)
            nmi_matrix_1 = nmi_matrix_2
            if epoch < Total_con_epochs:
                Iteration += 1
                print("Iteration " + str(Iteration) + ":")

        pg = [p for p in model.parameters() if p.requires_grad]
        #  this code matters, to re-initialize the optimizers
        optimizer = torch.optim.Adam(pg, lr=args.learning_rate, weight_decay=args.weight_decay)

    accs.append(acc)
    nmis.append(nmi)
    aris.append(ari)
    purs.append(pur)

    # if acc > ACC_tmp:
    #     ACC_tmp = acc
    #     state = model.state_dict()
    #     torch.save(state, './models/' + args.dataset + '.pth')

    t2 = time.time()
    print("Time cost: " + str(t2 - t1))
    print('End......')


print(accs, np.mean(accs)/0.01, np.std(accs)/0.01)
print(nmis, np.mean(nmis)/0.01, np.std(nmis)/0.01)
# print(aris, np.mean(aris)/0.01, np.std(aris)/0.01)
# print(purs, np.mean(purs)/0.01, np.std(purs)/0.01)


def PLOT_LOSS(record_loss_con=[]):
    import matplotlib.pyplot as plt
    from matplotlib import pyplot
    plt.style.use('seaborn-whitegrid')
    palette = pyplot.get_cmap('Set1')
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 50,
             }
    fontsize = 60
    fig = plt.figure()
    ax = fig.add_subplot()
    length = 133      # 400 / 3
    # print(len(record_loss_con))
    iters = np.linspace(0, length - 1, length, dtype=int)

    loss = []
    v1 = 0
    v2 = 3
    for i in range(length):
        loss.append(record_loss_con[3 * i + 1][v1][v2])
    ax.plot(iters, loss, color=palette(1), linestyle='-', label='InfoNCE loss (View 1; View 4)', linewidth=4.0)

    loss = []
    v1 = 3
    v2 = 4
    for i in range(length):
        loss.append(record_loss_con[3 * i + 1][v1][v2])
    ax.plot(iters, loss, color=palette(2), linestyle='-', label='InfoNCE loss (View 4; View 5)', linewidth=4.0)

    ax.legend(prop=font1, frameon=1, fancybox=0, framealpha=1)
    ax.set_xlabel('Training epochs', fontsize=fontsize)
    ax.set_ylabel('Loss values', fontsize=fontsize)
    plt.xticks([0, 33, 66, 99, 132], [0, 100, 200, 300, 400], rotation=0, fontsize=fontsize)
    plt.yticks(rotation=0, fontsize=fontsize)
    plt.show()


if Dataname == 'Caltech' and args.contrastive_loss == 'InfoNCE' and args.bi_level_iteration == 4:
    PLOT_LOSS(record_loss_con)
