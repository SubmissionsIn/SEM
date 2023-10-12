from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score, normalized_mutual_info_score
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaluate(label, pred):
    # v_measure = v_measure_score(label, pred)
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur


def inference(loader, model, device, view, data_size):
    """
    Xs
    Zs
    Hs
    """
    model.eval()
    pred_vectors = []
    Xs = []
    Zs = []
    Hs = []
    Qs = []
    for v in range(view):
        pred_vectors.append([])
        Xs.append([])
        Zs.append([])
        Hs.append([])
        Qs.append([])
    labels_vector = []

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            zs, _, _, hs, _ = model.forward(xs)
        for v in range(view):
            zs[v] = zs[v].detach()
            hs[v] = hs[v].detach()
            Xs[v].extend(xs[v].cpu().detach().numpy())
            Zs[v].extend(zs[v].cpu().detach().numpy())
            Hs[v].extend(hs[v].cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    labels_vector = np.array(labels_vector).reshape(data_size)
    for v in range(view):
        Xs[v] = np.array(Xs[v])
        Zs[v] = np.array(Zs[v])
        Hs[v] = np.array(Hs[v])
        Qs[v] = np.array(Qs[v])
        pred_vectors[v] = np.array(pred_vectors[v])
    return Xs, [], Zs, labels_vector, Hs


def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output, dim=1)
        q_output = F.softmax(q_output, dim=1)
    log_mean_output = ((p_output + q_output)/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2


def guassian_kernel_mmd(source, target, kernel_mul=2, kernel_num=4, fix_sigma=None):
    """Gram kernel matrix
    source: sample_size_1 * feature_size
    target: sample_size_2 * feature_size
    kernel_mul: bandwith of kernels
    kernel_num: number of kernels
    return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)
            [ K_ss K_st
              K_ts K_tt ]
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val)


def MMD(source, target, kernel_mul=2, kernel_num=4, fix_sigma=None):
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel_mmd(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, m:]
    YX = kernels[m:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)   # K_ss，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)   # K_tt,Target<->Target

    loss = XX.sum() + XY.sum() + YX.sum() + YY.sum()
    return loss


def valid(model, device, dataset, view, data_size, class_num, eval_h=True, eval_z=True,
          times_for_K=1.0, Measure='CMI', test=True, sample_num=1000):
    test_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    X_vectors, pred_vectors, z_vectors, labels_vector, h_vectors = inference(test_loader, model, device, view, data_size)
    final_z_features = []
    h_clusters = []
    z_clusters = []
    nmi_matrix_h = np.zeros((view, view))
    nmi_matrix_z = np.zeros((view, view))

    if eval_h and Measure == 'CMI':
        print("Clustering results on each view (H^v):")
        acc_avg, nmi_avg, ari_avg, pur_avg = 0, 0, 0, 0
        for v in range(view):
            kmeans = KMeans(n_clusters=int(class_num * times_for_K), n_init=100)
            if len(labels_vector) > 10000:
                kmeans = MiniBatchKMeans(n_clusters=int(class_num * times_for_K), batch_size=5000, n_init=100)
            y_pred = kmeans.fit_predict(h_vectors[v])
            h_clusters.append(y_pred)
            nmi, ari, acc, pur = evaluate(labels_vector, y_pred)
            print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v + 1, acc,
                                                                                     v + 1, nmi,
                                                                                     v + 1, ari,
                                                                                     v + 1, pur))
            acc_avg += acc
            nmi_avg += nmi
            ari_avg += ari
            pur_avg += pur

        print('Mean = {:.4f} Mean = {:.4f} Mean = {:.4f} Mean={:.4f}'.format(acc_avg / view,
                                                                             nmi_avg / view,
                                                                             ari_avg / view,
                                                                             pur_avg / view))
        kmeans = KMeans(n_clusters=class_num, n_init=100)
        if len(labels_vector) > 10000:
            kmeans = MiniBatchKMeans(n_clusters=int(class_num), batch_size=5000, n_init=100)
        z = np.concatenate(h_vectors, axis=1)
        pseudo_label = kmeans.fit_predict(z)
        print("Clustering results on all views ([H^1...H^V]): " + str(labels_vector.shape[0]))
        nmi, ari, acc, pur = evaluate(labels_vector, pseudo_label)
        print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
        for i in range(view):
            for j in range(view):
                if Measure == 'CMI':
                    cnmi, _, _, _ = evaluate(h_clusters[i], h_clusters[j])
                    nmi_matrix_h[i][j] = np.exp(cnmi) - 1
        print(nmi_matrix_h)

    if eval_h and Measure is not 'CMI':
        for i in range(view):
            for j in range(view):
                if Measure == 'JSD':
                    P = torch.tensor(h_vectors[i])
                    Q = torch.tensor(h_vectors[j])
                    divergence = js_div(P, Q).item()
                    nmi_matrix_h[i][j] = np.exp(1 - divergence) - 1
                if Measure == 'MMD':
                    if len(labels_vector) > sample_num:
                        P = torch.tensor(h_vectors[i][0: sample_num])
                        Q = torch.tensor(h_vectors[j][0: sample_num])
                    else:
                        P = torch.tensor(h_vectors[i])
                        Q = torch.tensor(h_vectors[j])
                    mmd = MMD(P, Q, kernel_mul=4, kernel_num=4)
                    nmi_matrix_h[i][j] = np.exp(-mmd)
        print(nmi_matrix_h)

    if eval_z and Measure == 'CMI':
        print("Clustering results on each view (Z^v):")
        acc_avg, nmi_avg, ari_avg, pur_avg = 0, 0, 0, 0
        for v in range(view):
            kmeans = KMeans(n_clusters=int(class_num * times_for_K), n_init=100)
            if len(labels_vector) > 10000:
                kmeans = MiniBatchKMeans(n_clusters=int(class_num * times_for_K), batch_size=5000, n_init=100)
            y_pred = kmeans.fit_predict(z_vectors[v])
            final_z_features.append(z_vectors[v])
            z_clusters.append(y_pred)
            nmi, ari, acc, pur = evaluate(labels_vector, y_pred)
            print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} PUR{}={:.4f}'.format(v + 1, acc,
                                                                                     v + 1, nmi,
                                                                                     v + 1, ari,
                                                                                     v + 1, pur))
            acc_avg += acc
            nmi_avg += nmi
            ari_avg += ari
            pur_avg += pur

        print('Mean = {:.4f} Mean = {:.4f} Mean = {:.4f} Mean={:.4f}'.format(acc_avg/view,
                                                                             nmi_avg/view,
                                                                             ari_avg/view,
                                                                             pur_avg/view))
        kmeans = KMeans(n_clusters=class_num, n_init=100)
        if len(labels_vector) > 10000:
            kmeans = MiniBatchKMeans(n_clusters=int(class_num), batch_size=5000, n_init=100)
        h = np.concatenate(final_z_features, axis=1)
        pseudo_label = kmeans.fit_predict(h)
        print("Clustering results on all views ([Z^1...Z^V]): " + str(labels_vector.shape[0]))
        nmi, ari, acc, pur = evaluate(labels_vector, pseudo_label)
        print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
        for i in range(view):
            for j in range(view):
                if Measure == 'CMI':
                    cnmi, _, _, _ = evaluate(z_clusters[i], z_clusters[j])
                    nmi_matrix_z[i][j] = np.exp(cnmi) - 1
        print(nmi_matrix_z)

    if eval_z and Measure is not 'CMI':
        for i in range(view):
            for j in range(view):
                if Measure == 'JSD':
                    P = torch.tensor(z_vectors[i])
                    Q = torch.tensor(z_vectors[j])
                    divergence = js_div(P, Q).item()
                    nmi_matrix_z[i][j] = np.exp(1 - divergence) - 1
                if Measure == 'MMD':
                    if len(labels_vector) > sample_num:
                        P = torch.tensor(z_vectors[i][0: sample_num])
                        Q = torch.tensor(z_vectors[j][0: sample_num])
                    else:
                        P = torch.tensor(z_vectors[i])
                        Q = torch.tensor(z_vectors[j])
                    mmd = MMD(P, Q, kernel_mul=4, kernel_num=4)
                    nmi_matrix_z[i][j] = np.exp(-mmd)

        print(nmi_matrix_z)

    if test or Measure is not 'CMI':
        kmeans = KMeans(n_clusters=class_num, n_init=100)
        if len(labels_vector) > 10000:
            kmeans = MiniBatchKMeans(n_clusters=int(class_num), batch_size=5000, n_init=100)
        h = np.concatenate(z_vectors, axis=1)
        pseudo_label = kmeans.fit_predict(h)
        print("Clustering results on all views ([Z^1...Z^V]): " + str(labels_vector.shape[0]))
        nmi, ari, acc, pur = evaluate(labels_vector, pseudo_label)
        print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))

    return acc, nmi, ari, pur, nmi_matrix_h, nmi_matrix_z
