import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        # self.temperature_l = temperature_l
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        # self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_feature_InfoNCE(self, h_i, h_j, batch_size=256):
        self.batch_size = batch_size

        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_feature_PSCL(self, z1, z2, r=3.0):  #  r=3.0
        mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(r)).float().unsqueeze(1)
        mask2 = (torch.norm(z2, p=2, dim=1) < np.sqrt(r)).float().unsqueeze(1)
        z1 = mask1 * z1 + (1 - mask1) * F.normalize(z1, dim=1) * np.sqrt(r)
        z2 = mask2 * z2 + (1 - mask2) * F.normalize(z2, dim=1) * np.sqrt(r)
        loss_part1 = -2 * torch.mean(z1 * z2) * z1.shape[1]
        square_term = torch.matmul(z1, z2.T) ** 2
        loss_part2 = torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)) * \
                     z1.shape[0] / (z1.shape[0] - 1)

        return loss_part1 + loss_part2

    def forward_feature_RINCE(self, out_1, out_2, lam=0.001, q=0.5, temperature=0.5, batch_size=256):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
            lam, q, temperature
        """
        # # gather representations in case of distributed training
        # # out_1_dist: [batch_size * world_size, dim]
        # # out_2_dist: [batch_size * world_size, dim]
        # if torch.distributed.is_available() and torch.distributed.is_initialized():
        #     out_1_dist = SyncFunction.apply(out_1)
        #     out_2_dist = SyncFunction.apply(out_2)
        # else:
        self.batch_size = batch_size

        out_1_dist = out_1
        out_2_dist = out_2


        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        similarity = torch.exp(torch.mm(out, out_dist.t()) / temperature)
        # neg_mask = self.compute_neg_mask()
        N = 2 * self.batch_size
        neg_mask = self.mask_correlated_samples(N)
        neg = torch.sum(similarity * neg_mask.to(self.device), 1)

        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # InfoNCE loss
        # loss = -(torch.mean(torch.log(pos / (pos + neg))))

        # RINCE loss
        neg = ((lam*(pos + neg))**q) / q
        pos = -(pos**q) / q
        loss = pos.mean() + neg.mean()

        return loss
