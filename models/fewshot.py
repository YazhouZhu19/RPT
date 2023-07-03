import numpy as np
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .encoder import Res101Encoder
from .attention import MultiHeadAttention
from .attention import MultiLayerPerceptron


class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()
        self.criterion_MSE = nn.MSELoss()
        self.alpha = torch.Tensor([1., 0.])
        self.fg_sampler = np.random.RandomState(1289)
        self.fg_num = 10  # number of foreground partitions
        self.MHA = MultiHeadAttention(n_head=3, d_model=512, d_k=512, d_v=512)
        self.MLP = MultiLayerPerceptron(dim=512, mlp_dim=1024)

    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False, t_loss_scaler=1, n_iters=20):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """

        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        self.iter = 3
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W


        # Extract features #
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts, tao = self.encoder(imgs_concat)
        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]

        # Get threshold #
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  # t for query features
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        # Compute loss #
        outputs = []
        for epi in range(supp_bs):
            # Partition the foreground object into N parts, the coarse support prototypes
            fg_partition_prototypes = [[[self.compute_multiple_prototypes(
                self.fg_num, supp_fts[n][[epi], way, shot], supp_mask[[epi], way, shot], self.fg_sampler)
                         for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in range(len(supp_fts))]

            # calculate coarse query prototype
            supp_fts_ = [[[self.getFeatures(supp_fts[n][[epi], way, shot], supp_mask[[epi], way, shot])
                           for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in
                         range(len(supp_fts))]
            fg_prototypes = [self.getPrototype(supp_fts_[n]) for n in range(len(supp_fts))]  # the coarse foreground
            qry_pred = [torch.stack(
                [self.getPred(qry_fts[n][epi], fg_prototypes[n][way], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'
            qry_prototype_coarse = [self.getFeatures(qry_fts[n][epi], qry_pred[n][epi]) for n in range(len(qry_fts))]


            # # The first BATE block
            for i in range(self.iter):
                fg_partition_prototypes = [[[self.BATE(fg_partition_prototypes[n][way][shot][epi], qry_prototype_coarse[n])
                       for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in range(len(supp_fts))]  
                supp_proto = [[[torch.mean(fg_partition_prototypes[n][way][shot], dim=1) for shot in range(self.n_shots)]
                          for way in range(self.n_ways)] for n in range(len(supp_fts))]  
                # CQPC module
                qry_pred_coarse = [torch.stack(
                    [self.getPred(qry_fts[n][epi], supp_proto[n][way][epi], self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]
                qry_prototype_coarse = [self.getFeatures(qry_fts[n][epi], qry_pred_coarse[n][epi])
                                    for n in range(len(qry_fts))]

            # Get query predictions #
            qry_pred = [torch.stack(
                [self.getPred(qry_fts[n][epi], supp_proto[n][way][epi], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]  # N x Wa x H' x W'

            # Combine predictions of different feature maps #
            qry_pred_up = [F.interpolate(qry_pred[n], size=img_size, mode='bilinear', align_corners=True)
                           for n in range(len(qry_fts))]
            pred = [self.alpha[n] * qry_pred_up[n] for n in range(len(qry_fts))]
            preds = torch.sum(torch.stack(pred, dim=0), dim=0) / torch.sum(self.alpha)

            preds = torch.cat((1.0 - preds, preds), dim=1)


            outputs.append(preds)

        output = torch.stack(outputs, dim=1)  
        output = output.view(-1, *output.shape[2:])

        return output


    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getFeatures_fg(self, fts, mask):
        """
        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        fts_ = fts.squeeze(0).permute(1, 2, 0)

        fts_ = fts_.view(fts_.size()[0] * fts_.size()[1], fts_.size()[2])
        mask_ = F.interpolate(mask.unsqueeze(0), size=fts.shape[-2:], mode='bilinear')
        mask_ = mask_.view(-1)

        l = math.ceil(mask_.sum())
        c = torch.argsort(mask_, descending=True, dim=0)
        fg = c[:l]

        fts_fg = fts_[fg]

        return fts_fg

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts

        return fg_prototypes

    

    def compute_multiple_prototypes(self, fg_num, sup_fts, sup_fg, sampler):
        """

        Parameters
        ----------
        fg_num: int
            Foreground partition numbers
        sup_fts: torch.Tensor
             [B, C, h, w], float32
        sup_fg: torch. Tensor
             [B, h, w], float32 (0,1)
        sampler: np.random.RandomState

        Returns
        -------
        fg_proto: torch.Tensor
            [B, k, C], where k is the number of foreground proxies

        """

        B, C, h, w = sup_fts.shape  # B=1, C=512
        fg_mask = F.interpolate(sup_fg.unsqueeze(0), size=sup_fts.shape[-2:], mode='bilinear')
        fg_mask = fg_mask.squeeze(0).bool()   # [B, h, w] --> bool
        batch_fg_protos = []

        for b in range(B):
            fg_protos = []

            fg_mask_i = fg_mask[b]  # [h, w]

            # Check if zero
            with torch.no_grad():
                if fg_mask_i.sum() < fg_num:
                    fg_mask_i = fg_mask[b].clone()  # don't change original mask
                    fg_mask_i.view(-1)[:fg_num] = True

            # Iteratively select farthest points as centers of foreground local regions
            all_centers = []
            first = True
            pts = torch.stack(torch.where(fg_mask_i), dim=1)
            for _ in range(fg_num):
                if first:
                    i = sampler.choice(pts.shape[0])
                    first = False
                else:
                    dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
                    # choose the farthest point
                    i = torch.argmax((dist ** 2).sum(-1).min(1)[0])
                pt = pts[i]  # center y, x
                all_centers.append(pt)

            # Assign fg labels for fg pixels
            dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
            fg_labels = torch.argmin((dist ** 2).sum(-1), dim=1)

            # Compute fg prototypes
            fg_feats = sup_fts[b].permute(1, 2, 0)[fg_mask_i]  # [N, C]
            for i in range(fg_num):
                proto = fg_feats[fg_labels == i].mean(0)  # [C]
                fg_protos.append(proto)

            fg_protos = torch.stack(fg_protos, dim=1)  # [C, k]
            batch_fg_protos.append(fg_protos)
        fg_proto = torch.stack(batch_fg_protos, dim=0).transpose(1, 2)  # [B, k, C]

        return fg_proto

    def BATE(self, fg_prototypes, qry_prototype_coarse):

        # S&W module
        A = torch.mm(fg_prototypes, qry_prototype_coarse.t())
        kc = ((A.min() + A.mean())/2).floor()
        if A is not None:
            S = torch.zeros(A.size(), dtype=torch.float).cuda()
            S[A < kc] = -10000.0
        A = torch.softmax((A+S), dim=0)
        A = torch.mm(A, qry_prototype_coarse)
        # rest Transformer operation
        T = self.MHA(A.unsqueeze(0), A.unsqueeze(0), A.unsqueeze(0))
        T = self.MLP(T)

        return T

