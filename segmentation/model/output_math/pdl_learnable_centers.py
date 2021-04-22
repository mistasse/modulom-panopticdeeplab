# ------------------------------------------------------------------------------
# Learnable centers Panoptic-Deeplab variant
# Written by  Maxime Istasse (maxime.istasse@uclouvain.be)
# ------------------------------------------------------------------------------

from . import pdl, utils

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableCenters(pdl.PanopticDLBase):
    @utils.override(pdl.PanopticDLBase.centroids_truth)
    def centroids_truth(self, *, detach=False):
        votes = self.offset_pred('absolute')
        if detach:
            votes = votes.detach()
        instance = self.instance_truth()
        all_centers = []
        for b in range(self.NHW()[0]):
            centers = [votes[0,:,instance[b,0] == i].mean(dim=1).round()
                        for i in self.unique_ins_truth(True)[b]]
            centers = torch.stack(centers, dim=0) if centers else torch.zeros([0,2])
            all_centers.append(centers)
        return all_centers

    @utils.override(pdl.PanopticDLBase.offset_loss)
    def offset_loss(self):
        N,H,W = self.NHW()
        closest_centroid = torch.full((N,2,H,W), 0, dtype=torch.float32, device=self.device)
        instance_id = self._closest(self.centroids_truth(), self.offset_pred('absolute')).to(torch.int64)
        for b in range(N):
            centroid_coords = self.centroids_truth()[b]
            if not centroid_coords.numel():
                continue
            index = instance_id[b].view(-1)
            closest_centroid[b] = torch.index_select(centroid_coords, dim=0, index=index).view(H,W,2).permute(2,0,1)

        true_centroid = self.pixelwise_centroids_truth()


        loss = self.modules.offset_loss(self.offset_pred('absolute'), true_centroid) + (
            self.modules.offset_loss(self.offset_pred('absolute'), true_centroid)
          - self.modules.offset_loss(self.offset_pred('absolute'), closest_centroid)
        )
        loss = self._weighted_average(loss, self.offset_weights())
        return loss
