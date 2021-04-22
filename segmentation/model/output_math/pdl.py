# ------------------------------------------------------------------------------
# Panoptic-Deeplab output mathematics
# Written by  Maxime Istasse (maxime.istasse@uclouvain.be)
# ------------------------------------------------------------------------------

import abc
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import build, utils


class Utils(utils.OutputMath):

    def NHW(self):
        tensor = self.raw_batch['image']
        N,C,H,W = tensor.shape
        return N,H,W

    def _mask2d(self, tensor, mask):
        return tensor[mask.expand_as(tensor)].view(*tensor.shape[:-2],-1)

    def meshgrid(self):
        N,H,W = self.NHW()
        grid = torch.stack(torch.meshgrid(torch.arange(H, device=self.device),
                                          torch.arange(W, device=self.device)),
                            )[None]  # 1,2,H,W
        return grid.float()

    def zero(self, dtype=None):
        return torch.zeros((), dtype=dtype, device=self.device)

    def cst(self, cst, dtype=None):
        return torch.full((), cst, dtype=dtype, device=self.device)

    @staticmethod
    def _unravel_index(index, shape):
        res = []
        for size in shape[::-1]:
            res.append(index % size)
            index = index // size
        return tuple(res[::-1])

    def image(self):
        image = self.raw_batch['image']
        mean = torch.tensor(self.config.DATASET.MEAN, device=self.device)[None,:,None,None]
        std = torch.tensor(self.config.DATASET.STD, device=self.device)[None,:,None,None]
        return image*std+mean

    def _weighted_average(self, tensor, weights):
        weights = weights.expand_as(tensor)
        weighted_sum = tensor.mul(weights).sum()
        weights_sum = weights.sum()
        if weights_sum > 0:
            return weighted_sum.div(weights_sum)
        return weighted_sum.mul(0)


class PanopticVariables(Utils):
    def is_thing_class(self, c):
        return c >= self.config.DATASET.FIRST_THING_CLASS

    def panoptic_truth(self):
        return self.raw_batch['panoptic_truth']

    def thing_mask_truth(self):
        return self.is_thing_class(self.semantic_truth())

    def void_mask_truth(self):
        return self.panoptic_truth() == 0

    def crowd_mask_truth(self):
        return self.thing_mask_truth() & (self.panoptic_truth() < 1000)

    def instance_truth(self, mode='id'):
        if mode == 'id':
            N,H,W, = self.NHW()
            new_instances = torch.full((N,1,H,W), -1, device=self.device)
            base_instance = torch.where(self.thing_mask_truth() & ~self.crowd_mask_truth(), self.panoptic_truth(), new_instances)
            base_uniques = [torch.unique(base_instance[i]).tolist() for i in range(base_instance.shape[0])]
            for b in range(N):
                instances = base_instance[b]
                current = 0
                for i in base_uniques[b]:
                    if i == -1:
                        continue
                    new_instances[b,instances == i] = current
                    current += 1
            return new_instances

        raise NotImplementedError(mode)

    def unique_ins_truth(self, skip_bg):
        if skip_bg:
            return [ids[1:] if ids[0] == -1 else ids
                    for ids in self.unique_ins_truth(False)]

        instances = self.instance_truth()
        return [instances[b].unique().tolist()
                for b in range(instances.shape[0])]

    def semantic_truth(self, mode='id'):
        """Semantic under different forms:"""
        if mode == 'id':
            panoptic = self.panoptic_truth()
            return torch.where(panoptic >= 1000, panoptic // 1000, panoptic)

        raise NotImplementedError(mode)

    def loss(self):
        raise NotImplementedError

    def panoptic_pred(self):
        return torch.where(self.thing_mask_pred(), 1000*self.semantic_pred()+self.instance_pred(), self.semantic_pred())


class PanDLGroundtruth(PanopticVariables):
    def ins_mask_truth(self, b, i):
        return self.instance_truth()[b].squeeze(0) == i

    def centroids_truth(self):
        meshgrid = self.meshgrid()
        instance = self.instance_truth()
        all_centers = []
        for b in range(self.NHW()[0]):
            centers = [meshgrid[0,:,instance[b,0] == i].mean(dim=1).round()
                        for i in self.unique_ins_truth(True)[b]]
            centers = torch.stack(centers, dim=0) if centers else torch.zeros([0,2], device=self.device)
            all_centers.append(centers)
        return all_centers

    def pixelwise_centroids_truth(self):
        N,H,W = self.NHW()
        C = 2
        all_centroids_truth = torch.full((N,C,H,W), 0., device=self.device)
        for b in range(N):
            for i in self.unique_ins_truth(True)[b]:
                all_centroids_truth[b,:,self.ins_mask_truth(b,i)] = self.centroids_truth()[b][i][:,None]
        return all_centroids_truth

    def offset_truth(self):
        return self.pixelwise_centroids_truth()-self.meshgrid()

    def center_truth(self, std=8, std_margin=4):
        margin = std_margin*std
        arange = torch.arange(-margin,margin+1, dtype=torch.float32)

        N,H,W = self.NHW()
        all_maps = torch.zeros((N,1,H,W), dtype=torch.float32)
        gaussian = torch.exp(-(arange[None,:]**2 + arange[:,None]**2)
                             *0.5/std**2)
        gH, gW = gaussian.shape
        centroids = self.centroids_truth()
        for b in range(N):
            map = all_maps[b]
            for i in self.unique_ins_truth(True)[b]:
                y, x = centroids[b][i].tolist()
                # Clip x and y within the image
                #  0123
                #  a   b      => a=0 b=3
                y, x = np.clip(y, 0, H), np.clip(x, 0, W)
                # Compute bounds
                #   0123   7
                # c a  b   d   => c=-2 d=7
                y1, x1 = int(y)-margin, int(x)-margin
                y2, x2 = y1+gH, x1+gW
                # Compute offset to border
                #   0123   7
                # c a  b   d   => c:2 d:7-4
                cy1, cy2 = 0-min(y1,0), H-max(y2,H)
                cx1, cx2 = 0-min(x1,0), W-max(x2,W)
                # c=0  b=4
                y1, x1 = y1+cy1, x1+cx1
                y2, x2 = y2+cy2, x2+cx2

                map[0,y1:y2, x1:x2] = torch.max(gaussian[cy1:gH+cy2, cx1:gW+cx2],
                                                map[0,y1:y2, x1:x2])

        return all_maps.to(self.device)


class PanDLPrediction(PanopticVariables):
    def offset_pred(self, mode):
        if mode == 'activations':
            return self.raw_batch['offset_pred']
        if mode == 'relative':
            return self.offset_pred('activations')
        if mode == 'absolute':
            return self.meshgrid() + self.offset_pred('relative')
        raise NotImplementedError(mode)

    def center_pred(self, mode):
        if mode == 'activations':
            return self.raw_batch['center_pred']
        if mode == 'heatmap':
            return self.center_pred('activations')
        if mode == 'hitmap':
            heatmap = self.center_pred('heatmap')
            NMS_KERNEL = self.config.POST_PROCESSING.NMS_KERNEL
            CENTER_THRESHOLD = self.config.POST_PROCESSING.CENTER_THRESHOLD
            hitmap = (heatmap >= CENTER_THRESHOLD) & \
                (heatmap == F.max_pool2d(heatmap, NMS_KERNEL, stride=1,
                                         padding=NMS_KERNEL//2))
            return hitmap
        raise NotImplementedError(mode)

    def semantic_pred(self, mode='id'):
        if mode == 'activations':
            return self.raw_batch['semantic_pred']
        if mode == 'logits':
            return self.semantic_pred('activations')
        if mode == 'softmax':
            return self.semantic_pred('logits').softmax(dim=1)
        if mode == 'argmax':
            return self.semantic_pred('logits').argmax(dim=1, keepdim=True)
        if mode == 'id':
            all_map = self.semantic_pred('argmax').clone()
            for b in range(self.NHW()[0]):
                map = all_map[b]
                for i in self.unique_ins_pred(True)[b]:
                    map[self.ins_mask_pred(b,i)] = self.ins_class_pred(b,i)
            return all_map
        raise NotImplementedError(mode)

    def ins_mask_pred(self, b, i):
        return self.instance_pred()[b] == i

    def ins_class_pred(self, b, i):
        semantic = self.semantic_pred('argmax')
        ins_mask = self.instance_pred()[b] == i
        kls = torch.argmax(torch.bincount(semantic[b,ins_mask]))
        return kls

    def centroids_pred(self):
        all_centroids = []
        for b in range(self.NHW()[0]):
            coords = torch.nonzero(self.center_pred('hitmap')[b,0])
            all_centroids.append(coords)
        return all_centroids

    def thing_mask_pred(self):
        return self.is_thing_class(self.semantic_pred('argmax'))

    def _closest(self, all_centroids, all_votes, cpu=25):
        N,_,H,W = all_votes.shape
        all_closest = torch.full((N,1,H,W), -1, dtype=torch.int16, device=self.device)
        for b in range(len(all_centroids)):
            centroids = all_centroids[b]
            n_centroids = len(centroids)
            if n_centroids == 0:
                continue
            votes = all_votes[b]
            on_cpu = cpu is not None and n_centroids >= cpu

            _votes = votes.to('cpu') if on_cpu else votes               #   C,H,W
            _centroids = centroids.to('cpu') if on_cpu else centroids   # P,C

            _distances2 = _votes[None,:,:,:].sub(_centroids[:,:,None,None]).pow(2).sum(dim=1)
            _closest = _distances2.argmin(dim=0)

            closest = _closest.to(self.device) if on_cpu else _closest

            all_closest[b] = closest
        return all_closest

    def instance_pred(self, mode='id'):
        if mode == 'closest':
            all_votes = self.offset_pred('absolute')
            all_centroids = self.centroids_pred()
            return self._closest(all_centroids, all_votes)
        if mode == 'id':
            return torch.where(self.thing_mask_pred(), self.instance_pred('closest'), torch.full([1,1,1,1], -1, device=self.device, dtype=torch.int16))
        raise NotImplementedError(mode)

    def unique_ins_pred(self, skip_bg):
        if skip_bg:
            return [ids[1:] if ids[0] == -1 else ids
                    for ids in self.unique_ins_pred(False)]

        instances = self.instance_pred()
        return [instances[b].unique().tolist()
                for b in range(instances.shape[0])]


class PanDLLosses(PanDLPrediction, PanDLGroundtruth, PanopticVariables):

    def semantic_weights(self):
        N,H,W, = self.NHW()
        semantic_weights = torch.full((N,1,H,W), 1, device=self.device, dtype=torch.float32)
        # semantic_weights[self.void_mask_truth()] = 1
        for b in range(N):
            for i in self.unique_ins_truth(True)[b]:
                mask = self.ins_mask_truth(b,i)
                size = torch.count_nonzero(mask)
                if size < self.config.DATASET.SMALL_INSTANCE_AREA:
                    semantic_weights[b,0,mask] = self.config.DATASET.SMALL_INSTANCE_WEIGHT
        return semantic_weights

    def offset_weights(self):
        N,H,W, = self.NHW()
        offset_weights = self.thing_mask_truth() & ~self.crowd_mask_truth()
        return offset_weights.float()

    def center_weights(self):
        N,H,W, = self.NHW()
        center_weights = self.cst(1, torch.bool)[None,None,None,None] & ~self.crowd_mask_truth() & ~self.void_mask_truth()
        return center_weights.float()

    def semantic_loss(self):
        loss = self.modules.semantic_loss(
            self.semantic_pred('logits'), self.semantic_truth()[:,0],
            semantic_weights=self.semantic_weights()[:,0]
        )
        return loss

    def center_loss(self):
        loss = self.modules.center_loss(self.center_pred('heatmap'), self.center_truth())
        loss = self._weighted_average(loss, self.center_weights())
        return loss

    def offset_loss(self):
        loss = self.modules.offset_loss(self.offset_pred('relative'), self.offset_truth())
        loss = self._weighted_average(loss, self.offset_weights())
        return loss

    def loss(self):
        return (
            self.offset_loss() * self.config.LOSS.OFFSET.WEIGHT +
            self.center_loss() * self.config.LOSS.CENTER.WEIGHT +
            self.semantic_loss() * self.config.LOSS.SEMANTIC.WEIGHT
            )


class PanDLOutputsCompatibility(PanDLPrediction):
    def semantic(self):
        return self.semantic_pred(mode='activations')
    def center(self):
        return self.center_pred(mode='activations')
    def offset(self):
        return self.offset_pred(mode='activations')
    def foreground(self):
        return self.thing_mask_pred()
    def __iter__(self):
        return iter(('semantic', 'center', 'offset', 'foreground'))


class CPUComputedTensors(PanDLLosses, PanDLGroundtruth):
    @utils.override(PanDLGroundtruth.semantic_truth)
    def semantic_truth(self):
        return self.raw_batch['semantic_truth'][:,None]
    @utils.override(PanDLGroundtruth.offset_truth)
    def offset_truth(self):
        return self.raw_batch['offset_truth']
    @utils.override(PanDLGroundtruth.center_truth)
    def center_truth(self):
        return self.raw_batch['center_truth']

    @utils.override(PanDLLosses.semantic_weights)
    def semantic_weights(self):
        return self.raw_batch['semantic_weights_truth'][:,None]
    @utils.override(PanDLLosses.offset_weights)
    def offset_weights(self):
        return self.raw_batch['offset_weights_truth'][:,None]
    @utils.override(PanDLLosses.center_weights)
    def center_weights(self):
        return self.raw_batch['center_weights_truth'][:,None]


class CityscapesClasses(PanopticVariables):
    _TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    _EVAL_ID_TO_TRAIN_ID = {v: k for k, v  in enumerate(_TRAIN_ID_TO_EVAL_ID)}
    @utils.override(PanopticVariables.semantic_truth)
    def semantic_truth(self, mode='id'):
        if mode == 'id':
            base_sem = super().semantic_truth()
            new_sem = torch.zeros_like(base_sem)
            for base_id, new_id in self._EVAL_ID_TO_TRAIN_ID.items():
                new_sem[base_sem == base_id] = new_id
            return new_sem
        if mode == '_id':
            return super().semantic_truth()
        return super().semantic_truth(mode)


class CityscapesClasses(PanopticVariables):
    _TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    _EVAL_ID_TO_TRAIN_ID = {v: k for k, v  in enumerate(_TRAIN_ID_TO_EVAL_ID)}
    @utils.override(PanopticVariables.semantic_truth)
    def semantic_truth(self, mode='id'):
        if mode == 'id':
            base_sem = super().semantic_truth()
            new_sem = torch.zeros_like(base_sem)
            for base_id, new_id in self._EVAL_ID_TO_TRAIN_ID.items():
                new_sem[base_sem == base_id] = new_id
            return new_sem
        if mode == '_id':
            return super().semantic_truth()
        return super().semantic_truth(mode)


class OracleCenters:
    @utils.override(PanDLPrediction.center_pred)
    def center_pred(self, mode):
        if mode == 'heatmap':
            return self.center_truth()
        return super().center_pred(mode)


class OracleOffsets:
    @utils.override(PanDLPrediction.offset_pred)
    def offset_pred(self, mode):
        if mode == 'relative':
            return self.offset_truth()
        return super().offset_pred(mode)


class OracleSemantics:
    @utils.override(PanDLPrediction.semantic_pred)
    def semantic_pred(self, mode='id'):
        if mode == 'logits':
            return 999*F.one_hot(self.semantic_truth().squeeze(1)).permute(0,3,1,2).float()
        return super().semantic_pred(mode)


class PanopticDLBase(
        PanDLOutputsCompatibility,
        PanDLLosses,
        PanDLPrediction,
        PanDLGroundtruth,
        PanopticVariables,
    ):
    pass
