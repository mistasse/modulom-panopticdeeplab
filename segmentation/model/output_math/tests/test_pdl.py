# ------------------------------------------------------------------------------
# Tests for Panoptic-Deeplab output mathematics and their learnable centers
# variant
# Written by  Maxime Istasse (maxime.istasse@uclouvain.be)
# ------------------------------------------------------------------------------


from .. import pdl

import torch
import torch.nn as nn
import torch.nn.functional as F


from ..pdl import PanDLGroundtruth, PanDLLosses, PanDLPrediction, PanopticVariables
from ..pdl_learnable_centers import LearnableCenters

from ....config import config as default_config


def test_gt():
    class OutMath(PanDLGroundtruth, PanopticVariables):
        config = default_config.clone()
        config.DATASET.FIRST_THING_CLASS = 2

    panoptic = torch.tensor([
        [   0,    1, 1, 3001],
        [   0,    0, 0, 3001],
        [3002, 3002, 0, 3001],
        [3002, 3002, 0, 3001],
    ])
    batch = OutMath(dict(
        panoptic_truth=panoptic[None,None],
        image=torch.zeros((1, 3, *panoptic.shape)),
        ))
    assert torch.equal(batch.instance_truth(),
        torch.tensor([[[[-1, -1, -1,  0],
                        [-1, -1, -1,  0],
                        [ 1,  1, -1,  0],
                        [ 1,  1, -1,  0]]]]).long())

    assert batch.meshgrid().equal(
        torch.tensor([[[[0., 0., 0., 0.],
                        [1., 1., 1., 1.],
                        [2., 2., 2., 2.],
                        [3., 3., 3., 3.]],
                       [[0., 1., 2., 3.],
                        [0., 1., 2., 3.],
                        [0., 1., 2., 3.],
                        [0., 1., 2., 3.]]]]))

    assert batch.unique_ins_truth(False) == [[-1, 0, 1]]
    assert batch.unique_ins_truth(True) == [[0, 1]]

    assert batch.centroids_truth()[0].allclose(
        torch.tensor([[2., 3.],
                      [2., 0.]]))

    assert batch.ins_mask_truth(0,0)[None].equal(
        torch.tensor([[[False, False, False,  True],
                       [False, False, False,  True],
                       [False, False, False,  True],
                       [False, False, False,  True]]]).bool())
    assert batch.pixelwise_centroids_truth().allclose(
        torch.tensor([[[[    0.,     0.,     0.,     2.],
                        [    0.,     0.,     0.,     2.],
                        [    2.,     2.,     0.,     2.],
                        [    2.,     2.,     0.,     2.]],
                       [[    0.,     0.,     0.,     3.],
                        [    0.,     0.,     0.,     3.],
                        [    0.,     0.,     0.,     3.],
                        [    0.,     0.,     0.,     3.]]]]), equal_nan=True)

    assert batch.offset_truth().allclose(
        torch.tensor([[[[ 0.,  0.,  0.,  2.],
                        [-1., -1., -1.,  1.],
                        [ 0.,  0., -2.,  0.],
                        [-1., -1., -3., -1.]],

                        [[ 0., -1., -2.,  0.],
                        [ 0., -1., -2.,  0.],
                        [ 0., -1., -2.,  0.],
                        [ 0., -1., -2.,  0.]]]]), equal_nan=True)

    assert (batch.center_truth() != 0).all()
    assert (batch.center_truth() > 0.999).sum() == 2

    assert (batch.center_truth(std=1, std_margin=2) != 0).all()
    assert (batch.center_truth(std=1, std_margin=1) == 0).any()
    assert (batch.center_truth(std=1, std_margin=1) > 0.999).sum() == 2


def test_pred():
    class OutMath(PanDLPrediction, PanopticVariables):
        config = default_config.clone()
        config.DATASET.FIRST_THING_CLASS = 2
        config.POST_PROCESSING.NMS_KERNEL = 3
    nan = 0
    batch = OutMath({
        "offset_pred": torch.tensor([[[[    nan,     nan,     nan,  1.5000],
                                       [    nan,     nan,     nan,  0.5000],
                                       [ 0.5000,  0.5000,     nan, -0.5000],
                                       [-0.5000, -0.5000,     nan, -1.5000]],
                                      [[    nan,     nan,     nan,  0.0000],
                                       [    nan,     nan,     nan,  0.0000],
                                       [ 0.5000, -0.5000,     nan,  0.0000],
                                       [ 0.5000, -0.5000,     nan,  0.0000]]]]),
        "semantic_pred": F.one_hot(torch.tensor([[ 0, 1,  1, 2],
                                                 [ 0, 0,  0, 2],
                                                 [ 2, 2,  0, 2],
                                                 [ 2, 2,  0, 2],
                                                ])).permute(2,0,1)[None],
        "center_pred": torch.tensor([[[[    nan,     nan,     nan,  0],
                                       [    nan,     nan,     nan,  1.],
                                       [      1,     .99,     nan,  0],
                                       [    .99,     .99,     nan,  0]]]]),
    })
    H,W = batch.raw_batch['offset_pred'].shape[2:]
    batch.raw_batch['image'] = torch.zeros(1,3,H,W)

    assert batch.center_pred('hitmap').equal(
        torch.tensor([[[[False, False, False, False],
                        [False, False, False,  True],
                        [ True, False, False, False],
                        [False, False, False, False]]]]))
    # FIXME: Is this deterministic? Would be better to make
    # it invariant to instance ID permutation
    assert batch.panoptic_pred().equal(
        torch.tensor([[[[   0,    1,    1, 2000],
                        [   0,    0,    0, 2000],
                        [2001, 2001,    0, 2000],
                        [2001, 2001,    0, 2000]]]]))
    assert batch.offset_pred('absolute').allclose(
        torch.tensor([[[[0.0000, 0.0000, 0.0000, 1.5000],
                        [1.0000, 1.0000, 1.0000, 1.5000],
                        [2.5000, 2.5000, 2.0000, 1.5000],
                        [2.5000, 2.5000, 3.0000, 1.5000]],
                       [[0.0000, 1.0000, 2.0000, 3.0000],
                        [0.0000, 1.0000, 2.0000, 3.0000],
                        [0.5000, 0.5000, 2.0000, 3.0000],
                        [0.5000, 0.5000, 2.0000, 3.0000]]]]))


def test_losses():
    from ...loss import DeepLabCE, L1Loss, MSELoss
    class OutMath(PanDLLosses, PanopticVariables):
        config = default_config.clone()
        config.DATASET.FIRST_THING_CLASS = 2
        config.POST_PROCESSING.NMS_KERNEL = 3

        config.DATASET.SMALL_INSTANCE_AREA = 2
        config.DATASET.SMALL_INSTANCE_WEIGHT = 3

        class modules:
            loss_weights = dict(semantic=1, offset=1, center=1)
            semantic_loss = DeepLabCE()
            offset_loss = L1Loss(reduction='none')
            center_loss = MSELoss(reduction='none')

    panoptic = torch.tensor([
        [   0,    1, 1, 3001],
        [   0,    0, 0, 3001],
        [3002, 3002, 0, 3001],
        [3002, 3002, 0, 3001],
    ])
    batch = OutMath(dict(
        panoptic_truth=panoptic[None,None],
        image=torch.zeros((1, 3, *panoptic.shape))
        ))

    batch.raw_batch['offset_pred'] = batch.offset_truth()
    batch.raw_batch['center_pred'] = batch.center_truth()
    batch.raw_batch['semantic_pred'] = 999*F.one_hot(batch.semantic_truth().squeeze(1)).permute(0,3,1,2).float()

    assert batch.loss() == 0
    assert batch._cache


def test_no_instance():
    from ...loss import DeepLabCE, L1Loss, MSELoss
    class OutMath(PanDLLosses, PanopticVariables):
        config = default_config.clone()
        config.DATASET.FIRST_THING_CLASS = 4
        config.POST_PROCESSING.NMS_KERNEL = 3

        config.DATASET.SMALL_INSTANCE_AREA = 2
        config.DATASET.SMALL_INSTANCE_WEIGHT = 3

        class modules:
            loss_weights = dict(semantic=1, offset=1, center=1)
            semantic_loss = DeepLabCE()
            offset_loss = L1Loss(reduction='none')
            center_loss = MSELoss(reduction='none')

    panoptic = torch.tensor([
        [   0,    1, 1, 3],
        [   0,    0, 0, 3],
        [   3,    3, 0, 3],
        [   3,    3, 0, 3],
    ])
    batch = OutMath(dict(
        panoptic_truth=panoptic[None,None],
        image=torch.zeros((1, 3, *panoptic.shape))
        ))

    batch.raw_batch['offset_pred'] = batch.offset_truth()
    batch.raw_batch['center_pred'] = batch.center_truth()
    batch.raw_batch['semantic_pred'] = 999*F.one_hot(batch.semantic_truth().squeeze(1)).permute(0,3,1,2).float()

    assert batch.panoptic_pred().equal(torch.tensor(
        [[[[0, 1, 1, 3],
           [0, 0, 0, 3],
           [3, 3, 0, 3],
           [3, 3, 0, 3]]]])
    )

    assert batch.loss() == 0


def test_only_instances():
    from ...loss import DeepLabCE, L1Loss, MSELoss
    class OutMath(PanDLLosses, PanopticVariables):
        config = default_config.clone()
        config.DATASET.FIRST_THING_CLASS = 1
        config.POST_PROCESSING.NMS_KERNEL = 3

        config.DATASET.SMALL_INSTANCE_AREA = 2
        config.DATASET.SMALL_INSTANCE_WEIGHT = 3

        class modules:
            loss_weights = dict(semantic=1, offset=1, center=1)
            semantic_loss = DeepLabCE()
            offset_loss = L1Loss(reduction='none')
            center_loss = MSELoss(reduction='none')

    panoptic = torch.tensor([
        [   1001, 1002, 1002, 2003],
        [   1001, 1001, 1001, 2003],
        [   1004, 1004, 1001, 2003],
        [   1004, 1004, 1001, 2003],
    ])
    batch = OutMath(dict(
        panoptic_truth=panoptic[None,None],
        image=torch.zeros((1, 3, *panoptic.shape))
        ))

    batch.raw_batch['offset_pred'] = batch.offset_truth()
    batch.raw_batch['center_pred'] = batch.center_truth()
    batch.raw_batch['semantic_pred'] = 999*F.one_hot(batch.semantic_truth().squeeze(1)).permute(0,3,1,2).float()
    print(batch.panoptic_pred())
    assert batch.panoptic_pred().equal(torch.tensor([[
         [[1001, 1000, 1000, 2003],
          [1001, 1001, 1001, 2003],
          [1002, 1002, 1001, 2003],
          [1002, 1002, 1001, 2003]]]])
    )

    assert batch.loss() == 0


def test_with_grads():
    from ...loss import DeepLabCE, L1Loss, MSELoss
    class OutMath(PanDLLosses, PanopticVariables):
        config = default_config.clone()
        config.DATASET.FIRST_THING_CLASS = 1
        config.POST_PROCESSING.NMS_KERNEL = 3

        config.DATASET.SMALL_INSTANCE_AREA = 2
        config.DATASET.SMALL_INSTANCE_WEIGHT = 3

        class modules:
            loss_weights = dict(semantic=1, offset=1, center=1)
            semantic_loss = DeepLabCE()
            offset_loss = L1Loss(reduction='none')
            center_loss = MSELoss(reduction='none')

    panoptic = torch.tensor([
        [   1001, 1002, 1002, 2003],
        [   1001, 1001, 1001, 2003],
        [   1004, 1004, 1001, 2003],
        [   1004, 1004, 1001, 2003],
    ])
    batch = OutMath(dict(
        panoptic_truth=panoptic[None,None],
        image=torch.zeros((1, 3, *panoptic.shape))
        ))

    batch.raw_batch['offset_pred'] = batch.offset_truth()
    batch.raw_batch['center_pred'] = batch.center_truth()
    batch.raw_batch['semantic_pred'] = 999*F.one_hot(batch.semantic_truth().squeeze(1)).permute(0,3,1,2).float()

    batch.raw_batch['offset_pred'] = nn.Parameter(batch.raw_batch['offset_pred'] + 0.1*torch.randn_like(batch.raw_batch['offset_pred']))
    batch.raw_batch['center_pred'] = nn.Parameter(batch.raw_batch['center_pred'] + 0.1*torch.randn_like(batch.raw_batch['center_pred']))
    batch.raw_batch['semantic_pred'] = nn.Parameter(batch.raw_batch['semantic_pred'] + 0.1*torch.randn_like(batch.raw_batch['semantic_pred']))


    assert batch.loss() != 0
    batch.loss().backward()
    assert not batch.raw_batch['center_pred'].grad.allclose(batch.zero()[None,None,None,None])


def test_learnable_centers_and_batch():
    from ...loss import DeepLabCE, L1Loss, MSELoss
    class OutMath(LearnableCenters, PanDLLosses, PanopticVariables):
        config = default_config.clone()
        config.DATASET.FIRST_THING_CLASS = 3
        config.POST_PROCESSING.NMS_KERNEL = 3

        config.DATASET.SMALL_INSTANCE_AREA = 2
        config.DATASET.SMALL_INSTANCE_WEIGHT = 3

        class modules:
            loss_weights = dict(semantic=1, offset=1, center=1)
            semantic_loss = DeepLabCE()
            offset_loss = L1Loss(reduction='none')
            center_loss = MSELoss(reduction='none')

    panoptic = torch.tensor([
        [   0,    0,    1, 1, 3001],
        [   0,    0,    0, 0, 3001],
        [3002, 3002, 3002, 0, 3001],
        [3002, 3002, 3002, 0, 3001],
        [3002, 3002, 3002, 0, 3001],
    ])
    offset_pred=torch.full((2,2,5,5), -1)
    batch = OutMath(dict(
        panoptic_truth=torch.stack([panoptic, torch.flip(panoptic, (0,1))], dim=0)[:,None],
        offset_pred=offset_pred,
        image=torch.zeros((2, 3, *panoptic.shape))
        ))
    assert batch.centroids_truth()[0].allclose(
        torch.tensor([[1., 3.],
                      [2., 0.]]))

    batch.raw_batch['center_pred'] = batch.center_truth()
    batch.raw_batch['semantic_pred'] = 999*F.one_hot(batch.semantic_truth().squeeze(1)).permute(0,3,1,2).float()

    batch.raw_batch['center_pred'] = nn.Parameter(batch.raw_batch['center_pred'])
    batch.raw_batch['semantic_pred'] = nn.Parameter(batch.raw_batch['semantic_pred'])

    print(batch.panoptic_pred())
    assert batch.panoptic_pred().allclose(torch.tensor(
       [[[[   0,    0,    1,    1, 3000],
          [   0,    0,    0,    0, 3000],
          [3001, 3001, 3001,    0, 3000],
          [3001, 3001, 3001,    0, 3000],
          [3001, 3001, 3001,    0, 3000]]],
        [[[3001,    0, 3000, 3000, 3000],
          [3001,    0, 3000, 3000, 3000],
          [3001,    0, 3001, 3000, 3000],
          [3001,    0,    0,    0,    0],
          [3001,    1,    1,    0,    0]]]]))

    assert batch.semantic_loss() == 0
    assert batch.center_loss() == 0
    assert batch.offset_loss() != 0

    batch.loss().backward()


def test_learnable_centers_no_instances():
    from ...loss import DeepLabCE, L1Loss, MSELoss
    class OutMath(pdl.OracleCenters, pdl.OracleSemantics,
                  LearnableCenters, PanDLLosses, PanopticVariables):
        config = default_config.clone()
        config.DATASET.FIRST_THING_CLASS = 3
        config.POST_PROCESSING.NMS_KERNEL = 3

        config.DATASET.SMALL_INSTANCE_AREA = 2
        config.DATASET.SMALL_INSTANCE_WEIGHT = 3

        class modules:
            loss_weights = dict(semantic=1, offset=1, center=1)
            semantic_loss = DeepLabCE()
            offset_loss = L1Loss(reduction='none')
            center_loss = MSELoss(reduction='none')

    panoptic = torch.tensor([
        [   0,    0,    1, 1, 1],
        [   0,    0,    0, 0, 1],
        [2, 2, 2, 0, 1],
        [2, 2, 2, 0, 1],
        [2, 2, 2, 0, 1],
    ])
    offset_pred=torch.full((2,2,5,5), -1)
    batch = OutMath(dict(
        panoptic_truth=torch.stack([panoptic, torch.flip(panoptic, (0,1))], dim=0)[:,None],
        offset_pred=offset_pred,
        image=torch.zeros((2, 3, *panoptic.shape))
        ))
    assert all(l.numel() == 0 for l in batch.centroids_truth())

    # batch.raw_batch['center_pred'] = batch.center_truth()
    # batch.raw_batch['semantic_pred'] = 999*F.one_hot(batch.semantic_truth().squeeze(1)).permute(0,3,1,2).float()

    assert batch.semantic_loss() == 0
    assert batch.center_loss() == 0
    assert batch.offset_loss() == 0
