# ------------------------------------------------------------------------------
# Output Mathematics builder and model wrapper
# Written by  Maxime Istasse (maxime.istasse@uclouvain.be)
# ------------------------------------------------------------------------------


import collections

import torch.nn as nn

from . import pdl, pdl_learnable_centers


def build_output_mathematics_class(config):
    blocks = []

    if config.OM.BASE == 'panoptic-deeplab':
        blocks.append(pdl.PanopticDLBase)
    else:
        raise Exception('OM base %s was not recognized' % config.OM.BASE)

    if config.DATASET.DATASET.startswith('cityscapes'):
        blocks.append(pdl.CityscapesClasses)
    else:
        raise Exception('Dataset %s was not recognized' % config.DATASET.DATASET)

    for name in config.OM.MIXINS:

        if name == 'cpu-tensors':
            blocks.append(pdl.CPUComputedTensors)

        elif name.startswith('oracle-'):
            name = {n: True for n in name.split('-')[1:]}
            if name.pop('semantics', False):
                blocks.append(pdl.OracleSemantics)
            if name.pop('centers', False):
                blocks.append(pdl.OracleCenters)
            if name.pop('offsets', False):
                blocks.append(pdl.OracleOffsets)
            if name:
                raise NotImplementedError(list(name))

        elif name == 'learnable-centers':
            blocks.append(pdl_learnable_centers.LearnableCenters)

        else:
            raise NotImplementedError(name)

    _config = config
    class OutputMathematics(*blocks[::-1]):
        config = _config
    return OutputMathematics


def wrap_meta_architecture(meta_architecture, om_class):
    class _WrappedMetaArchitecture(MetaArchitectureWrapper, meta_architecture):
        _om_class = om_class
    return _WrappedMetaArchitecture


class MetaArchitectureWrapper(nn.Module):
    """A wrapper around a PyTorch model in order to instantate OutputMathematics from batches.
    """
    _om_class = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, targets=None):
        outputs = super().forward(x, targets=None)  # don't handle loss down there

        raw_batch = {key+'_pred': value for key, value in outputs.items()}
        if targets:
            for key, value in targets.items():
                raw_batch[key+'_truth'] = value
        raw_batch['image'] = x

        session = self._om_class(raw_batch, module=self, device=x.device)
        class modules:
            semantic_loss = self.semantic_loss
            offset_loss = self.offset_loss
            center_loss = self.center_loss
        session.modules = modules

        if targets is not None:
            return self.loss(session)
        return session

    def loss(self, session):
        assert isinstance(session, self._om_class)
        batch_size = session.NHW()[0]
        self.loss_meter_dict['Semantic loss'].update(session.semantic_loss().detach().cpu().item(), batch_size)
        if getattr(session, "center_loss", None) is not None:
            self.loss_meter_dict['Center loss'].update(session.center_loss().detach().cpu().item(), batch_size)
        if getattr(session, "offset_loss", None) is not None:
            self.loss_meter_dict['Offset loss'].update(session.offset_loss().detach().cpu().item(), batch_size)

        # In distributed DataParallel, this is the loss on one machine, need to average the loss again
        # in train loop.
        session['loss'] = session.loss()
        self.loss_meter_dict['Loss'].update(session.loss().detach().cpu().item(), batch_size)
        return session