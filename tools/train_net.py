# ------------------------------------------------------------------------------
# Training code.
# Example command:
# python -m torch.distributed.launch --nproc_per_node=4 tools/train_net.py --cfg PATH_TO_CONFIG_FILE
# Written by Bowen Cheng (bcheng9@illinois.edu)
# Modified by  Maxime Istasse (maxime.istasse@uclouvain.be)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import logging
import time

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

import _init_paths
from fvcore.common.file_io import PathManager
from segmentation.config import config, update_config
from segmentation.utils.logger import setup_logger
from segmentation.model import build_segmentation_model_from_cfg
from segmentation.utils import comm
from segmentation.solver import build_optimizer, build_lr_scheduler
from segmentation.data import build_train_loader_from_cfg, build_test_loader_from_cfg
from segmentation.solver import get_lr_group_id
from segmentation.utils import save_debug_images
from segmentation.utils import AverageMeter
from segmentation.utils.utils import get_loss_info_str, to_cuda, get_module


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--timing',
                        help="Whether to log timing",
                        default=False, action="store_true")
    parser.add_argument('--gpumem',
                        help="Whether to log gpumem",
                        default=False, action="store_true")


    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger = logging.getLogger('segmentation')
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
        setup_logger(output=config.OUTPUT_DIR, distributed_rank=args.local_rank)

    logger.info(pprint.pformat(args))
    logger.info(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    distributed = len(gpus) > 1
    device = torch.device('cuda:{}'.format(args.local_rank))

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )

    # build model
    model = build_segmentation_model_from_cfg(config)
    logger.info("Model:\n{}".format(model))

    logger.info("Rank of current process: {}. World size: {}".format(comm.get_rank(), comm.get_world_size()))

    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    if comm.get_world_size() > 1:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

    data_loader = build_train_loader_from_cfg(config)
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_lr_scheduler(config, optimizer)

    data_loader_iter = iter(data_loader)

    start_iter = 0
    max_iter = config.TRAIN.MAX_ITER
    best_param_group_id = get_lr_group_id(optimizer)

    # initialize model
    if os.path.isfile(config.MODEL.WEIGHTS):
        model_weights = torch.load(config.MODEL.WEIGHTS)
        get_module(model, distributed).load_state_dict(model_weights, strict=False)
        logger.info('Pre-trained model from {}'.format(config.MODEL.WEIGHTS))
    elif not config.MODEL.BACKBONE.PRETRAINED:
        if os.path.isfile(config.MODEL.BACKBONE.WEIGHTS):
            pretrained_weights = torch.load(config.MODEL.BACKBONE.WEIGHTS)
            get_module(model, distributed).backbone.load_state_dict(pretrained_weights, strict=False)
            logger.info('Pre-trained backbone from {}'.format(config.MODEL.BACKBONE.WEIGHTS))
        else:
            logger.info('No pre-trained weights for backbone, training from scratch.')

    # load model
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(config.OUTPUT_DIR, 'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            start_iter = checkpoint['start_iter']
            get_module(model, distributed).load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info('Loaded checkpoint (starting from iter {})'.format(checkpoint['start_iter']))

    data_time = AverageMeter()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    # Debug output.
    if config.DEBUG.DEBUG:
        debug_out_dir = os.path.join(config.OUTPUT_DIR, 'debug_train')
        PathManager.mkdirs(debug_out_dir)

    # Train loop.
    try:
        for i in range(start_iter, max_iter):
            # data
            start_time = time.time()
            data = next(data_loader_iter)
            if not distributed:
                data = to_cuda(data, device)
            _data_time = time.time()
            data_time.update(_data_time - start_time)

            image = data.pop('image')
            out_dict = model(image, data)

            loss = out_dict['loss']

            torch.cuda.synchronize(device)
            _forward_time = time.time()
            if args.gpumem:
                gpumem = torch.cuda.memory_allocated(device)
                peak_usage = torch.cuda.max_memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Get lr.
            lr = optimizer.param_groups[best_param_group_id]["lr"]
            lr_scheduler.step()

            _batch_time = time.time()
            batch_time.update(_batch_time - start_time)
            loss_meter.update(loss.detach().cpu().item(), image.size(0))

            if args.timing:
                logger.info('timing - forward %f' % (_forward_time - _data_time))
                logger.info('timing - both %f' % (_batch_time - _data_time))
            if args.gpumem:
                logger.info('gpumem - %f' % gpumem)
                logger.info('gpumem - peak %f' % peak_usage)


            if i == 0 or (i + 1) % config.PRINT_FREQ == 0:
                msg = '[{0}/{1}] LR: {2:.7f}\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'.format(
                        i + 1, max_iter, lr, batch_time=batch_time, data_time=data_time)
                msg += get_loss_info_str(get_module(model, distributed).loss_meter_dict)
                logger.info(msg)
            if i == 0 or (i + 1) % config.DEBUG.DEBUG_FREQ == 0:
                # TODO: Add interface for save_debug_images
                # if comm.is_main_process() and config.DEBUG.DEBUG:
                #     save_debug_images(
                #         dataset=data_loader.dataset,
                #         batch_images=image,
                #         batch_targets=data,
                #         batch_outputs=out_dict,
                #         out_dir=debug_out_dir,
                #         iteration=i,
                #         target_keys=config.DEBUG.TARGET_KEYS,
                #         output_keys=config.DEBUG.OUTPUT_KEYS,
                #         iteration_to_remove=i - config.DEBUG.KEEP_INTERVAL
                #     )
                if i>0 and (args.gpumem or args.timing):
                    break
            if i == 0 or (i + 1) % config.CKPT_FREQ == 0:
                if comm.is_main_process():
                    torch.save({
                        'start_iter': i + 1,
                        'state_dict': get_module(model, distributed).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                    }, os.path.join(config.OUTPUT_DIR, 'checkpoint.pth.tar'))
    except Exception:
        logger.exception("Exception during training:")
        raise
    finally:
        if comm.is_main_process():
            torch.save(get_module(model, distributed).state_dict(),
                       os.path.join(config.OUTPUT_DIR, 'final_state.pth'))
        logger.info("Training finished.")


if __name__ == '__main__':
    main()
