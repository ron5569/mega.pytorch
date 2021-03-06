# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from mega_core.data import make_data_loader
from mega_core.utils.comm import get_world_size, synchronize
from mega_core.utils.metric_logger import MetricLogger
from mega_core.engine.inference import inference

from apex import amp

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    writer
):
    logger = logging.getLogger("mega_core.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_names = cfg.DATASETS.TEST

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images, targets = convert_to_cuda(cfg, device, images, targets)

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:

            writer.add_scalar("total loss",losses, iteration)
            for k,v in loss_dict.items():
                writer.add_scalar(k,v, iteration)

            write_log(eta_string, iteration, logger, meters, optimizer)
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if data_loader_val is not None and test_period > 0 and iteration % test_period == 0:
            meters_val = MetricLogger(delimiter="  ")
            synchronize()
            data_loaders=make_data_loader(cfg, is_train=False, is_distributed=(get_world_size() > 1), is_for_period=False)
            dataset_list = cfg.DATASETS.TEST

            for data_loader, dataset_name in zip(data_loaders, dataset_list):
                _ = inference(cfg,  # The result can be used for additional logging, e. g. for TensorBoard
                    model,
                    # The method changes the segmentation mask format in a data loader,
                    # so every time a new data loader is created:
                    data_loader,
                    dataset_name="[Validation]",
                    iou_types=iou_types,
                    box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                    device=cfg.MODEL.DEVICE,
                    expected_results=cfg.TEST.EXPECTED_RESULTS,
                    expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                    output_folder=None,
                )
                logger.info(f"write test set, {dataset_name}_map {_['map']}, iteration: {iteration}")
                writer.add_scalar(f"{dataset_name}_map", _["map"], iteration)

            model.train()
            with torch.no_grad():
                # Should be one image for each GPU:
                for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                    images_val["cur"] = images_val["cur"].to(device)
                    for key in ("ref", "ref_l", "ref_m", "ref_g"):
                        if key in images_val.keys():
                            images_val[key] = [img.to(device) for img in images_val[key]]


                    targets_val = [target.to(device) for target in targets_val]
                    loss_dict = model(images_val, targets_val)
                    losses = sum(loss for loss in loss_dict.values())
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    meters_val.update(loss=losses_reduced, **loss_dict_reduced)

                    writer.add_scalar("Validation total loss", losses, iteration)
                    for k, v in loss_dict.items():
                        writer.add_scalar("Validation_" + k, v, iteration)


            write_log(eta_string, iteration, logger, meters_val, optimizer, "Validation")
            synchronize()
            #synchronize()

        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


def write_log(eta_string, iteration, logger, meters, optimizer, train_test_str="training"):
    logger.info(
        meters.delimiter.join(
            [
                "[{train_test}]: ",
                "eta: {eta}",
                "iter: {iter}",
                "{meters}",
                "lr: {lr:.6f}",
                "max mem: {memory:.0f}",
            ]
        ).format(
            train_test = train_test_str,
            eta=eta_string,
            iter=iteration,
            meters=str(meters),
            lr=optimizer.param_groups[0]["lr"],
            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
        )
    )


def convert_to_cuda(cfg, device, images, targets):
    if not cfg.MODEL.VID.ENABLE:
        images = images.to(device)
    else:
        method = cfg.MODEL.VID.METHOD
        if method in ("base",):
            images = images.to(device)
        elif method in ("rdn", "mega", "fgfa", "dff"):
            images["cur"] = images["cur"].to(device)
            for key in ("ref", "ref_l", "ref_m", "ref_g"):
                if key in images.keys():
                    images[key] = [img.to(device) for img in images[key]]
        else:
            raise ValueError("method {} not supported yet.".format(method))
    targets = [target.to(device) for target in targets]
    return images, targets
