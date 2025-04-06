# Copyright (c) Megvii Inc. All rights reserved.

from __future__ import annotations

import ast
import random
from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from yolox.data.datasets import Dataset


@dataclass
class YoloxConfig:
    name: str

    # ---------------- model config ---------------- #
    # detect classes number of model
    num_classes: int = 80
    # factor of model depth
    depth: float = 1.00
    # factor of model width
    width: float = 1.00
    # depthwise model
    depthwise: bool = False
    # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
    act: Literal["silu", "relu", "lrelu"] = "silu"

    seed: Optional[Any] = None
    output_dir: str = "./out"
    print_interval: int = 100

    # ---------------- dataloader config ---------------- #
    # deterministic data loading
    deterministic: bool = False
    # set worker to 4 for shorter dataloader init time
    # If your training process cost many memory, reduce this value.
    data_num_workers: int = 4
    input_size: tuple[int, int] = (640, 640)  # (height, width)
    # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
    # To disable multiscale training, set the value to 0.
    multiscale_range: int = 5
    # you can use this to specify a multiscale range.
    random_size: Optional[tuple[int, int]] = None
    # dir of dataset images, if data_dir is None, this project will use `datasets` dir
    data_dir: Optional[str] = None
    # name of annotation file for training
    train_ann: str = "instances_train2017.json"
    # name of annotation file for evaluation
    val_ann: str = "instances_val2017.json"
    # name of annotation file for testing
    test_ann: str = "instances_test2017.json"

    # --------------- transform config ----------------- #
    # prob of applying mosaic aug
    mosaic_prob: float = 1.0
    # prob of applying mixup aug
    mixup_prob: float = 1.0
    # prob of applying hsv aug
    hsv_prob: float = 1.0
    # prob of applying flip aug
    flip_prob: float = 0.5
    # rotation angle range, for example, if set to 2, the true range is (-2, 2)
    degrees: float = 10.0
    # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
    translate: float = 0.1
    mosaic_scale: tuple[float, float] = (0.1, 2)
    # apply mixup aug or not
    enable_mixup: bool = True
    mixup_scale: tuple[float, float] = (0.5, 1.5)
    # shear angle range, for example, if set to 2, the true range is (-2, 2)
    shear: float = 2.0

    # --------------  training config --------------------- #
    # epoch number used for warmup
    warmup_epochs: int = 0 #  5
    # max training epoch
    max_epoch: int = 300
    # minimum learning rate during warmup
    warmup_lr: int = 0
    min_lr_ratio: float = 0.05
    # learning rate for one image. During training, lr will multiply batchsize.
    basic_lr_per_img: float = 0.01 / 64.0
    basic_lr_per_img: float = 0.001 / 64.0
    # name of LRScheduler
    scheduler: str = "yoloxwarmcos"
    # last #epoch to close augmention like mosaic
    no_aug_epochs: int = 15
    # apply EMA during training
    ema: bool = True

    # weight decay of optimizer
    weight_decay: float = 5e-4
    # momentum of optimizer
    momentum: float = 0.9
    # log period in iter, for example,
    # if set to 1, user could see log every iteration.
    print_interval: int = 10
    # eval period in epoch, for example,
    # if set to 1, model will be evaluate after every epoch.
    eval_interval: int = 1
    # save history checkpoint or not.
    # If set to False, yolox will only save latest and best ckpt.
    save_history_ckpt: bool = True

    # -----------------  testing config ------------------ #
    # output image size during evaluation/test
    test_size: tuple[int, int] = (640, 640)
    # confidence threshold during evaluation/test,
    # boxes whose scores are less than test_conf will be filtered
    test_conf: float = 0.01
    # nms threshold
    nmsthre: float = 0.65

    # keypoint model#
    pose = False
    dataset: Optional[Dataset] = None

    @classmethod
    def get_named_config(cls, name: str) -> Optional[YoloxConfig]:
        return _NAMED_CONFIG.get(name.replace('-', '_'))

    def validate(self):
        h, w = self.input_size
        assert h % 32 == 0 and w % 32 == 0, "input size must be multiples of 32"

    def update(self, opts: dict[str, str]):
        for k, v in opts.items():
            # only update value with same key
            if hasattr(self, k):
                src_value = getattr(self, k)
                src_type = type(src_value)

                # pre-process input if source type is list or tuple
                if isinstance(src_value, (list, tuple)):
                    v = v.strip("[]()")
                    v = [t.strip() for t in v.split(",")]

                    # find type of tuple
                    if len(src_value) > 0:
                        src_item_type = type(src_value[0])
                        v = [src_item_type(t) for t in v]

                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        v = ast.literal_eval(v)

                if k == 'seed':
                    # Special handling for seed, which has a default of None
                    v = int(v)
                setattr(self, k, v)
            else:
                raise AttributeError(f'Unknown model configuration option: {k}')

    def get_model(self):
        from yolox.models import YoloPafpn, Yolox, YoloxHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YoloPafpn(self.depth, self.width, in_channels=in_channels, depthwise=self.depthwise, act=self.act)
            head = YoloxHead(self.num_classes, self.width, in_channels=in_channels, depthwise=self.depthwise, act=self.act)
            self.model = Yolox(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get dataset according to cache and cache_type parameters.
        Args:
            cache (bool): Whether to cache imgs to ram or disk.
            cache_type (str, optional): Defaults to "ram".
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
        """
        from yolox.data import CocoDataset, TrainTransform

        return CocoDataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        """
        Get dataloader according to cache_img parameter.
        Args:
            no_aug (bool, optional): Whether to turn off mosaic data enhancement. Defaults to False.
            cache_img (str, optional): cache_img is equivalent to cache_type. Defaults to None.
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
                None: Do not use cache, in this case cache_data is also None.
        """
        from yolox.data import (
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            TrainTransform,
            YoloBatchSampler,
            worker_init_reset_seed
        )
        from yolox.utils import wait_for_the_master

        # if cache is True, we will create self.dataset before launch
        # else we will create self.dataset after launch
        if self.dataset is None:
            with wait_for_the_master():
                assert cache_img is None, \
                    "cache_img must be None if you didn't create self.dataset before launch"
                self.dataset = self.get_dataset(cache=False, cache_type=cache_img)

        self.dataset = MosaicDetection(
            dataset=self.dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        worker_init_fn = None if self.deterministic else worker_init_reset_seed

        train_loader = DataLoader(
            self.dataset,
            num_workers=self.data_num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
            worker_init_fn=worker_init_fn
        )

        return train_loader

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if self.random_size is None:
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1] = targets[..., 1] * scale_x
            targets[..., 3] = targets[..., 3] * scale_x
            targets[..., 2] = targets[..., 2] * scale_y
            targets[..., 4] = targets[..., 4] * scale_y
            if targets.shape[2] > 5:
                targets[..., 5::3] = targets[..., 5::3] * scale_x
                targets[..., 6::3] = targets[..., 6::3] * scale_y
        return inputs, targets

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            #optimizer = torch.optim.SGD(
            #    pg0, lr=lr, momentum=self.momentum, nesterov=True
            #)
            optimizer = torch.optim.AdamW(
                pg0, lr=lr)
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_dataset(self, **kwargs):
        from yolox.data import CocoDataset, ValTransform
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)

        return CocoDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="val2017" if not testdev else "test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_eval_loader(self, batch_size, is_distributed, **kwargs):
        valdataset = self.get_eval_dataset(**kwargs)

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import CocoEvaluator

        return CocoEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )

    def get_trainer(self, args):
        from yolox.core import Trainer
        trainer = Trainer(self, args)
        # NOTE: trainer shouldn't be an attribute of exp object
        return trainer

    def eval(self, model, evaluator, is_distributed, half=False, return_outputs=False):
        return evaluator.evaluate(model, is_distributed, half, return_outputs=return_outputs)


def validate_config(config: YoloxConfig):
    h, w = config.input_size
    assert h % 32 == 0 and w % 32 == 0, "input size must be multiples of 32"


class YoloxS(YoloxConfig):
    def __init__(self):
        super().__init__("yolox_s")
        self.depth = 0.33
        self.width = 0.50


class YoloxM(YoloxConfig):
    def __init__(self):
        super().__init__("yolox_m")
        self.depth = 0.67
        self.width = 0.75


class YoloxL(YoloxConfig):
    def __init__(self):
        super().__init__("yolox_l")
        self.depth = 1.0
        self.width = 1.0


class YoloxX(YoloxConfig):
    def __init__(self):
        super().__init__("yolox_x")
        self.depth = 1.33
        self.width = 1.25


class YoloxTiny(YoloxConfig):
    def __init__(self):
        super().__init__("yolox_tiny")
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.test_size = (416, 416)
        self.enable_mixup = False


class YoloxNano(YoloxConfig):
    def __init__(self):
        super().__init__("yolox_nano")
        self.depth = 0.33
        self.width = 0.25
        self.depthwise = True
        self.input_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.test_size = (416, 416)
        self.mosaic_prob = 0.5
        self.enable_mixup = False


class YoloxPose(YoloxConfig):
    def __init__(self):
        super().__init__("yolox_pose")
        self.depth = 0.33
        self.width = 0.25
        self.depthwise = True
        self.input_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.test_size = (416, 416)
        self.mosaic_prob = 0.0
        self.enable_mixup = False
        self.num_classes = 1
        self.max_epoch = 300
        self.num_keypoints = 17
        self.coco_flip_map = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

        self.pose = True
        # name of annotation file for training
        self.train_ann: str = "person_keypoints_train2017.json"
        # name of annotation file for evaluation
        self.val_ann: str = "person_keypoints_val2017.json"

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        """
        Get dataloader according to cache_img parameter.
        Args:
            no_aug (bool, optional): Whether to turn off mosaic data enhancement. Defaults to False.
            cache_img (str, optional): cache_img is equivalent to cache_type. Defaults to None.
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
                None: Do not use cache, in this case cache_data is also None.
        """
        from yolox.data import (
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            TrainTransformPose,
            YoloBatchSampler,
            worker_init_reset_seed
        )
        from yolox.utils import wait_for_the_master

        # if cache is True, we will create self.dataset before launch
        # else we will create self.dataset after launch
        if self.dataset is None:
            with wait_for_the_master():
                assert cache_img is None, \
                    "cache_img must be None if you didn't create self.dataset before launch"
                self.dataset = self.get_dataset(cache=False, cache_type=cache_img)

        self.dataset = MosaicDetection(
            dataset=self.dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransformPose(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob, 
                keypoint_flip_indices=self.coco_flip_map),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        worker_init_fn = None if self.deterministic else worker_init_reset_seed

        train_loader = DataLoader(
            self.dataset,
            num_workers=self.data_num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
            worker_init_fn=worker_init_fn
        )

        return train_loader

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get dataset according to cache and cache_type parameters.
        Args:
            cache (bool): Whether to cache imgs to ram or disk.
            cache_type (str, optional): Defaults to "ram".
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
        """
        from yolox.data import CocoKeypointDataset, TrainTransformPose
        dataset = CocoKeypointDataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=TrainTransformPose(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
                keypoint_flip_indices=self.coco_flip_map
            ),
            cache=cache,
            cache_type=cache_type,
        )
        return dataset
    def get_model(self):
        from yolox.models import YoloPafpn, Yolox, YoloxPoseHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YoloPafpn(self.depth, self.width, in_channels=in_channels, depthwise=self.depthwise, act=self.act)
            head = YoloxPoseHead(17, self.width, in_channels=in_channels, depthwise=self.depthwise, act=self.act)
            self.model = Yolox(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model


    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import CocoPoseEvaluator

        return CocoPoseEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
            num_keypoints=17
        )


_NAMED_CONFIG: dict[str, YoloxConfig] = {
    config.name: config
    for config in (YoloxS(), YoloxM(), YoloxL(), YoloxX(), YoloxTiny(), YoloxNano(), YoloxPose())
}
