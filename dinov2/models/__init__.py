# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from . import vision_transformer as vits
from .nextvit import NextVitSmall, load_weights


logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        if args.arch in vits.__dict__:
            teacher = vits.__dict__[args.arch](**vit_kwargs)
        else:
            teacher = NextVitSmall()
            teacher = load_weights(teacher)
        if only_teacher:
            return teacher, teacher.embed_dim
        if args.arch in vits.__dict__:
            student = vits.__dict__[args.arch](
                **vit_kwargs,
                drop_path_rate=args.drop_path_rate,
                drop_path_uniform=args.drop_path_uniform,
            )
        else:
            student = NextVitSmall()
            student = load_weights(student)
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)
