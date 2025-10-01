# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import importlib
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(app, args, resume_preempt=False,debug=False,save_mask=False): # app_main(params['app'], args=params, debug=debug)

    logger.info(f'Running pre-training of app: {app}')

    return importlib.import_module(f'app_scratch.{app}.train_v2').main(
        args=args,
        resume_preempt=resume_preempt,debug=debug,save_mask=save_mask)

