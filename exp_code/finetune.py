import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
from tqdm import tqdm
from runners.diffusion import Diffusion
from torchvision import transforms
import torchvision
from datasets import get_dataset, data_transform, inverse_data_transform
from utils import UnlabeledImageFolder
from accelerate import Accelerator
torch.set_printoptions(sci_mode=False)


def parse_args_and_config(accelerator):
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=2333, help="Random seed")
    parser.add_argument("--taylor_batch_size", type=int, default=128, help="batch size for taylor expansion")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--kd",
        action="store_true",
        default=False,
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )

    parser.add_argument(
        "--load_pruned_model", type=str, default=None, help="load pruned models"
    )

    parser.add_argument(
        "--save_pruned_model", type=str, default=None, help="load pruned models"
    )

    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )

    parser.add_argument(
        "--pruner",
        type=str,
        default="taylor",
        choices=["taylor", "random", "magnitude", "reinit", "first_order_taylor", "second_order_taylor"],
    )

    parser.add_argument(
        "--restore_from",
        type=str,
        default=None,
        help="Restore from user a checkpoint",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        default=0.0,
        help="pruning ratio",
    )
    
    parser.add_argument("--sequence", action="store_true")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)
    
    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

        #tb_path = os.path.join(args.exp, "tensorboard", args.doc)
    if accelerator.is_main_process:
        if not args.test and not args.sample:
            if not args.resume_training:
                if os.path.exists(args.log_path):
                    overwrite = False
                    if args.ni:
                        overwrite = True
                    else:
                        response = input("Folder already exists. Overwrite? (Y/N)")
                        if response.upper() == "Y":
                            overwrite = True

                    if overwrite:
                        shutil.rmtree(args.log_path)
                        #shutil.rmtree(tb_path)
                        os.makedirs(args.log_path)
                        #if os.path.exists(tb_path):
                        #    shutil.rmtree(tb_path)
                    else:
                        print("Folder exists. Program halted.")
                        sys.exit(0)
                else:
                    os.makedirs(args.log_path)

                with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                    yaml.dump(new_config, f, default_flow_style=False)
            os.makedirs(os.path.join(args.log_path, 'vis'), exist_ok=True)
            #new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
            # setup logger
            level = getattr(logging, args.verbose.upper(), None)
            if not isinstance(level, int):
                raise ValueError("level {} not supported".format(args.verbose))

            handler1 = logging.StreamHandler()
            handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
            formatter = logging.Formatter(
                "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
            )
            handler1.setFormatter(formatter)
            handler2.setFormatter(formatter)
            logger = logging.getLogger()
            logger.addHandler(handler1)
            logger.addHandler(handler2)
            logger.setLevel(level)
        else:
            level = getattr(logging, args.verbose.upper(), None)
            if not isinstance(level, int):
                raise ValueError("level {} not supported".format(args.verbose))

            handler1 = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
            )
            handler1.setFormatter(formatter)
            logger = logging.getLogger()
            logger.addHandler(handler1)
            logger.setLevel(level)

    if args.sample:
        os.makedirs(os.path.join(args.exp, "image_samples", args.image_folder, str(accelerator.process_index)), exist_ok=True)
        args.image_folder = os.path.join(
            args.exp, "image_samples", args.image_folder, str(accelerator.process_index)
        )
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            if not (args.fid or args.interpolation):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input(
                        f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                    )
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.image_folder)
                    os.makedirs(args.image_folder)
                else:
                    print("Output image folder exists. Program halted.")
                    sys.exit(0)

    # add device
    #device = #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #logging.info("Using device: {}".format(device))
    #new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    accelerator = Accelerator()
    args, config = parse_args_and_config(accelerator)
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    try:
        runner = Diffusion(args, config)
        runner.accelerator = accelerator
        if args.sample:
            runner.sample()
        elif args.test:
            runner.test()
        else:
            runner.train(kd=args.kd)
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
