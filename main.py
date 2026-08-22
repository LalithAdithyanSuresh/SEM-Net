import os

# Set local writable cache directory to avoid PermissionError on locked home directories (/home/cks)
if 'TORCH_HOME' not in os.environ:
    os.environ['TORCH_HOME'] = os.path.abspath('./tmp/torch_cache')
os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)

import cv2
import random
import numpy as np
import torch
import torch.hub
torch.hub.set_dir(os.path.join(os.environ['TORCH_HOME'], 'hub'))

import argparse
from shutil import copyfile
from src.config import Config
from src.sem import sem

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, reads from config file if not specified
    """
    config = load_config(mode)
    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


    # --- DDP Initialization ---
    world_size = len(config.GPU)
    if world_size > 1:
        # If running via torchrun, these will be set
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Set device for this process
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        config.DEVICE = torch.device(f"cuda:{local_rank}")
        config.RANK = rank
        config.WORLD_SIZE = world_size
        print(f"RANK {rank} initialized on device {config.DEVICE}")
    else:
        config.RANK = 0
        config.WORLD_SIZE = 1
        if torch.cuda.is_available():
            print('Cuda is available')
            config.DEVICE = torch.device("cuda")
        else:
            print('Cuda is unavailable, use cpu')
            config.DEVICE = torch.device("cpu")

    # --- Prepare Fast Temp Directory if requested ---
    if getattr(config, 'TMP_DIR', None) is not None:
        if config.RANK == 0:
            from src.utils import prepare_tmp_dir
            config, _ = prepare_tmp_dir(
                config,
                config.TMP_DIR,
                is_training=(config.MODE == 1)
            )
        # If running in multi-GPU DDP mode, other processes must wait for Rank 0 to finish copying
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
            
        # Non-zero ranks update their config.PATH and datasets paths to match Rank 0's updated config
        if config.RANK != 0:
            from src.utils import prepare_tmp_dir_non_zero_rank
            prepare_tmp_dir_non_zero_rank(
                config,
                config.TMP_DIR,
                is_training=(config.MODE == 1)
            )

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)


    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)



    # build the model and initialize
    model = sem(config)
    model.load()


    # model training
    if config.MODE == 1:
        if config.RANK == 0:
            config.print()
            print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()



def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints',
                        help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--model', type=int, default='2', choices=[1, 2, 3],
                        help='1: landmark prediction model, 2: inpaint model, 3: joint model')
    parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
    parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
    parser.add_argument('--landmark', type=str, help='path to the landmarks directory or a landmark file')
    parser.add_argument('--output', type=str, help='path to the output directory')
    parser.add_argument('--tmp-dir', type=str, default=None, help='use fast local storage directory for inputs, masks, outputs, and checkpoints')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        if os.path.exists('./config.yml'):
            copyfile('./config.yml', config_path)
        else:
            raise FileNotFoundError(f"Configuration file not found at {config_path} and no template ./config.yml exists.")

    # load config file
    config = Config(config_path)
    config.PATH = args.path
    print(config_path)

    # Save tmp-dir in config
    if args.tmp_dir is not None:
        config.TMP_DIR = args.tmp_dir

    # train mode
    if mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model
        if args.input is not None:
            config.TRAIN_INPAINT_IMAGE_FLIST = args.input
        if args.mask is not None:
            config.TRAIN_MASK_FLIST = args.mask
        if args.output is not None:
            config.RESULTS = args.output

    # test mode
    elif mode == 2:
        config.MODE = 2
        config.MODEL = args.model if args.model is not None else 3

        if args.input is not None:
            config.TEST_INPAINT_IMAGE_FLIST = args.input

        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask

        if args.output is not None:
            config.RESULTS = args.output

    return config


if __name__ == "__main__":
    import sys
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        print("\n[Training Interrupted by User (Ctrl+C)] Exiting...")
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        sys.exit(0)
