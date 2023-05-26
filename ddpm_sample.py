from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel
import argparse, os, torch
from tqdm import tqdm
import torch_pruning as tp
import accelerate

parser = argparse.ArgumentParser()
parser.add_argument("--total_samples", type=int, default=50000)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--output_dir", type=str, default="samples")
parser.add_argument("--model_path", type=str, default="samples")
parser.add_argument("--ddim_steps", type=int, default=100)
parser.add_argument("--pruned_model_ckpt", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--skip_type", type=str, default="uniform")

args = parser.parse_args()

if __name__ == "__main__":
    os.makedirs(args.output_dir, exist_ok=True)
    # pruned model
    accelerator = accelerate.Accelerator()

    if os.path.isdir(args.model_path):
        if args.pruned_model_ckpt is not None:
            print("Loading pruned model from {}".format(args.pruned_model_ckpt))
            unet = torch.load(args.pruned_model_ckpt).eval()
        else:
            print("Loading model from {}".format(args.model_path))
            subfolder = 'unet' if os.path.isdir(os.path.join(args.model_path, 'unet')) else None
            unet = UNet2DModel.from_pretrained(args.model_path, subfolder=subfolder).eval()
        pipeline = DDIMPipeline(
            unet=unet,
            scheduler=DDIMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
        )
    # standard model
    else:  
        print("Loading pretrained model from {}".format(args.model_path))
        pipeline = DDIMPipeline.from_pretrained(
            args.model_path,
        )

    pipeline.scheduler.skip_type = args.skip_type
    # Test Flops
    pipeline.to(accelerator.device)
    if accelerator.is_main_process:
        if 'cifar' in args.model_path:
            example_inputs = {'sample': torch.randn(1, 3, 32, 32).to(accelerator.device), 'timestep': torch.ones((1,)).long().to(accelerator.device)}
        else:
            example_inputs = {'sample': torch.randn(1, 3, 256, 256).to(accelerator.device), 'timestep': torch.ones((1,)).long().to(accelerator.device)}
        macs, params = tp.utils.count_ops_and_params(pipeline.unet, example_inputs)
        print(f"MACS: {macs/1e9} G, Params: {params/1e6} M")

    # Create subfolders for each process
    save_sub_dir = os.path.join(args.output_dir, 'process_{}'.format(accelerator.process_index))
    os.makedirs(save_sub_dir, exist_ok=True)
    generator = torch.Generator(device=pipeline.device).manual_seed(args.seed+accelerator.process_index)

    # Set up progress bar
    if not accelerator.is_main_process:
        pipeline.set_progress_bar_config(disable=True)
    
    # Sampling
    accelerator.wait_for_everyone()
    with torch.no_grad():
        # num_batches of each process
        num_batches = (args.total_samples) // (args.batch_size * accelerator.num_processes)
        if accelerator.is_main_process:
            print("Samping {}x{}={} images with {} process(es)".format(num_batches*args.batch_size, accelerator.num_processes, num_batches*accelerator.num_processes*args.batch_size, accelerator.num_processes))
        for i in tqdm(range(num_batches), disable=not accelerator.is_main_process):
            images = pipeline(batch_size=args.batch_size, num_inference_steps=args.ddim_steps, generator=generator).images
            for j, image in enumerate(images):
                filename = os.path.join(save_sub_dir, f"{i * args.batch_size + j}.png")
                image.save(filename)

    # Finished
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print(f"Saved {num_batches*accelerator.num_processes*args.batch_size} samples to {args.output_dir}")
    #accelerator.end_training()
    
