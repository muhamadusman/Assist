"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from datetime import datetime
from PIL import Image

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def save_images(img,seed):#,rank):
    outdir = '/proj/assist/users/x_muhak/guided-diffusion/output1'
    split_arrays_3d = np.split(img, img.shape[0], axis=0)
    localseed=1
    for img in split_arrays_3d:
        name = localseed+seed
        # name = "Sample_"+str(rank)+"_"+str(name)
        name = "Sample_"+str(name)
        print ("Subject "+ str(name) + " Generated")
        localseed = localseed+1
        img = np.squeeze(img)
    
        #Extracting individual channels 
        channel1 = img[:,:,0]
        channel2 = img[:,:,1]
        channel3 = img[:,:,2]
        channel4 = img[:,:,3]
        channel5 = img[:,:,4]

        Image.fromarray(channel1).save(f'{outdir}/T1/{name}_T1.png')
        Image.fromarray(channel2).save(f'{outdir}/T2/{name}_T2.png')
        Image.fromarray(channel3).save(f'{outdir}/FLAIR/{name}_flair.png')
        Image.fromarray(channel4).save(f'{outdir}/T1ce/{name}_T1ce.png')
        Image.fromarray(channel5).save(f'{outdir}/SEG/{name}_Seg.png')
 
def main():
    slice = 0 
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    qwe = 0
    seed =0
    while len(all_images) * args.batch_size < args.num_samples:
        slice = slice+1
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        # print ("Device : ",dist_util.dev())
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 5, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        print ("Shape of sample : ",sample.cpu().numpy().shape)
        
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        
        for sample in gathered_samples:
            all_images.extend([sample.cpu().numpy()])
            # img = sample.cpu().numpy()
            # save_images(img,seed)#, dist.get_rank())
            # seed = seed + args.batch_size
        
        # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")
    
    

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npy")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.save(out_path, arr)#, label_arr)
            print ("The shape of array : ",arr.shape)
        else:
            np.save(out_path, arr)
            print ("The shape of array : ",arr.shape)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=16,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    import time
    t = time.localtime()
    print ( " Time Started :  ", time.strftime("%H:%M:%S", t))
    main()
    t2 = time.localtime()
    print ( " Time Finished :  ", time.strftime("%H:%M:%S", t2))
    
