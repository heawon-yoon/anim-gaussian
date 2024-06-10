
import torch
import sys
from avatar import Avatar, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams


def render_and_anim(dataset, pipe):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Avatar(dataset, gaussians)
    scene.gaussians.load_ply('data/humans/point_cloud.ply')

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    #scene.render_canonical(nframes=20, is_train_progress=False, pose_type=None, pipe=pipe, bg=background)
    anim_path = './data/SFU/0005/0005_SideSkip001_poses.npz'
    scene.animate(anim_path, pipe=pipe, bg=background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="rendering script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    args = parser.parse_args(sys.argv[1:])
    args.test = True
    print("Optimizing " + args.model_path)

    render_and_anim(lp.extract(args), pp.extract(args))

    # All done
    print("\nrender complete.")