import torch

from .unet import *
from .multi_level import MultiLevelModel


def get_model_opt(args):
    if args.model == "unet":
        if args.method == "gaze_sup":
            # model = MultiLevelModel(in_channels=args.in_channels, num_levels=args.num_levels, num_classes=1)
            model1 = UNet_teacher(in_channels=args.in_channels, out_channels=2, feat_dim=128)
            # model2 = UNet(in_channels=args.in_channels, out_channels=2, feat_dim=128)
            model2 = UNet_student(in_channels=args.in_channels, out_channels=2, feat_dim=128)
        else:
            model = UNet(in_channels=args.in_channels, out_channels=1, feat_dim=128)
    else:
        raise NotImplementedError

    def make_opt(m):
        if m is None:
            return None
        if args.opt == "adam":
            return torch.optim.Adam(
                m.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        elif args.opt == "sgd":
            return torch.optim.SGD(
                m.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise NotImplementedError(f"Optimizer {args.opt} not implemented")
        
    optimizer1 = make_opt(model1)
    optimizer2 = make_opt(model2) if model2 is not None else None

    # �~O��~T�~[~^�~]~^空�~Z~Dmodel/optimizer
    models_list = [m for m in [model1, model2] if m is not None]
    opts_list = [o for o in [optimizer1, optimizer2] if o is not None]
    return models_list, opts_list
