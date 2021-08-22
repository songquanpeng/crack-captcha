from munch import Munch

from config import setup_cfg, validate_cfg, load_cfg, save_cfg, print_cfg
from data.loader import get_dataloader
from solver.solver import Solver


def main(args):
    solver = Solver(args)
    if args.mode == 'train':
        loaders = Munch(train=get_dataloader(dataloader_mode='train', **args),
                        test=get_dataloader(dataloader_mode='test', **args))
        solver.train(loaders)
    elif args.mode == 'eval':
        solver.evaluate()
    else:
        assert False, f"Unimplemented mode: {args.mode}"


if __name__ == '__main__':
    cfg = load_cfg()
    setup_cfg(cfg)
    validate_cfg(cfg)
    save_cfg(cfg)
    print_cfg(cfg)
    main(cfg)
