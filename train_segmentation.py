import argparse

from pointprogrammernet.training import train_segmentation


def arguments():
    parser = argparse.ArgumentParser(description="Train final PointProgrammerNet on ShapeNetPart")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--data-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="results/segmentation")
    args = parser.parse_args()
    args.seeds = [int(seed) for seed in args.seeds.split(",")]
    return args


if __name__ == "__main__":
    train_segmentation(arguments())
