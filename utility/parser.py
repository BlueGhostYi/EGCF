import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="RS")

    parser.add_argument("--seed_flag", type=bool, default=True, help="Fix random seed or not")

    parser.add_argument("--seed", type=int, default=2023, help="random seed for init")

    parser.add_argument("--data_path", nargs="?", default="./Data/", help="Input data path.")

    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")

    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")

    parser.add_argument("--verbose", type=int, default=1, help="Test interval")

    parser.add_argument("--stop", type=int, default=500, help="early stopp")

    parser.add_argument("--save", type=bool, default=False, help="save model or not")

    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")

    return parser.parse_args()
