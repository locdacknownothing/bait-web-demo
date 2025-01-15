from os.path import join, dirname
import sys

sys.path.append(join(dirname(__file__), ".."))
import argparse
from pathlib import Path

from dataset import convert_data, ent_maps
from file import read_df, run_command
from spacy_model.save_dataset import save_data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-data-path",
        "-t",
        type=str,
        help="The csv/xlsx file path of training dataset",
        required=True,
    )
    parser.add_argument(
        "--val-data-path",
        "-v",
        type=str,
        help="The csv/xlsx file path of validation dataset",
        required=True,
    )
    parser.add_argument(
        "--train-save-path",
        type=str,
        help="The saving spacy file path of training dataset",
        default="train.spacy",
    )
    parser.add_argument(
        "--val-save-path",
        type=str,
        help="The saving spacy file path of validation dataset",
        default="val.spacy",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        help="The output folder to save model",
        default="./result",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=int,
        help="Device to use for training",
        default=0,
    )

    args = parser.parse_args()
    return args


def train(args):
    src_train = args.train_data_path
    src_val = args.val_data_path

    df_train = read_df(src_train)
    df_val = read_df(src_val)

    data_train = convert_data(df_train, ent_maps)
    data_val = convert_data(df_val, ent_maps)

    print(f"Train dataset: {len(data_train)}")
    print(f"Validation dataset: {len(data_val)}")

    save_data(data_train, args.train_save_path)
    save_data(data_val, args.val_save_path)

    config_path = "./configs/config.cfg"
    output = Path(args.output_path)
    output.mkdir(exist_ok=True, parents=True)
    train_command = f"python3 -m spacy train {config_path} --output {str(output)} --paths.train {args.train_save_path} --paths.dev {args.val_save_path} --gpu-id {args.device}"
    run_command(train_command)


if __name__ == "__main__":
    args = parse_args()
    train(args)
