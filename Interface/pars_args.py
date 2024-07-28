import argparse

def get_args_parser():
    parser = argparse.ArgumentParser("Training parameters", add_help=False)
    parser.add_argument("--prob_path", type=str, default="fm",
                        help="The probability path could be either Independent CFM (icfm) or Lipman's Flow Matching (fm).")
    parser.add_argument("--max_patience", type=int, default=50, help="Maximum patience.")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--M", type=int, default=1048, help="The number of neurons in scale (s) and translation (t) nets.")
    parser.add_argument("--sigma", type=float, default=0.1, help="Sigma.")
    parser.add_argument("--T", type=int, default=100, help="T.")

    parser.add_argument("--name", type=str, default="FM", help="Name of the model.")
    parser.add_argument("--extra_name", type=str, default="", help="Extra name.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Device.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers.")
    parser.add_argument("--output-dir", type=str, default="Results/", help="Output directory.")
    return parser
