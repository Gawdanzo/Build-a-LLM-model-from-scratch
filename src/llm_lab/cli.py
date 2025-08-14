import argparse
from pathlib import Path

def download_sample(args=None):
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    sample = (Path(__file__).resolve().parents[2] / "data" / "raw" / "instruction-data.json")
    if sample.exists():
        target = out / "instruction-data.json"
        target.write_bytes(sample.read_bytes())
        print(f"Saved sample to {target}")
    else:
        print("Sample instruction-data.json not found in data/raw.")

def main():
    parser = argparse.ArgumentParser(prog="llm-lab")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_dl = sub.add_parser("download-sample", help="Copy the sample instruction-data.json to a target folder")
    p_dl.add_argument("--out_dir", type=str, default="./data", help="Where to place the sample file")
    p_dl.set_defaults(func=download_sample)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
