import argparse
from .process_data import Processor

def main():
    parser = argparse.ArgumentParser(
        description="Compute features on L2 order‑book CSVs."
    )
    parser.add_argument("input_folder",  help="Folder containing raw CSVs")
    parser.add_argument("output_folder", help="Folder to write per‑product CSVs")
    args = parser.parse_args()

    proc = Processor()
    proc.run(args.input_folder, args.output_folder)
