#!/usr/bin/env python3
import os
from orderbook_features.process_data import Processor

def main():
    # base of this project
    cwd = os.getcwd()

    # where your raw L2 CSVs live
    input_folder = os.path.join(cwd, "data", "prices", "round_four")

    # where you want the per‑product CSVs to go
    output_folder = os.path.join(cwd, "data", "processed_round_4")
    os.makedirs(output_folder, exist_ok=True)

    print(f"Reading from:  {input_folder}")
    print(f"Writing to:    {output_folder}")

    proc = Processor()
    proc.run(input_folder, output_folder)

    print("✅ Done!")

if __name__ == "__main__":
    main()
