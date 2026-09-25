"""
Run the targeted mast-cell pipeline twice so the two mast definitions can be
compared directly:

  1. high_confidence: top 0.2-0.5% mast score and >=2 TPSAB1/TPSB2/CPA3 markers
  2. sensitive: top 0.5-1.0% mast score with >=2 core markers, or CPA3 plus
     TPSAB1/TPSB2 co-expression

Outputs:
  /content/drive/MyDrive/VisiumHD/DCIS/targeted_mast_compare/high_confidence/
  /content/drive/MyDrive/VisiumHD/DCIS/targeted_mast_compare/sensitive/
"""

import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_all_targeted_mast.py")
DRIVE_ROOT = Path("/content/drive/MyDrive/VisiumHD")
OUT_ROOT = DRIVE_ROOT / "DCIS" / "targeted_mast_compare"
H5_PATH = DRIVE_ROOT / "DCIS" / "feature_slice.h5"
SIGF_PATH = DRIVE_ROOT / "reference" / "pik3ca_gs_with_symbols.csv"
MASKS = ("high_confidence", "sensitive")


def run(mask_name: str) -> None:
    out_dir = OUT_ROOT / mask_name
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["ACTIVE_MAST_MASK"] = mask_name
    env["TARGETED_MAST_OUT"] = str(out_dir).replace("\\", "/")
    env["TARGETED_MAST_OUT_ROOT"] = str(OUT_ROOT).replace("\\", "/")
    env["BASE_DATA"] = str(DRIVE_ROOT / "DCIS").replace("\\", "/")
    env["H5"] = str(H5_PATH).replace("\\", "/")
    env["SIGF"] = str(SIGF_PATH).replace("\\", "/")

    print("\n" + "=" * 80, flush=True)
    print(f"Running targeted mast pipeline: {mask_name}", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print("=" * 80, flush=True)

    subprocess.run([sys.executable, str(SCRIPT)], check=True, env=env)


def main() -> None:
    for mask_name in MASKS:
        run(mask_name)

    print("\n" + "=" * 80)
    print("Comparison run complete.")
    for mask_name in MASKS:
        print(f"  {mask_name}: {OUT_ROOT / mask_name}")


if __name__ == "__main__":
    main()
