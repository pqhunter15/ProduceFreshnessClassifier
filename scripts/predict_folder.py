from pathlib import Path
import argparse
import pandas as pd
import sys

# Add the parent directory to sys.path to import from src
sys.path.append(str(Path(__file__).parent.parent))

from src.freshness_inference.model import FreshnessClassifier


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run freshness inference on all images in a folder."
    )
    parser.add_argument("--folder-path", default="sample_images/", help="Folder of images.")
    parser.add_argument(
        "--output-csv",
        default="predictions.csv",
        help="Where to save the predictions CSV.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold for Rotten class.",
    )
    args = parser.parse_args()

    classifier = FreshnessClassifier(threshold=args.threshold)

    folder = Path(args.folder_path)
    image_paths = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS
    )

    rows = []
    for image_path in image_paths:
        try:
            result = classifier.predict_image(str(image_path)).to_dict()
            rows.append(result)
        except Exception as e:
            rows.append(
                {
                    "image_path": str(image_path),
                    "predicted_class": None,
                    "predicted_label": None,
                    "fresh_probability": None,
                    "rotten_probability": None,
                    "threshold": args.threshold,
                    "error": str(e),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    print(df.head())
    print(f"\nSaved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
