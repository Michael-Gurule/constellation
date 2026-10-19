from argparse import ArgumentParser
from datetime import datetime, timedelta

from constellation.legacy.features.feature_pipeline import FeaturePipeline
from constellation.legacy.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def main():
    parser = ArgumentParser(description="Run feature engineering pipeline")
    parser.add_argument("--days", type=int, default=7, help="Number of days to process")
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save processed features"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CONSTELLATION - Feature Engineering Pipeline")
    print("=" * 60)
    print()

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    print(f"Processing {args.days} days of data")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print()

    # Run pipeline
    pipeline = FeaturePipeline()
    results = pipeline.process_all_data(start_date, end_date, save=not args.no_save)

    # Summary
    print("\n" + "=" * 60)
    print("Feature Engineering Complete!")
    print("=" * 60)

    for subsystem, df in results.items():
        if not df.empty:
            print(f"\n{subsystem.upper()}:")
            print(f"  Records: {len(df):,}")
            print(f"  Features: {len(df.columns)}")
            print(f"  Size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            if not args.no_save:
                print("  ✓ Saved to data/processed/")


if __name__ == "__main__":
    main()
