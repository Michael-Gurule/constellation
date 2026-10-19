from argparse import ArgumentParser
from datetime import datetime, timedelta

from constellation.legacy.models.model_trainer import ModelTrainer


def main():
    parser = ArgumentParser(description="Train ML models")
    parser.add_argument(
        "--subsystem",
        type=str,
        default="attitude_control",
        choices=["attitude_control", "communications"],
        help="Subsystem to train models for",
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Number of days of data to use"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("CONSTELLATION - Model Training")
    print("=" * 60)
    print(f"Subsystem: {args.subsystem}")
    print(f"Training data: {args.days} days")
    print()

    # Initialize trainer
    trainer = ModelTrainer()

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    # Train all models
    models = trainer.train_all_models(args.subsystem, start_date, end_date)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nTrained models: {len(models)}")
    for name in models.keys():
        print(f"  ✓ {name}")


if __name__ == "__main__":
    main()
