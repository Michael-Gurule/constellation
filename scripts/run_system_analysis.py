from argparse import ArgumentParser

from constellation.legacy.optimization.system_integrator import ConstellationSystem


def main():
    parser = ArgumentParser(description="Run CONSTELLATION system analysis")
    parser.add_argument(
        "--subsystem",
        type=str,
        default="attitude_control",
        choices=["attitude_control", "communications"],
        help="Subsystem to analyze",
    )
    parser.add_argument("--export", action="store_true", help="Export results to files")

    args = parser.parse_args()

    print("=" * 60)
    print("CONSTELLATION - System Analysis")
    print("=" * 60)
    print(f"Subsystem: {args.subsystem}")
    print()

    # Initialize system
    system = ConstellationSystem()

    # Run analysis
    results = system.run_complete_analysis(subsystem=args.subsystem)

    # Export if requested
    if args.export:
        system.export_results()

    # Print summary
    print("\n" + "=" * 60)
    print("Analysis Summary")
    print("=" * 60)

    if "health_scores" in results:
        health = results["health_scores"]
        print(f"\nComponents: {len(health)}")
        print(f"Average Health: {health['health_score'].mean():.1f}")
        print(f"Min Health: {health['health_score'].min():.1f}")
        print(f"Critical Components: {(health['health_score'] < 40).sum()}")

    if "alerts" in results:
        from constellation.legacy.optimization.alert_generator import AlertSeverity

        alerts = results["alerts"]
        print(f"\nTotal Alerts: {len(alerts)}")
        print(
            f"  Critical: {sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)}"
        )
        print(
            f"  Warning: {sum(1 for a in alerts if a.severity == AlertSeverity.WARNING)}"
        )

    if "maintenance_schedule" in results:
        schedule = results["maintenance_schedule"]
        if not schedule.empty:
            print(f"\nMaintenance Tasks: {len(schedule)}")
            print(f"Total Duration: {schedule['duration_hours'].sum():.1f} hours")


if __name__ == "__main__":
    main()
