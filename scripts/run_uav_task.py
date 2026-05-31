from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carlaair_active_world.scenario import ScenarioConfig

DEFAULT_SCENARIO = ROOT / "configs" / "scenarios" / "town10hd_v0.json"
DEFAULT_OUTPUT_DIR = ROOT / "recordings" / "active_uav_task"


def main() -> None:
    if sys.version_info[:2] != (3, 10):
        raise SystemExit(
            f"This task expects the CarlaAir Python 3.10 environment, but the current interpreter is "
            f"{sys.version_info.major}.{sys.version_info.minor}."
        )

    try:
        from carlaair_active_world.task_app import ActiveUAVTaskApp
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing runtime dependency. Activate the CarlaAir env first, then ensure CARLA and AirSim "
            "are installed in that Python 3.10 environment."
        ) from exc

    parser = argparse.ArgumentParser(description="Interactive CarlaAir UAV + traffic task V0.")
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-hz", type=float, default=2.0)
    args = parser.parse_args()

    scenario = ScenarioConfig.load(args.scenario)
    app = ActiveUAVTaskApp(scenario=scenario, output_dir=args.output_dir, sample_hz=args.sample_hz)
    app.connect()
    app.setup()
    app.start_sampler()

    print(app.describe())
    print("Type 'help' for commands.")

    try:
        while True:
            line = input("> ").strip()
            if not line:
                continue
            result = app.handle_command(line)
            if result == "quit":
                break
            print(result)
    except KeyboardInterrupt:
        pass
    finally:
        saved = app.stop_sampler()
        app.cleanup()
        print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
