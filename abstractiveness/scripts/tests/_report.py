"""
Shared runner of every check script: collects failures, prints OK or FAIL, returns the exit code.

Each check_*.py defines checks(failures) -> summary, which appends a message per violated
property (or raises AssertionError) and returns a one-line summary for the OK line.
"""
import pathlib
import traceback


def run(script: str, checks) -> int:
    """Run one check script's checks and report, 0 when every property holds, 1 otherwise."""
    name = pathlib.Path(script).stem
    failures, summary = [], ""
    try:
        summary = checks(failures) or ""
    except AssertionError as error:
        failures.append(str(error) or "assertion failed")
    except Exception:  # noqa: BLE001, a crash is a failure of the check, reported with its traceback
        failures.append("raised:\n" + traceback.format_exc())
    if failures:
        print(f"FAIL {name}")
        for line in failures:
            print(f"  {line}")
        return 1
    print(f"OK {name}" + (f": {summary}" if summary else ""))
    return 0
