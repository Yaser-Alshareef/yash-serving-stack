import os
import random
import subprocess
import time

TARGET = os.environ.get(
    "TARGET_CONTAINER",
    "serving"
)

MIN_INTERVAL = float(
    os.environ.get("MIN_INTERVAL_S", "12")
)

MAX_INTERVAL = float(
    os.environ.get("MAX_INTERVAL_S", "15")
)

print(
    f"chaos: will crash {TARGET} "
    f"every {MIN_INTERVAL}-{MAX_INTERVAL}s"
)

while True:
    time.sleep(
        random.uniform(
            MIN_INTERVAL,
            MAX_INTERVAL
        )
    )

    result = subprocess.run(
        [
            "docker",
            "exec",
            TARGET,
            "sh",
            "-c",
            "kill -9 1"
        ],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(
            f"chaos: crashed {TARGET}"
        )
    else:
        print(
            "chaos: crash failed: "
            + result.stderr.strip()
        )