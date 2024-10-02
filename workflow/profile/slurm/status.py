#!/usr/bin/env python
import subprocess


def job_status(jobid: int):
    output = str(
        subprocess.check_output(
            "sacct -j %s --format State --noheader | head -1 | awk '{print $1}'"
            % jobid,
            shell=True,
        ).strip()
    )

    running_status = ["PENDING", "CONFIGURING", "COMPLETING", "RUNNING", "SUSPENDED"]
    if "COMPLETED" in output:
        return "success"
    elif any(r in output for r in running_status):
        return "running"
    else:
        return "failed"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("jobid", type=int, help="jobid to moniter status")
    args = parser.parse_args()

    print(job_status(args.jobid))
