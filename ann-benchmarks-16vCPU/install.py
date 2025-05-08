import argparse
import os
import subprocess
import sys
from multiprocessing import Pool

from ann_benchmarks.main import positive_int


def build(library, args, no_cache=False):
    print("Building %s..." % library)
    if args is not None and len(args) != 0:
        q = " ".join(["--build-arg " + x.replace(" ", "\\ ") for x in args])
    else:
        q = ""

    no_cache_flag = "--no-cache" if no_cache else ""

    try:
        cmd = f"docker build {q} {no_cache_flag} --rm -t ann-benchmarks-{library} -f ann_benchmarks/algorithms/{library}/Dockerfile ."
        print(f"Running command: {cmd}")
        subprocess.check_call(
            cmd,
            shell=True,
        )
        return {library: "success"}
    except subprocess.CalledProcessError:
        return {library: "fail"}


def build_multiprocess(args):
    library, build_args, no_cache = args
    return build(library, build_args, no_cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--proc", default=1, type=positive_int, help="the number of process to build docker images")
    parser.add_argument("--algorithm", metavar="NAME", help="build only the named algorithm image", default=None)
    parser.add_argument("--build-arg", help="pass given args to all docker builds", nargs="+")
    parser.add_argument("--no-cache", help="force docker build without cache", action='store_true')
    args = parser.parse_args()

    print("Building base image...")
    subprocess.check_call(
         "docker build \
         --rm -t ann-benchmarks -f ann_benchmarks/algorithms/base/Dockerfile .",
         shell=True,
     )
    
    if args.algorithm:
        tags = [args.algorithm]
    elif os.getenv("LIBRARY"):
        tags = [os.getenv("LIBRARY")]
    else:
        tags = [fn.split(".")[-1] for fn in os.listdir("ann_benchmarks/algorithms")]

    print("Building algorithm images... with (%d) processes" % args.proc)

    if args.proc == 1:
        install_status = [build(tag, args.build_arg, args.no_cache) for tag in tags]
    else:
        pool = Pool(processes=args.proc)
        tasks = [(tag, args.build_arg, args.no_cache) for tag in tags]
        install_status = pool.map(build_multiprocess, tasks)
        pool.close()
        pool.join()

    print("\n\nInstall Status:\n" + "\n".join(str(algo) for algo in install_status))

    # Exit 1 if any of the installations fail.
    for x in install_status:
        for (k, v) in x.items():
            if v == "fail":
                sys.exit(1)
