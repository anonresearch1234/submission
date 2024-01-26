import argparse
import datetime
import os
import pdb
import pickle
import random
import socket
import sys
import time
from copy import deepcopy
from re import S
from turtle import st

from gurobipy import GRB, Env, Model

import models.cvar as cvar
import models.entropy as entropy
import models.expectation as expectation
import models.instance_test as instance_test
import models.meandev as meandev
from models.det import deterministic_solve
from models.solvers import Problem, benders_solve, level_solve


def get_args():
    parser = argparse.ArgumentParser(description="IFS Experiment.")

    # Creating two variables using the add_argument method
    parser.add_argument("--loss",
                        default="expectation",
                        type=str,
                        help="which loss to use: det, expectation, CVAR, etc."
                        "If det (aka, DBA), we will save a Excel file.")
    parser.add_argument("--method",
                        default="level",
                        help="which method to use: single, bender, or level.")
    parser.add_argument(
        "--input",
        type=str,
        default="sample_data.pkl",
        help="input path")
    parser.add_argument("--scenario_size",
                        default=None,
                        type=int,
                        help="Number of scenarios for the second stage. "
                        "If unspecified, use all scenarios from the input.")
    parser.add_argument("--threshold",
                        default=None,
                        type=float,
                        help="The threshold for CVAR, entropy, etc. "
                        "Default is None, which uses threshold in code.")
    parser.add_argument("--seed",
                        default=None,
                        type=int,
                        help="The random seed for scenario sampling. "
                        "If not specified, then do not use any seeds.")
    parser.add_argument("--logname",
                        type=str,
                        default="out/out.log",
                        help="Output log path.")
    parser.add_argument("--supply_multiplier",
                        default=1.0,
                        type=float,
                        help="The supply multiplier to change supply amounts.")
    parser.add_argument("--timeout",
                        default=1000.0,
                        type=float,
                        help="Max time (seconds) to run the optimizer.")
    parser.add_argument("--throughput_scale",
                        default=1.0,
                        type=float,
                        help="Divide throughput in the dataset by this value")

    parser.add_argument("--calculate_second_stage_costs",
                        default=False,
                        action="store_true",
                        help="Only meaningful for `level` method. If True, "
                        "then calculate the second stage costs.")
    parser.add_argument("--compare_LP",
                        default=False,
                        action="store_true",
                        help="If True, compare Integer Linear Programming "
                        "and its LP Relaxation for second stage problems.")
    parser.add_argument("--bound_lp_binary",
                        default=False,
                        action="store_true",
                        help="If True, compare Integer Linear Programming "
                        "and its LP Relaxation for second stage problems.")

    parser.add_argument("--gurobi_log",
                        default=False,
                        action="store_true",
                        help="If True, record the Gurobi log.")
    args, unknown = parser.parse_known_args()

    return args


def get_env():
    env = Env(empty=True)
    env.setParam('LogToConsole', 0)
    env.start()
    return env


def run_experiment(args=None, env=None):
    if args is None:
        args = get_args()
    args.method == args.method.lower()
    args.loss == args.loss.lower()

    # Sanity Check Args
    assert args.method in ["single", "bender", "level", "det"]
    if args.method == "single":
        assert args.loss in ["expectation", "det", "test"]

    # Load data from disk and perform sampling if needed
    print("Loading data from disk...")
    data = pickle.load(open(args.input, "rb"))

    problem = Problem(data,
                      args.supply_multiplier,
                      args.timeout,
                      throughput_scale=args.throughput_scale,
                      logname=args.logname)
    problem.bound_lp_binary = args.bound_lp_binary

    if args.calculate_second_stage_costs or args.compare_LP:
        # Create a deep copy of the same problem for evaluation
        original_problem = Problem(deepcopy(data),
                                   args.supply_multiplier,
                                   args.timeout,
                                   throughput_scale=args.throughput_scale,
                                   logname="temp.log")
    else:
        original_problem = None

    if args.seed:
        random.seed(args.seed)
        problem.log("Seed set to {}".format(args.seed))

    if args.scenario_size:
        problem.sample_scenarios(num_scenarios=args.scenario_size)

    if env is None:
        env = get_env()

    # Setup Logging: use extension of .gurobi
    if args.gurobi_log or args.loss == "det":
        env.setParam('LogFile', args.logname + ".gurobi")
        print("Enable Gurobi Logfile:", args.logname)
    else:
        env.setParam('LogFile', "")  # No logfile.
        print("Disable Gurobi Logfile")

    problem.log("=" * 30, socket.gethostname(), "===", time.time(), "=" * 30)
    problem.log("Args:", args)
    problem.log("Sum Undocking Cost:", problem.sum_undock)
    # Import the Converge Functions
    if args.loss in ["expectation", "cvar", "entropy", "entropic", "meandev"]:

        if args.loss == "expectation":
            problem.log("Running Expectation Experiment.")
            module = expectation
        elif args.loss == "cvar":
            problem.log("Running CVAR Experiment.")
            module = cvar
        elif args.loss in ["entropy", "entropic"]:
            problem.log("Running Entropy Experiment.")
            module = entropy
        elif args.loss == "meandev":
            problem.log("Running Mean Deviation Experiment.")
            module = meandev

        benders_converge_func = module.benders_converge
        level_converge_func = module.level_converge
        module.set_threshold(args.threshold, problem)

    elif args.loss in ["det", "test"]:
        pass  # Handle separately
    else:
        raise ValueError("Loss should be one of expectation, cvar, det, "
                         "entropic, meandev, or test.")

    tic = time.time()
    problem.log("Model Start Time", tic)
    problem.log(datetime.datetime.now())
    problem.log("Computer Name: " + socket.gethostname())
    problem.start_time = tic
    if args.loss == "det":
        problem.log("Running Deterministic Experiment.")
        out = deterministic_solve(env, problem)
        out.save(args.logname.replace(".log", ".xls"))
    elif args.loss == "test":
        problem.log("Running Test Experiment.")
        out = instance_test.single_stage_solve(env, problem)
    elif args.method == "single":
        assert args.loss == "expectation"
        out = expectation.single_stage_solve(env, problem)
    elif args.method == "bender":
        out = benders_solve(env, problem, benders_converge_func)
    elif args.method == "level":
        out = level_solve(env,
                          problem,
                          original_problem,
                          benders_converge_func,
                          level_converge_func,
                          compare_LP=args.compare_LP)

    toc = time.time()
    if problem.solver_duration is None:
        dur = toc - tic
    else:
        dur = problem.solver_duration
    problem.log(
        f"It takes {dur} seconds to run the experiment, with results {out}.")

    problem.log("+" * 30, socket.gethostname(), "DONE", time.time(), "+" * 30)

    return out, dur


if __name__ == "__main__":
    run_experiment()
