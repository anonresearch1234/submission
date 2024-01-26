"""Common solvers for all methods."""
import pdb
import pickle
import random
import time
from collections import defaultdict
from math import ceil, inf

import numpy as np
import pandas as pd
from gurobipy import GRB, Model
from termcolor import colored

np.set_printoptions(precision=40)


def load_problem(loc):
    """Load a problem from a pickle file.

    Args:
        loc (str): location of the pickle file.

    Returns:
        problem (object): a Problem object with all input data.
    """
    (inputs, supply_multiplier, timeout, throughput_scale,
     logname) = pickle.load(open(loc, "rb"))

    problem = Problem(inputs,
                      supply_multiplier=supply_multiplier,
                      timeout=timeout,
                      throughput_scale=throughput_scale,
                      logname=logname)
    return problem

class Problem:
    """Problem Setting for the two-stage stochastic program."""
    def dump(self, loc):
        inputs = (self.demand_costs, self.shipping_costs, self.row_costs,
         self.sum_undock, self.demands, self.supplies, self.existing_rows,
         self.datacenters, self.throughput_capacity, self.capacity_indices,
         self.center_throughput, self.dates, self.potential_center_dates,
         self.completion_date, self.completion_date_index, self.sample_size,
         self.sample_demand_costs, self.sample_shipping_costs,
         self.sum_sample_undock, self.sample_undock_costs_sum,
         self.sample_demands, self.sample_potential_center_dates)

        # dump with pickle
        pickle.dump((inputs, self.supply_multiplier, self.timeout, self.throughput_scale, self.logname), open(loc, "wb"))
          

    def __init__(self,
                 inputs,
                 supply_multiplier=1,
                 timeout=1000,
                 throughput_scale=1.0,
                 logname="out.log"):
        (self.demand_costs, self.shipping_costs, self.row_costs,
         self.sum_undock, self.demands, self.supplies, self.existing_rows,
         self.datacenters, self.throughput_capacity, self.capacity_indices,
         self.center_throughput, self.dates, self.potential_center_dates,
         self.completion_date, self.completion_date_index, self.sample_size,
         self.sample_demand_costs, self.sample_shipping_costs,
         self.sum_sample_undock, self.sample_undock_costs_sum,
         self.sample_demands, self.sample_potential_center_dates) = inputs

        self.inputs = inputs

        self.throughput_capacity = [(centers, [
            (t, ceil(max_tp / throughput_scale)) for t, max_tp in tps
        ]) for centers, tps in self.throughput_capacity]
        print("Len of Throughput (after change)", len(self.throughput_capacity))

        all_tp = [
            max_tp for centers, tps in self.throughput_capacity
            for t, max_tp in tps
        ]
        all_tp = np.array(all_tp).reshape(-1).astype(float)
        print(f"Throughput Stats: mean = {np.mean(all_tp)},"
              f"std = {np.std(all_tp)}")

        self.supply_multiplier = supply_multiplier
        self.start_time = None # We must set this before solving the problem. Intentionally set to None here to trigger error if not set.
        self.throughput_scale = throughput_scale
        self.timeout = timeout
        self.logname = logname
        self.logfile = open(logname, "a")

        self.solver_duration = None  # To store solver's runtime

        # use all scenarios by default
        self.scenario_indices = list(range(self.sample_size))

    def __del__(self):
        try:
            self.logfile.close()
        except:
            pass

    def log(self, msg, *args, **kwargs):
        """Write to both the terminal and teh logfile

        All args, kwargs will be passed to the Python print function.

        Args:
            msg (str): message to write.
        """
        # Write to both the log file and the console
        print(msg, *args, **kwargs)
        print(msg, *args, file=self.logfile, **kwargs)

    def flush(self):
        self.logfile.flush()

    def sample_scenarios(self, num_scenarios, sample_set:list=[]):
        """Sample scenarios from all valid scenarios.

        Args:
            num_scenarios (int): number of scenarios to sample.
            sample_set (list, Optional): a provided index for scenarios to sample. 
        """
        assert num_scenarios <= self.sample_size, (
            "The new number of scenarios"
            " is even less than the previous number of scenarios")

        if not sample_set:
            sample_set = random.sample(range(self.sample_size), num_scenarios)
        self.scenario_indices = sample_set

        self.log("Scenario Sampled Set: {}".format(sample_set))

        self.sample_size = num_scenarios
        self.sample_demand_costs = {
            i: self.sample_demand_costs[sample_set[i]]
            for i in range(num_scenarios)
        }
        self.sample_shipping_costs = {
            i: self.sample_shipping_costs[sample_set[i]]
            for i in range(num_scenarios)
        }
        self.sample_undock_costs_sum = {
            i: self.sample_undock_costs_sum[sample_set[i]]
            for i in range(num_scenarios)
        }
        self.sample_demands = {
            i: self.sample_demands[sample_set[i]] for i in range(num_scenarios)
        }
        self.sample_potential_center_dates = {
            i: self.sample_potential_center_dates[sample_set[i]]
            for i in range(num_scenarios)
        }

    def sample_subrestriction_stats(self):
        return samples_subrestriction_violation_stats(self.throughput_capacity,
                                                      self.sample_demand_costs)


def benders_solve(env, problem, benders_converge_func):
    """Solve the problem using Benders' decomposition.

    Args:
        env (object): Gurobi environment.
        problem (object): a Problem object with all input data.
        benders_converge_func (function): a bender function with defined risk
            measures.

    Returns:
        loss (float): total loss of the two-stage stochastic problem.
    """
    tic = time.time()
    master, (theta, delta, rho,
             sigma) = benders_converge_func(first_stage_model(env, problem),
                                            second_stage_model(env, problem),
                                            problem)

    toc = time.time()
    problem.solver_duration = toc - tic
    problem.log("Benders Solver Duration: {}".format(problem.solver_duration))
    return master.getObjective().getValue() + problem.sum_undock


def level_solve(env,
                problem,
                original_problem,
                benders_converge_func,
                level_converge_func,
                compare_LP=False):
    """Solve the problem using the proposed hybrid method.

    Args:
        env (object): Gurobi environment.
        problem (object): a Problem object with all input data.
        benders_converge_func (function): a bender function with defined risk
            measures.
        level_converge_func (function): a level function with defined risk
            measures.

    Returns:
        loss (float): total loss of the two-stage stochastic problem.
    """
    print(colored("WITH IN level_solve", "red"))
    tic = time.time()
    first_stage = first_stage_model(env,
                                    problem,
                                    relax=True,
                                    bound_lp_binary=problem.bound_lp_binary)
    second_stage = second_stage_model(env, problem)
    level_converge_func(first_stage, second_stage, level_model(env, problem),
                        problem)
    toc_level_converge = time.time()
    problem.log(f"First Phase (Level) Duration: {toc_level_converge - tic}")
    tic_bender_start = time.time()

    master, (theta, delta, rho, sigma), x, w, z = first_stage
    for var in z.values():
        var.setAttr('vtype', GRB.BINARY)
    for var in w.values():
        var.setAttr('vtype', GRB.BINARY)
    for var in x.values():
        var.setAttr('vtype', GRB.INTEGER)

    master, (theta, delta, rho,
             sigma) = benders_converge_func(first_stage, second_stage, problem)
    toc = time.time()
    problem.log(f"Second Phase (Bender) Duration: {toc - tic_bender_start}")
    problem.solver_duration = toc - tic
    problem.log("Level Solver Duration: {}".format(problem.solver_duration))
    _, (_, delta, rho, sigma), x, w, z = first_stage

    # Print Results
    total_loss = master.getObjective().getValue() + problem.sum_undock
    problem.log("Total Loss:", total_loss)
    
    sample_problems, sample_lambda = second_stage

    try:
        problem.log(
            "First Stage Cost:",
            master.getObjective().getValue() - theta.x + problem.sum_undock)
        problem.log("master.getObjective", master.getObjective().getValue())
        problem.log("sum_undock", problem.sum_undock)
        total_row_cost = 0
        for r in problem.datacenters:
            total_row_cost += x[r].x * problem.row_costs[r]
        problem.log("Row-building Cost (within the first stage):",
                    total_row_cost)
    except Exception as e:
        # When the master is not solved (converged) due to timeout,
        # errors might occur.
        problem.log("Incomplete solution (error reporting row costs)):", str(e))

    if original_problem is None:
        # No original problem passed
        return total_loss

    # Log the Second-Stage Costs (for all scenarios, including unseen)
    second_stage_problems, second_stage_lambda = second_stage_model(
        env, original_problem)  # optimize each second-stage scenario
    all_second_stage_costs = []
    for i in range(original_problem.sample_size):
        second_stage_problems[i].setObjective(
            second_stage_lambda[i]['d'].sum('*') +
            sum(delta[n, t].x * second_stage_lambda[i]['delta'][n, t]
                for n in original_problem.capacity_indices
                for t in original_problem.dates[original_problem.
                                                completion_date_index:]) +
            sum(rho[r].x * second_stage_lambda[i]['rho'][r]
                for r in original_problem.datacenters) +
            sum(sigma[s].x * second_stage_lambda[i]['sigma'][s]
                for s in original_problem.supplies), GRB.MAXIMIZE)

        second_stage_problems[i].setParam(
            'TimeLimit',
            max(0, problem.timeout - (time.time() - problem.start_time)))
        second_stage_problems[i].optimize()
        all_second_stage_costs.append(
            second_stage_problems[i].getObjective().getValue() +
            original_problem.sample_undock_costs_sum[i])
        problem.log(f"(Testing) Second Stage, Scenario {i}'s Cost:",
                    all_second_stage_costs[-1])

    problem.log("All Second Stage Costs:", all_second_stage_costs)

    if not compare_LP:
        return total_loss

    # Solve the LP relaxation and the IP of the second stage for each scenario
    second_stage_primal_LP_costs, all_decisions_lp = second_stage_model_primal(
        env, original_problem, first_stage, relaxation=True)
    problem.log("Second Stage LP Costs:", second_stage_primal_LP_costs)

    second_stage_primal_IP_costs, all_decisions_ip = second_stage_model_primal(
        env, original_problem, first_stage, relaxation=False)
    problem.log("Second Stage IP Costs:", second_stage_primal_IP_costs)

    diff = np.array(second_stage_primal_LP_costs) - np.array(
        second_stage_primal_IP_costs)
    problem.log("Overall LP - IP diff", diff[diff != 0])

    if np.sum(diff) < 1e-4:
        return total_loss  # we are done, no gap.

    for i in range(original_problem.sample_size):
        problem.log(
            f"Scenario {i}'s LP - IP Gap:",
            second_stage_primal_LP_costs[i] - second_stage_primal_IP_costs[i])

    # TODO: compare the total risk from IP and LP

    return total_loss


def first_stage_model(env, problem, relax=False, bound_lp_binary=False):
    """Build the first stage model.

    This function returns:
        master, (theta, delta, rho, sigma), x, z
    where the (theta, delta, rho, sigma) are the variables of the first stage.
    The x and z are the decision variables of the first stage.


    Args:
        env (object): Gurobi environment.
        problem (object): a Problem object with all input data.

    Returns:
        master: the master Gurobi model
        theta (Gurobi variable): a value (number) for the second stage.
        delta (Gurobi tupledict): contains date information for the decision, 
            with the format `(int, date) -> int (number of demands can deploy)`.
        rho (Gurobi tupledict): contains row information for datacenter with
            format `(int, date) -> int (number of available rows)`.
        sigma (Gurobi tupledict): number of available supplies with the format
            `supplier -> int (number of available supplies)`.
        x (Gurobi tupledict): number of rows to build for the first stage, with
            the format `datacenter -> number of rows`.
        z (Gurobi tupledict): dock date information with the format
            `(demand, date, datacenter) -> boolean (decision for dock date)`
    """
    master = Model('master', env=env)

    supply_multiplier = problem.supply_multiplier

    # variable representing the average of the objective values of the
    # second stage problems
    theta = master.addVar(obj=1.0, name='theta')

    # stage variables
    if relax:
        if bound_lp_binary:
            z = master.addVars(problem.demand_costs.keys(),
                               obj=problem.demand_costs,
                               lb=0,
                               ub=1,
                               vtype=GRB.CONTINUOUS,
                               name='z')
            w = master.addVars(problem.shipping_costs.keys(),
                               obj=problem.shipping_costs,
                               lb=0,
                               ub=1,
                               vtype=GRB.CONTINUOUS,
                               name='w')
            x = master.addVars(problem.row_costs.keys(),
                               obj=problem.row_costs,
                               lb=0,
                               vtype=GRB.CONTINUOUS,
                               name='x')
            problem.log(
                "Using LP relaxation with [0, 1] bounds for binary variables.")
        else:
            z = master.addVars(problem.demand_costs.keys(),
                               obj=problem.demand_costs,
                               vtype=GRB.CONTINUOUS,
                               name='z')
            w = master.addVars(problem.shipping_costs.keys(),
                               obj=problem.shipping_costs,
                               vtype=GRB.CONTINUOUS,
                               name='w')
            x = master.addVars(problem.row_costs.keys(),
                               obj=problem.row_costs,
                               vtype=GRB.CONTINUOUS,
                               name='x')
            problem.log(
                "Using LP relaxation without bounds for binary variables.")
    else:
        z = master.addVars(problem.demand_costs.keys(),
                           obj=problem.demand_costs,
                           vtype=GRB.BINARY,
                           name='z')
        w = master.addVars(problem.shipping_costs.keys(),
                           obj=problem.shipping_costs,
                           vtype=GRB.BINARY,
                           name='w')
        x = master.addVars(problem.row_costs.keys(),
                           obj=problem.row_costs,
                           vtype=GRB.INTEGER,
                           name='x')

    # state variables
    delta = master.addVars(problem.capacity_indices,
                           problem.dates[problem.completion_date_index:],
                           name='delta')
    rho = master.addVars(problem.datacenters, name='rho')
    sigma = master.addVars(problem.supplies.keys(), name='sigma')

    # constraints
    master.addConstrs((z.sum(d, '*', '*') <= 1 for d in problem.demands),
                      'demand_assignment')
    master.addConstrs(
        (z.sum(d, '*', '*') == w.sum(d, '*') for d in problem.demands),
        'demand_supply')

    for n, (p, tcs) in enumerate(problem.throughput_capacity):
        for t, capacity in tcs:
            if t >= problem.completion_date:
                master.addConstr(
                    delta[n, t] == capacity - sum(
                        z.sum('*', r, t)
                        for r in p
                        if (r, t) in problem.potential_center_dates),
                    f'future_throughput{(n, t)}')
            elif any((r, t) in problem.potential_center_dates for r in p):
                master.addConstr(
                    capacity >= sum(
                        z.sum('*', r, t)
                        for r in p
                        if (r, t) in problem.potential_center_dates),
                    f'current_throughput{(n, t)}')

    master.addConstrs((problem.existing_rows[r] >= sum(
        z.sum('*', r, t)
        for t in problem.dates[:problem.completion_date_index])
                       for r in problem.datacenters), 'current_available_rows')
    master.addConstrs(
        (rho[r] == problem.existing_rows[r] + x[r] - z.sum('*', r, '*')
         for r in problem.datacenters), 'future_available_rows')
    master.addConstrs(
        (sigma[s]
         == int(supply_multiplier * problem.supplies[s]) - w.sum('*', s)
         for s in problem.supplies), 'supply')

    return master, (theta, delta, rho, sigma), x, w, z


def second_stage_model(env, problem):
    """Create the second stage model.

    Args:
        env (object): Gurobi environment.
        problem (object): a Problem object with all input data.

    Returns:
        sample_problems (list): a list of Gurobi model with each scenario. Each
            element (model) is associated with a scenario, containing variables
            and contraints for the second stage.
        sample_lambda (list): a list of dictionary, where each element contains
            "d", "free", "delta", "rho", and "sigma" values.
    """
    sample_problems = []
    sample_lambda = []

    for i in range(problem.sample_size):

        m_ = Model(f'sample_problem_{i}', env=env)
        sample_problems.append(m_)

        sample_lambda.append({})
        sample_lambda[i]['d'] = sample_problems[i].addVars(
            problem.sample_demands[i], lb=-inf, ub=0.0, name='d')
        sample_lambda[i]['free'] = sample_problems[i].addVars(
            problem.sample_demands[i], lb=-inf, name='free')
        sample_lambda[i]['delta'] = sample_problems[i].addVars(
            problem.capacity_indices,
            problem.dates[problem.completion_date_index:],
            lb=-inf,
            ub=0.0,
            name='delta')
        sample_lambda[i]['rho'] = sample_problems[i].addVars(
            problem.datacenters, lb=-inf, ub=0.0, name='rho')
        sample_lambda[i]['sigma'] = sample_problems[i].addVars(
            problem.supplies.keys(), lb=-inf, ub=0.0, name='sigma')

        sample_problems[i].addConstrs(
            (sample_lambda[i]['d'][d] + sample_lambda[i]['free'][d] +
             sum(sample_lambda[i]['delta'][n, t]
                 for n in problem.center_throughput[r]) +
             sample_lambda[i]['rho'][r] <= problem.sample_demand_costs[i][d, r,
                                                                          t]
             for d, r, t in problem.sample_demand_costs[i]), 'z dual')
        sample_problems[i].addConstrs(
            (-sample_lambda[i]['free'][d] + sample_lambda[i]['sigma'][s] <=
             problem.sample_shipping_costs[i][d, s]
             for d, s in problem.sample_shipping_costs[i]), 'w dual')

    return sample_problems, sample_lambda


def second_stage_model_primal(env,
                              problem,
                              first_stage,
                              relaxation=False,
                              debug=False):
    _, (_, delta, rho, sigma), x, w, z = first_stage

    if debug:
        for i in range(problem.sample_size):
            print(
                "Delta Unique Values:",
                np.unique([
                    delta[n, t].x
                    for n, (p, tcs) in enumerate(problem.throughput_capacity)
                    for t, capacity in tcs
                    if any((r, t) in problem.sample_potential_center_dates[i]
                           for r in p)
                ]))
        print("Rho Unique Values",
              np.unique([rho[r].x for r in problem.datacenters]))
        print("Sigma Unique Values",
              np.unique([sigma[s].x for s in problem.supplies]))

    variable_type = GRB.CONTINUOUS if relaxation else GRB.BINARY

    sample_problems = []
    sample_problems_costs = []

    sample_z = []
    sample_w = []

    all_decisions = []
    for i in range(problem.sample_size):

        m_ = Model(f'sample_problem_primal{i}', env=env)
        sample_problems.append(m_)

        sample_demand_costs = {
            key: np.float128(cost)
            for key, cost in problem.sample_demand_costs[i].items()
        }
        sample_shipping_costs = {
            key: np.float128(cost)
            for key, cost in problem.sample_shipping_costs[i].items()
        }

        # stage variables in the second stage
        sample_z.append(sample_problems[i].addVars(sample_demand_costs.keys(),
                                                   lb=0.0,
                                                   ub=1.0,
                                                   vtype=variable_type,
                                                   obj=sample_demand_costs,
                                                   name='z'))
        sample_w.append(sample_problems[i].addVars(sample_shipping_costs.keys(),
                                                   lb=0.0,
                                                   ub=1.0,
                                                   vtype=variable_type,
                                                   obj=sample_shipping_costs,
                                                   name='w'))

        sample_problems[i].addConstrs((sample_z[i].sum(d, '*', '*') <= 1
                                       for d in problem.sample_demands[i]),
                                      'demand_assignment')
        sample_problems[i].addConstrs(
            (sample_z[i].sum(d, '*', '*') == sample_w[i].sum(d, '*')
             for d in problem.sample_demands[i]), 'demand_supply')

        for n, (p, tcs) in enumerate(problem.throughput_capacity):
            for t, capacity in tcs:
                if any((r, t) in problem.sample_potential_center_dates[i]
                       for r in p):
                    sample_problems[i].addConstr(
                        sum(sample_z[i].sum('*', r, t) for r in p if (
                            r, t) in problem.sample_potential_center_dates[i])
                        <= min(1, delta[n, t].x), f'throughput{(n, t)}')

        sample_problems[i].addConstrs((sample_z[i].sum('*', r, '*') <= rho[r].x
                                       for r in problem.datacenters),
                                      'available_PBRs')
        sample_problems[i].addConstrs(
            (sample_w[i].sum('*', s) <= min(1, min(1, sigma[s].x))
             for s in problem.supplies), 'supply')

        sample_problems[i].optimize()

        sample_problems_costs.append(
            sample_problems[i].getObjective().getValue() +
            problem.sample_undock_costs_sum[i])

        z_vals = []
        for z_key in sample_demand_costs.keys():
            z_vals.append(sample_z[i][z_key].X)
            if debug and z_vals[-1] != 0:
                print(
                    f'{z_key} = {sample_z[i][z_key].X}. Costs: {sample_demand_costs[z_key]}'
                )

        w_vals = []
        for w_key in sample_shipping_costs.keys():
            w_vals.append(sample_w[i][w_key].X)
            if debug and w_vals[-1] != 0:
                print(
                    f'{w_key} = {sample_w[i][w_key].X}. Cost: {sample_shipping_costs[w_key]}'
                )

        # print(relaxation, "z", np.unique(z_vals))
        # print(relaxation, "w", np.unique(w_vals))
        # print(relaxation, "x", np.unique(sample_problems[i].x))

        all_decisions += z_vals
        all_decisions += w_vals

    # Check z and w values, are they close to 0 or 1.
    if relaxation:
        beyond_rounding_err = 0
        for val in all_decisions:
            if val > 0.99 or val < 0.01:
                continue
            beyond_rounding_err += 1
            problem.log("Rounding issue in LP: NOT WITHIN .01 THRESHOLD:", val)

        problem.log(f"Beyond Round Error: {beyond_rounding_err}")

    return sample_problems_costs, all_decisions


def level_model(env, problem):
    """Create a proposed hybrid level model for the problem.

    The returned variables are similar to the ones in the first_stage_model,
    but the returned values here are associated with the level method.

    Args:
        env (object): Gurobi environment.
        problem (object): a Problem object with all input data.

    Returns:
       level_model, (level_f, level_theta, level_z, level_w, level_x,
                         level_delta, level_rho, level_sigma)
    """
    level_model = Model('level', env=env)

    level_f = level_model.addVar(obj=1.0, name='level_f')
    level_theta = level_model.addVar(obj=1.0, name='level_theta')

    level_z = level_model.addVars(problem.demand_costs.keys(), name='level_z')
    level_w = level_model.addVars(problem.shipping_costs.keys(), name='level_w')
    level_x = level_model.addVars(problem.row_costs.keys(), name='level_x')

    level_delta = level_model.addVars(
        problem.capacity_indices,
        problem.dates[problem.completion_date_index:],
        name='level_delta')
    level_rho = level_model.addVars(problem.datacenters, name='level_rho')
    level_sigma = level_model.addVars(problem.supplies.keys(),
                                      name='level_sigma')

    # add constraints
    level_model.addConstrs(
        (level_z.sum(d, '*', '*') <= 1 for d in problem.demands),
        'demand_assignment')
    level_model.addConstrs((level_z.sum(d, '*', '*') == level_w.sum(d, '*')
                            for d in problem.demands), 'demand_supply')

    for n, (p, tcs) in enumerate(problem.throughput_capacity):
        for t, capacity in tcs:
            if t >= problem.completion_date:
                level_model.addConstr(
                    level_delta[n, t] == capacity - sum(
                        level_z.sum('*', r, t)
                        for r in p
                        if (r, t) in problem.potential_center_dates),
                    f'future_throughput{(n, t)}')
            elif any((r, t) in problem.potential_center_dates for r in p):
                level_model.addConstr(
                    capacity >= sum(
                        level_z.sum('*', r, t)
                        for r in p
                        if (r, t) in problem.potential_center_dates),
                    f'current_throughput{(n, t)}')

    level_model.addConstrs((problem.existing_rows[r] >= sum(
        level_z.sum('*', r, t)
        for t in problem.dates[:problem.completion_date_index])
                            for r in problem.datacenters),
                           'current_available_rows')
    level_model.addConstrs(
        (level_rho[r]
         == problem.existing_rows[r] + level_x[r] - level_z.sum('*', r, '*')
         for r in problem.datacenters), 'future_available_rows')
    level_model.addConstrs(
        (level_sigma[s]
         == int(problem.supply_multiplier * problem.supplies[s]) -
         level_w.sum('*', s) for s in problem.supplies), 'supply')

    level_model.addConstr(
        level_f >= problem.sum_undock +
        sum(problem.demand_costs[d, r, t] * level_z[d, r, t]
            for d, r, t in problem.demand_costs) +
        sum(problem.shipping_costs[d, s] * level_w[d, s]
            for d, s in problem.shipping_costs) +
        sum(problem.row_costs[r] * level_x[r] for r in problem.row_costs),
        'first_stage_objective')

    return level_model, (level_f, level_theta, level_z, level_w, level_x,
                         level_delta, level_rho, level_sigma)


def single_stage_model(first_stage, problem):
    """Create a single-stage model for the problem.

    The returned values are similar to the outputs from the `first_stage_model`.

    Args:
        first_stage (tuple): the first-stage model returned by the 
            `first_stage_model` function.
        problem (object): a Problem object with all input data.

    Returns:
        master, rho, sample_z, x, z
    """
    master, (_, delta, rho, sigma), x, w, z = first_stage

    sample_z = []
    sample_w = []
    for i in range(problem.sample_size):

        scaled_sample_demand_costs = {
            key: cost / problem.sample_size
            for key, cost in problem.sample_demand_costs[i].items()
        }
        scaled_sample_shipping_costs = {
            key: cost / problem.sample_size
            for key, cost in problem.sample_shipping_costs[i].items()
        }

        # stage variables in the second stage
        sample_z.append(
            master.addVars(scaled_sample_demand_costs.keys(),
                           vtype=GRB.BINARY,
                           obj=scaled_sample_demand_costs,
                           name='z'))
        sample_w.append(
            master.addVars(scaled_sample_shipping_costs.keys(),
                           vtype=GRB.BINARY,
                           obj=scaled_sample_shipping_costs,
                           name='w'))

        master.addConstrs((sample_z[i].sum(d, '*', '*') <= 1
                           for d in problem.sample_demands[i]),
                          'demand_assignment')
        master.addConstrs(
            (sample_z[i].sum(d, '*', '*') == sample_w[i].sum(d, '*')
             for d in problem.sample_demands[i]), 'demand_supply')

        for n, (p, tcs) in enumerate(problem.throughput_capacity):
            for t, capacity in tcs:
                if any((r, t) in problem.sample_potential_center_dates[i]
                       for r in p):
                    try:
                        master.addConstr(
                        sum(sample_z[i].sum('*', r, t) for r in p if (
                            r, t) in problem.sample_potential_center_dates[i])
                        <= delta[n, t], f'throughput{(n, t)}')
                    except Exception as e:
                        print(e)
                        pdb.set_trace()

        master.addConstrs((sample_z[i].sum('*', r, '*') <= rho[r]
                           for r in problem.datacenters), 'available_PBRs')
        master.addConstrs(
            (sample_w[i].sum('*', s) <= sigma[s] for s in problem.supplies),
            'supply')

    return master, rho, sample_z, x, z


def positives(array):
    return array[array > 0]


def key_groups(items):
    groups = defaultdict(set)
    for key, value in items:
        groups[key].add(value)
    return groups.values()


def demand_centers(demand_costs):
    return key_groups((d, r) for d, r, _ in demand_costs)


def throughput_centers_compatible(throughput_capacity, demand_costs):
    return np.array(
        [[sum(1
              for r in p
              if r in centers)
          for p, _ in throughput_capacity]
         for centers in demand_centers(demand_costs)])


def throughput_centers_count(throughput_capacity):
    return np.array([len(p) for p, _ in throughput_capacity])


def subrestriction_violation_stats(throughput_capacity, demand_costs):
    compatible = throughput_centers_compatible(throughput_capacity,
                                               demand_costs)
    count = throughput_centers_count(throughput_capacity)
    any_compatible = compatible > 0
    partial_compatible = any_compatible & (compatible < count)
    total_combinations = compatible.size
    compatible_combinations = any_compatible.sum()
    violating_combinations = partial_compatible.sum()

    total_violation_fraction = violating_combinations / total_combinations
    compatible_violation_fraction = violating_combinations / compatible_combinations

    capacity_compatible_demands = any_compatible.sum(0)
    capacity_violating_demands = partial_compatible.sum(0)
    demand_compatible_capacities = any_compatible.sum(1)
    demand_violating_capacities = partial_compatible.sum(1)

    total_demands, total_capacities = compatible.shape
    total_compatible_capacities = (capacity_compatible_demands > 0).sum()
    total_compatible_demands = (demand_compatible_capacities > 0).sum()
    assert total_compatible_demands == total_demands

    capacity_total_violation_fractions = capacity_violating_demands / total_demands
    capacity_compatible_violation_fractions = capacity_violating_demands / capacity_compatible_demands
    demand_total_violation_fractions = demand_violating_capacities / total_capacities
    demand_compatible_violation_fractions = demand_violating_capacities / demand_compatible_capacities

    capacity_total_violating_fractions = positives(
        capacity_total_violation_fractions)
    capacity_compatible_violating_fractions = positives(
        capacity_compatible_violation_fractions)
    demand_total_violating_fractions = positives(
        demand_total_violation_fractions)
    demand_compatible_violating_fractions = positives(
        demand_compatible_violation_fractions)

    capacity_total_violating_fraction = len(
        capacity_total_violating_fractions) / total_capacities
    capacity_compatible_violating_fraction = len(
        capacity_compatible_violating_fractions) / total_compatible_capacities
    demand_total_violating_fraction = len(
        demand_total_violating_fractions) / total_demands
    demand_compatible_violating_fraction = len(
        demand_compatible_violating_fractions) / total_compatible_demands
    assert demand_compatible_violating_fraction == demand_total_violating_fraction

    return (
        total_violation_fraction,
        compatible_violation_fraction,
        total_demands,
        total_capacities,
        total_compatible_capacities,
        total_compatible_demands,
        capacity_total_violating_fraction,
        capacity_compatible_violating_fraction,
        demand_total_violating_fraction,
        demand_compatible_violating_fraction,
        capacity_total_violation_fractions.mean(),
        capacity_total_violating_fractions.mean(),
        # capacity_compatible_violation_fractions.mean(),  # NaN
        capacity_compatible_violating_fractions.mean(),
        demand_total_violation_fractions.mean(),
        demand_total_violating_fractions.mean(),
        demand_compatible_violation_fractions.mean(),
        demand_compatible_violating_fractions.mean(),
    )


def samples_subrestriction_violation_stats(throughput_capacity,
                                           sample_demand_costs):
    return pd.DataFrame(
        (subrestriction_violation_stats(throughput_capacity, demand_costs)
         for demand_costs in sample_demand_costs),
        columns=[
            'total violation fraction',
            'compatible violation fraction',
            'total demands',
            'total capacities',
            'total compatible capacities',
            'total compatible demands',
            'capacity total violating fraction',
            'capacity compatible violating fraction',
            'demand total violating fraction',
            'demand compatible violating fraction',
            'capacity total violation fraction mean',
            'capacity total violating fraction mean',
            # 'capacity compatible violation fraction mean',  # NaN
            'capacity compatible violating fraction mean',
            'demand total violation fraction mean',
            'demand total violating fraction mean',
            'demand compatible violation fraction mean',
            'demand compatible violating fraction mean',
        ])
