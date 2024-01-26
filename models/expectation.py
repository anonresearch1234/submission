import pdb
import pickle
import time
from math import inf

from gurobipy import GRB

supply_multiplier = 6

import random

from .solvers import first_stage_model, single_stage_model

EPS = 0.0001

def set_threshold(thresh=None, problem=None):
    if problem is not None:
        problem.log("set_threshold in expectation.py does nothing.")


def single_stage_solve(env, problem, dump_loc:str=None, 
                       is_multi_stage_experiment:bool=False):
    problem.log("Running single_stage_solve")
    first_stage = first_stage_model(env, problem)

    _, (theta, delta, rho, sigma), x, w, z = first_stage

    master = single_stage_model(first_stage, problem)

    if type(master) is tuple:
        master = master[0]  # ignore other other debugging parameters

    master.setParam(
        'TimeLimit', max(0,
                         problem.timeout - (time.time() - problem.start_time)))
    master.optimize()

    if master.status != GRB.OPTIMAL:
        print("No feasible solution!!!")
        master.computeIIS()
        constrs = [c.ConstrName for c in master.getConstrs() if c.IISConstr]
        print(constrs)

        pdb.set_trace()

        # print the infeasible constraints
        for c in master.getConstrs():
            if c.getAttr('slack') < -EPS:
                print(c.getAttr('ConstrName'), c.getAttr('slack'))

    delta_s = {k: v.X for k, v in delta.items()}
    rho_s = {k: v.X for k, v in rho.items()}
    sigma_s = {k: v.X for k, v in sigma.items()}
    x_s = {k: v.X for k, v in x.items()}
    w_s = {k: v.X for k, v in w.items()}
    z_s = {k: v.X for k, v in z.items()}

    if dump_loc:
        pickle.dump([min(problem.dates), 
                     problem.dates[problem.completion_date_index],
                     delta_s, rho_s, sigma_s, x_s, w_s, z_s, 
                     master.getObjective().getValue()],
                     open(dump_loc, "wb"))

    obj = master.getObjective().getValue() + problem.sum_undock + sum(
        problem.sample_undock_costs_sum[i]
        for i in range(problem.sample_size)) / problem.sample_size

    # pdb.set_trace()

    if not is_multi_stage_experiment:
        return obj
    
    cost_in_T1_period = 0
    demand_undock_cost = pickle.load(open("multi-stage/demand_undock_cost.pkl", "rb"))
    for d, dc, t in z.keys():
        if z[d, dc, t].x > 1 - EPS and t < problem.dates[problem.completion_date_index]:
            # Demand Cost/Reward
            if problem.demand_costs[d, dc, t] + demand_undock_cost[d] <= 0:
                pdb.set_trace()
            cost_in_T1_period += problem.demand_costs[d, dc, t] + demand_undock_cost[d]
            # Shipping Cost
            for s in problem.supplies:
                if (d, s) in w.keys() and w[d, s].x > 1 - EPS:
                    cost_in_T1_period += problem.shipping_costs[d, s]
                    if problem.shipping_costs[d, s] < 0:
                        pdb.set_trace()
    # Row building cost
    for dc in problem.datacenters:
        cost_in_T1_period += problem.row_costs[dc] * round(x[dc].x)
        if problem.row_costs[dc] * round(x[dc].x) < 0:
            pdb.set_trace()


    return obj, cost_in_T1_period, (delta_s, rho_s, sigma_s)


def level_converge(first_stage, second_stage, level, problem):
    master, (theta, delta, rho, sigma), _, _, _ = first_stage

    sample_problems, sample_lambda = second_stage
    level_model, (level_f, level_theta, level_z, level_w, level_x, level_delta,
                  level_rho, level_sigma) = level

    level_model.optimize()
    level_constraint = level_model.addConstr(level_f + level_theta <= 0,
                                             'level_constraint')
    level_vars = level_model.getVars()

    coeff_delta = {}
    coeff_rho = {}
    coeff_sigma = {}

    ub = inf
    lb = -inf
    epsilon = 1
    alpha = 0.5
    for k in range(1, 10001):
        if time.time() - problem.start_time >= problem.timeout:
            problem.log(
                f"Timeout! {time.time() - problem.start_time} sec elapsed "
                f"with {k} iterations.")
            break

        # update subproblems and optimize
        for i in range(problem.sample_size):
            sample_problems[i].setObjective(
                sample_lambda[i]['d'].sum('*') +
                sum(level_delta[n, t].x * sample_lambda[i]['delta'][n, t]
                    for n in problem.capacity_indices
                    for t in problem.dates[problem.completion_date_index:]) +
                sum(level_rho[r].x * sample_lambda[i]['rho'][r]
                    for r in problem.datacenters) +
                sum(level_sigma[s].x * sample_lambda[i]['sigma'][s]
                    for s in problem.supplies), GRB.MAXIMIZE)

            sample_problems[i].setParam(
                'TimeLimit',
                max(0, problem.timeout - (time.time() - problem.start_time)))
            sample_problems[i].optimize()

        # compute the upper bound
        cur_f = (
            problem.sum_undock +
            sum(problem.demand_costs[d, r, t] * level_z[d, r, t].x
                for d, r, t in problem.demand_costs) +
            sum(problem.shipping_costs[d, s] * level_w[d, s].x
                for d, s in problem.shipping_costs) +
            sum(problem.row_costs[r] * level_x[r].x for r in problem.row_costs))
        cur_theta = (
            sum(sample_problems[i].getObjective().getValue()
                for i in range(problem.sample_size)) +
            sum(problem.sample_undock_costs_sum[i]
                for i in range(problem.sample_size))) / problem.sample_size
        ub = min(ub, cur_f + cur_theta)

        # get optimal solutions of the subproblems
        coeff_const = (
            sum(sample_lambda[i]['d'][d].x
                for i in range(problem.sample_size)
                for d in problem.sample_demands[i]) +
            sum(problem.sample_undock_costs_sum[i]
                for i in range(problem.sample_size))) / problem.sample_size

        for n in problem.capacity_indices:
            for t in problem.dates[problem.completion_date_index:]:
                key = n, t
                coeff_delta[key] = sum(
                    sample_lambda[i]['delta'][key].x
                    for i in range(problem.sample_size)) / problem.sample_size

        for r in problem.datacenters:
            coeff_rho[r] = sum(
                sample_lambda[i]['rho'][r].x
                for i in range(problem.sample_size)) / problem.sample_size

        for s in problem.supplies:
            coeff_sigma[s] = sum(
                sample_lambda[i]['sigma'][s].x
                for i in range(problem.sample_size)) / problem.sample_size

        # update the level model
        level_model.addConstr(
            level_theta >= coeff_const +
            sum(coeff_delta[n, t] * level_delta[n, t]
                for n in problem.capacity_indices
                for t in problem.dates[problem.completion_date_index:]) +
            sum(coeff_rho[r] * level_rho[r] for r in problem.datacenters) +
            sum(coeff_sigma[s] * level_sigma[s] for s in problem.supplies),
            f'level_cut_{k}')

        # update the first stage problem and compute the lower bound
        master.addConstr(
            theta >= coeff_const +
            sum(coeff_delta[n, t] * delta[n, t]
                for n in problem.capacity_indices
                for t in problem.dates[problem.completion_date_index:]) +
            sum(coeff_rho[r] * rho[r] for r in problem.datacenters) +
            sum(coeff_sigma[s] * sigma[s] for s in problem.supplies),
            f'level_cut_{k}')

        master.setParam(
            'TimeLimit',
            max(0, problem.timeout - (time.time() - problem.start_time)))
        master.optimize()
        lb = master.getObjective().getValue() + problem.sum_undock

        # break if the current trial point is an epsilon-optimal solution
        #  of the LP relaxation
        if ub - lb < epsilon:
            break

        # find the next trial point
        l = alpha * lb + (1 - alpha) * ub
        level_constraint.setAttr('rhs', l)
        level_model.setObjective(
            sum(var * var - (2 * var.x) * var for var in level_vars[2:]))

        level_model.setParam(
            'TimeLimit',
            max(0, problem.timeout - (time.time() - problem.start_time)))
        level_model.optimize()

        try:
            problem.log("Iteration " + str(k) + " : " + str(
                level_model.getObjective().getValue() + problem.sum_undock +
                sum(problem.sample_undock_costs_sum[i]
                    for i in range(problem.sample_size)) / problem.sample_size))
        except Exception as e:
            print(e)
            pdb.set_trace()


def benders_converge(first_stage, second_stage, problem):
    master, (theta, delta, rho, sigma), x, w, z = first_stage
    sample_problems, sample_lambda = second_stage
    start_time = time.time()

    previous_delta = {}
    for n in problem.capacity_indices:
        for t in problem.dates[problem.completion_date_index:]:
            previous_delta[n, t] = -1

    previous_rho = {}
    for r in problem.datacenters:
        previous_rho[r] = -1

    previous_sigma = {}
    for s in problem.supplies:
        previous_sigma[s] = -1

    for k in range(1, 10001):
        if time.time() - problem.start_time >= problem.timeout:
            problem.log(
                f"Timeout! {time.time() - problem.start_time} sec elapsed "
                f"with {k} iterations.")
            break

        master.setParam(
            'TimeLimit',
            max(0, problem.timeout - (time.time() - problem.start_time)))
        master.optimize()
        problem.log("-- Iteration " + str(k) + " : " +
                    str(master.getObjective().getValue() + problem.sum_undock))

        epsilon = 0.1

        # record solution and terminate if the solution does not change
        terminate = True
        for n in problem.capacity_indices:
            for t in problem.dates[problem.completion_date_index:]:
                key = n, t
                if abs(previous_delta[key] - delta[key].x) > epsilon:
                    previous_delta[key] = delta[key].x
                    terminate = False

        for r in problem.datacenters:
            if abs(previous_rho[r] - rho[r].x) > epsilon:
                previous_rho[r] = rho[r].x
                terminate = False

        for s in problem.supplies:
            if abs(previous_sigma[s] - sigma[s].x) > epsilon:
                previous_sigma[s] = sigma[s].x
                terminate = False

        if terminate:
            break

        problem.log("Optimizing subproblems ...  %s seconds " %
                    (time.time() - start_time))

        # update subproblems and optimize
        for i in range(problem.sample_size):
            sample_problems[i].setObjective(
                sample_lambda[i]['d'].sum('*') +
                sum(delta[n, t].x * sample_lambda[i]['delta'][n, t]
                    for n in problem.capacity_indices
                    for t in problem.dates[problem.completion_date_index:]) +
                sum(rho[r].x * sample_lambda[i]['rho'][r]
                    for r in problem.datacenters) +
                sum(sigma[s].x * sample_lambda[i]['sigma'][s]
                    for s in problem.supplies), GRB.MAXIMIZE)

            sample_problems[i].setParam(
                'TimeLimit',
                max(0, problem.timeout - (time.time() - problem.start_time)))
            sample_problems[i].optimize()

        # get optimal solutions of the subproblems and refine the approximation
        #  in the first stage problem
        coeff_const = (
            sum(sample_lambda[i]['d'][d].x
                for i in range(problem.sample_size)
                for d in problem.sample_demands[i]) +
            sum(problem.sample_undock_costs_sum[i]
                for i in range(problem.sample_size))) / problem.sample_size

        coeff_delta = {}
        for n in problem.capacity_indices:
            for t in problem.dates[problem.completion_date_index:]:
                key = n, t
                coeff_delta[key] = sum(
                    sample_lambda[i]['delta'][key].x
                    for i in range(problem.sample_size)) / problem.sample_size

        coeff_rho = {}
        for r in problem.datacenters:
            coeff_rho[r] = sum(
                sample_lambda[i]['rho'][r].x
                for i in range(problem.sample_size)) / problem.sample_size

        coeff_sigma = {}
        for s in problem.supplies:
            coeff_sigma[s] = sum(
                sample_lambda[i]['sigma'][s].x
                for i in range(problem.sample_size)) / problem.sample_size

        problem.log("Optimizing master ...  %s seconds " %
                    (time.time() - start_time))

        master.addConstr(
            theta >= coeff_const +
            sum(coeff_delta[n, t] * delta[n, t]
                for n in problem.capacity_indices
                for t in problem.dates[problem.completion_date_index:]) +
            sum(coeff_rho[r] * rho[r] for r in problem.datacenters) +
            sum(coeff_sigma[s] * sigma[s] for s in problem.supplies),
            f'benders_cut_{k}')

        # Print and log the risk measure for the second stage
        # Print the Upper Bound
        # problem.log(risk_measure(sample_problems))
        # problem.log("undock cost:", problem.sum_undock) # TODO: move out of the for loop
        problem.log("theta.x:", theta.x)
        sample_problem_curr_mean = sum(
            sample_problems[i].getObjective().getValue() +
            problem.sample_undock_costs_sum[i]
            for i in range(problem.sample_size)) / problem.sample_size
        problem.log("sample_problem_curr_mean:", sample_problem_curr_mean)
        problem.log(
            "(manual) Iteration Upper Bound",
            master.getObjective().getValue() + problem.sum_undock - theta.x +
            sample_problem_curr_mean)

        # compute and log the upper bound
        cur_f = (problem.sum_undock +
                 sum(problem.demand_costs[d, r, t] * z[d, r, t].x
                     for d, r, t in problem.demand_costs) +
                 sum(problem.shipping_costs[d, s] * w[d, s].x
                     for d, s in problem.shipping_costs) +
                 sum(problem.row_costs[r] * x[r].x for r in problem.row_costs))
        cur_theta = (
            sum(sample_problems[i].getObjective().getValue()
                for i in range(problem.sample_size)) +
            sum(problem.sample_undock_costs_sum[i]
                for i in range(problem.sample_size))) / problem.sample_size
        ub = cur_f + cur_theta
        problem.log("Iteration Upper Bound", ub)

    for i in range(problem.sample_size):
        problem.log(sample_problems[i].getObjective().getValue() +
                    problem.sample_undock_costs_sum[i])

    return master, (theta, delta, rho, sigma)
