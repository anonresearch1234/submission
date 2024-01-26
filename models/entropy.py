import time
from math import inf

from gmpy2 import exp, log
from gurobipy import GRB

TAU_DEFAULT = 1e-6
tau = TAU_DEFAULT


def set_threshold(thresh=None, problem=None):
    global tau
    if thresh is None:
        tau = TAU_DEFAULT
    else:
        tau = thresh

    if problem is not None:
        problem.log("Threshold in Entropy is set to: tau =", tau)


def level_converge(first_stage, second_stage, level, problem):
    master, (theta, delta, rho, sigma), _, _, _ = first_stage
    sample_problems, sample_lambda = second_stage
    level_model, (level_f, level_theta, level_z, level_w, level_x, level_delta,
                  level_rho, level_sigma) = level

    level_model.setParam(
        'TimeLimit', max(0,
                         problem.timeout - (time.time() - problem.start_time)))
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

        nums = [
            exp(tau * (sample_problems[i].getObjective().getValue() +
                       problem.sample_undock_costs_sum[i]))
            for i in range(problem.sample_size)
        ]
        denom = sum(nums)

        # compute the upper bound
        cur_f = problem.sum_undock + sum(
            problem.demand_costs[d, r, t] * level_z[d, r, t].x
            for d, r, t in problem.demand_costs) + sum(
                problem.shipping_costs[d, s] * level_w[d, s].x
                for d, s in problem.shipping_costs) + sum(
                    problem.row_costs[r] * level_x[r].x
                    for r in problem.row_costs)
        cur_theta = float(log(denom / problem.sample_size)) / tau
        ub = min(ub, cur_f + cur_theta)

        # get optimal solutions of the subproblems and refine the approximation
        #  in the first stage problem
        factors = [float(num / denom) for num in nums]

        coeff_const = cur_theta

        for n in problem.capacity_indices:
            for t in problem.dates[problem.completion_date_index:]:
                key = n, t
                coeff_delta[key] = sum(sample_lambda[i]['delta'][key].x *
                                       factors[i]
                                       for i in range(problem.sample_size))
                coeff_const -= coeff_delta[key] * level_delta[key].x

        for r in problem.datacenters:
            coeff_rho[r] = sum(sample_lambda[i]['rho'][r].x * factors[i]
                               for i in range(problem.sample_size))
            coeff_const -= coeff_rho[r] * level_rho[r].x

        for s in problem.supplies:
            coeff_sigma[s] = sum(sample_lambda[i]['sigma'][s].x * factors[i]
                                 for i in range(problem.sample_size))
            coeff_const -= coeff_sigma[s] * level_sigma[s].x

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

        # break if the current trial point is an epsilon-optimal solution of
        # the LP relaxation
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

        problem.log("Iteration " + str(k) + " : " +
                    str(level_model.getObjective().getValue() +
                        problem.sum_undock +
                        problem.sum_sample_undock / problem.sample_size))


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

    coeff_delta = {}
    coeff_rho = {}
    coeff_sigma = {}
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
        problem.log("> Iteration " + str(k) + " : " +
                    str(master.getObjective().getValue() + problem.sum_undock))

        epsilon = 0.1

        # record solution and terminate if the solution does not change
        terminate = True
        for n in problem.capacity_indices:
            for t in problem.dates[problem.completion_date_index:]:
                key = n, t
                if abs(previous_delta[key] - delta[key].x) > epsilon:
                    #if previous_delta[key] != delta[key].x:
                    previous_delta[key] = delta[key].x
                    terminate = False

        for r in problem.datacenters:
            if abs(previous_rho[r] - rho[r].x) > epsilon:
                #if previous_rho[r] != rho[r].x:
                previous_rho[r] = rho[r].x
                terminate = False

        for s in problem.supplies:
            if abs(previous_sigma[s] - sigma[s].x) > epsilon:
                #if previous_sigma[s] != sigma[s].x:
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

        nums = [
            exp(tau * (sample_problems[i].getObjective().getValue() +
                       problem.sample_undock_costs_sum[i]))
            for i in range(problem.sample_size)
        ]
        denom = sum(nums)

        # get optimal solutions of the subproblems and refine the approximation
        #  in the first stage problem
        factors = [float(num / denom) for num in nums]

        coeff_const = float(log(denom / problem.sample_size)) / tau

        for n in problem.capacity_indices:
            for t in problem.dates[problem.completion_date_index:]:
                key = n, t
                coeff_delta[key] = sum(sample_lambda[i]['delta'][key].x *
                                       factors[i]
                                       for i in range(problem.sample_size))
                coeff_const -= coeff_delta[key] * delta[key].x

        for r in problem.datacenters:
            coeff_rho[r] = sum(sample_lambda[i]['rho'][r].x * factors[i]
                               for i in range(problem.sample_size))
            coeff_const -= coeff_rho[r] * rho[r].x

        for s in problem.supplies:
            coeff_sigma[s] = sum(sample_lambda[i]['sigma'][s].x * factors[i]
                                 for i in range(problem.sample_size))
            coeff_const -= coeff_sigma[s] * sigma[s].x

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

        # Log and Compute the upper bound
        cur_f = problem.sum_undock + sum(
            problem.demand_costs[d, r, t] * z[d, r, t].x
            for d, r, t in problem.demand_costs) + sum(
                problem.shipping_costs[d, s] * w[d, s].x
                for d, s in problem.shipping_costs) + sum(
                    problem.row_costs[r] * x[r].x for r in problem.row_costs)
        cur_theta = float(log(denom / problem.sample_size)) / tau
        ub = cur_f + cur_theta
        problem.log("Iteration Upper Bound", ub)

    for i in range(problem.sample_size):
        problem.log(sample_problems[i].getObjective().getValue() +
                    problem.sample_undock_costs_sum[i])

    return master, (theta, delta, rho, sigma)
