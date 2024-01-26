import time
from math import floor, inf

from gurobipy import GRB

BETA_DEFAULT = 0.1

beta = BETA_DEFAULT


def set_threshold(thresh=None, problem=None):
    global beta
    if thresh is None:
        beta = BETA_DEFAULT
    else:
        beta = thresh

    if problem is not None:
        problem.log("Threshold in CVAR is set to: beta =", beta)


def level_converge(first_stage, second_stage, level, problem):
    # cvar parameters
    kappa = floor(beta * problem.sample_size)
    prob = 1 - kappa / (problem.sample_size * beta)

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
        # update subproblems and optimize
        if time.time() - problem.start_time >= problem.timeout:
            problem.log(
                f"Timeout! {time.time() - problem.start_time} sec elapsed "
                f"with {k} iterations.")
            break

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

        # rank scenarios by objective values
        order = sorted([(i, sample_problems[i].getObjective().getValue() +
                         problem.sample_undock_costs_sum[i])
                        for i in range(problem.sample_size)],
                       key=lambda x: x[1],
                       reverse=True)

        # compute the upper bound
        cur_f = problem.sum_undock + sum(
            problem.demand_costs[d, r, t] * level_z[d, r, t].x
            for d, r, t in problem.demand_costs) + sum(
                problem.shipping_costs[d, s] * level_w[d, s].x
                for d, s in problem.shipping_costs) + sum(
                    problem.row_costs[r] * level_x[r].x
                    for r in problem.row_costs)
        cur_theta = 0
        for _, obj in order[:kappa]:
            cur_theta += obj / (problem.sample_size * beta)
        if problem.sample_size * beta - kappa > 0.0001:
            cur_theta += order[kappa][1] * prob

        ub = min(ub, cur_f + cur_theta)

        # get optimal solutions of the subproblems and refine the approximation
        #  in the first stage problem
        coeff_const = 0
        for i, _ in order[:kappa]:
            coeff_const += (sum(sample_lambda[i]['d'][d].x
                                for d in sample_lambda[i]['d']) +
                            problem.sample_undock_costs_sum[i]) / (
                                problem.sample_size * beta)
        if problem.sample_size * beta - kappa > 0.0001:
            i0 = order[kappa][0]
            coeff_const += (sum(sample_lambda[i0]['d'][d].x
                                for d in sample_lambda[i0]['d']) +
                            problem.sample_undock_costs_sum[i0]) * prob

        for n in problem.capacity_indices:
            for t in problem.dates[problem.completion_date_index:]:
                key = n, t
                coeff_delta[key] = 0
                for i, _ in order[:kappa]:
                    coeff_delta[key] += sample_lambda[i]['delta'][key].x / (
                        problem.sample_size * beta)

                if problem.sample_size * beta - kappa > 0.0001:
                    coeff_delta[key] += sample_lambda[
                        order[kappa][0]]['delta'][key].x * prob

        for r in problem.datacenters:
            coeff_rho[r] = 0
            for i, _ in order[:kappa]:
                coeff_rho[r] += sample_lambda[i]['rho'][r].x / (
                    problem.sample_size * beta)

            if problem.sample_size * beta - kappa > 0.0001:
                coeff_rho[r] += sample_lambda[order[kappa]
                                              [0]]['rho'][r].x * prob

        for s in problem.supplies:
            coeff_sigma[s] = 0
            for i, _ in order[:kappa]:
                coeff_sigma[s] += sample_lambda[i]['sigma'][s].x / (
                    problem.sample_size * beta)

            if problem.sample_size * beta - kappa > 0.0001:
                coeff_sigma[s] += sample_lambda[order[kappa]
                                                [0]]['sigma'][s].x * prob

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
        #  the LP relaxation
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
    # cvar parameters
    kappa = floor(beta * problem.sample_size)
    prob = 1 - kappa / (problem.sample_size * beta)
    start_time = time.time()

    master, (theta, delta, rho, sigma), x, w, z = first_stage
    sample_problems, sample_lambda = second_stage

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
        problem.log("> Iteration " + str(k) + " : " +
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

        # rank scenarios by objective values
        order = sorted([(i, sample_problems[i].getObjective().getValue() +
                         problem.sample_undock_costs_sum[i])
                        for i in range(problem.sample_size)],
                       key=lambda x: x[1],
                       reverse=True)

        # get optimal solutions of the subproblems and refine the approximation
        #  in the first stage problem
        coeff_const = 0
        for i, _ in order[:kappa]:
            coeff_const += (sum(sample_lambda[i]['d'][d].x
                                for d in sample_lambda[i]['d']) +
                            problem.sample_undock_costs_sum[i]) / (
                                problem.sample_size * beta)
        if problem.sample_size * beta - kappa > 0.0001:
            i0 = order[kappa][0]
            coeff_const += (sum(sample_lambda[i0]['d'][d].x
                                for d in sample_lambda[i0]['d']) +
                            problem.sample_undock_costs_sum[i0]) * prob

        coeff_delta = {}
        for n in problem.capacity_indices:
            for t in problem.dates[problem.completion_date_index:]:
                key = n, t
                coeff_delta[key] = 0
                for i, _ in order[:kappa]:
                    coeff_delta[key] += sample_lambda[i]['delta'][key].x / (
                        problem.sample_size * beta)

                if problem.sample_size * beta - kappa > 0.0001:
                    coeff_delta[key] += sample_lambda[
                        order[kappa][0]]['delta'][key].x * prob

        coeff_rho = {}
        for r in problem.datacenters:
            coeff_rho[r] = 0
            for i, _ in order[:kappa]:
                coeff_rho[r] += sample_lambda[i]['rho'][r].x / (
                    problem.sample_size * beta)

            if problem.sample_size * beta - kappa > 0.0001:
                coeff_rho[r] += sample_lambda[order[kappa]
                                              [0]]['rho'][r].x * prob

        coeff_sigma = {}
        for s in problem.supplies:
            coeff_sigma[s] = 0
            for i, _ in order[:kappa]:
                coeff_sigma[s] += sample_lambda[i]['sigma'][s].x / (
                    problem.sample_size * beta)

            if problem.sample_size * beta - kappa > 0.0001:
                coeff_sigma[s] += sample_lambda[order[kappa]
                                                [0]]['sigma'][s].x * prob

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

        # compute and log the upper bound
        # We need the `order` variable to compute the upper bound.
        cur_f = problem.sum_undock + sum(
            problem.demand_costs[d, r, t] * z[d, r, t].x
            for d, r, t in problem.demand_costs) + sum(
                problem.shipping_costs[d, s] * w[d, s].x
                for d, s in problem.shipping_costs) + sum(
                    problem.row_costs[r] * x[r].x for r in problem.row_costs)
        cur_theta = 0
        for _, obj in order[:kappa]:
            cur_theta += obj / (problem.sample_size * beta)
        if problem.sample_size * beta - kappa > 0.0001:
            cur_theta += order[kappa][1] * prob

        ub = cur_f + cur_theta
        problem.log("Iteration Upper Bound", ub)

    for i in range(problem.sample_size):
        problem.log(sample_problems[i].getObjective().getValue() +
                    problem.sample_undock_costs_sum[i])

    return master, (theta, delta, rho, sigma)
