import os
import pickle
import random
import time
from math import inf
from statistics import mean

from gurobipy import GRB
from xlwt import Workbook

from .solvers import first_stage_model, second_stage_model, single_stage_model


def deterministic_solve(env, problem):
    """Solve the deterministic problem.

    Args:
        env (object): Gurobi environment.
        problem (Problem): an object of the Problem class.

    Returns:
        wb (Excel): pointer to the output Excel file.
    """
    master, rho, sample_z, x, z, wb = deterministic_model(
        second_stage_model(env, problem), env, problem)
    
    return wb


def deterministic_model(second_stage, env, problem):
    """Create a deterministic model.

    Save the detailed information to a Excel file, named by the problem.logname.

    Args:
        second_stage (tuple): second-stage information with (settings, lambdas).
        env (object): Gurobi environment.
        problem (Problem): an object of the Problem class.

    Returns:
         master, rho, sample_z, x, z, wb
    """
    wb = Workbook()
    sheet1 = wb.add_sheet(
        str(problem.supply_multiplier) + "xSup_" + str(problem.sample_size) +
        "scen")

    sample_problems, sample_lambda = second_stage

    # Set up workbook info
    sheet1.write(2, 0, 'Second stage objective values per scenario')
    sheet1.write(0, 2, 'Scenario that is fixed as the known future demand')
    for sample_index in range(problem.sample_size):
        sheet1.write(1, sample_index + 2, sample_index)
        sheet1.write(sample_index + 2, 1, sample_index)
    sheet1.write(problem.sample_size + 4, 0, 'Expectation across the scenarios')
    sheet1.write(problem.sample_size + 6, 0,
                 'First stage objective value (without the 2nd)')
    sheet1.write(problem.sample_size + 7, 0,
                 'Total master objective value (with the 2nd stage)')
    sheet1.write(problem.sample_size + 10, 0,
                 'Cost of first stage + scenario expectation')

    results = []

    # Fix a scenario and add the second stage variables and constraints
    for sample_index in range(problem.sample_size):
        problem.log("-------- Sample index = " + str(sample_index) + "--------")

        master, (_, delta, rho,
                 sigma), x, w, z = first_stage_model(env, problem)

        j = sample_index
        scaled_sample_demand_costs = {
            key: cost for key, cost in problem.sample_demand_costs[j].items()
        }
        scaled_sample_shipping_costs = {
            key: cost for key, cost in problem.sample_shipping_costs[j].items()
        }

        # stage variables in the second stage
        sample_z = master.addVars(scaled_sample_demand_costs.keys(),
                                  vtype=GRB.BINARY,
                                  obj=scaled_sample_demand_costs,
                                  name='z')
        sample_w = master.addVars(scaled_sample_shipping_costs.keys(),
                                  vtype=GRB.BINARY,
                                  obj=scaled_sample_shipping_costs,
                                  name='w')

        master.addConstrs(
            (sample_z.sum(d, '*', '*') <= 1 for d in problem.sample_demands[j]),
            'demand_assignment')
        master.addConstrs((sample_z.sum(d, '*', '*') == sample_w.sum(d, '*')
                           for d in problem.sample_demands[j]), 'demand_supply')

        for n, (p, tcs) in enumerate(problem.throughput_capacity):
            for t, capacity in tcs:
                if any((r, t) in problem.sample_potential_center_dates[j]
                       for r in p):
                    master.addConstr(
                        sum(
                            sample_z.sum('*', r, t)
                            for r in p
                            if (r,
                                t) in problem.sample_potential_center_dates[j])
                        <= delta[n, t], f'throughput{(n, t)}')

        master.addConstrs(
            (sample_z.sum('*', r, '*') <= rho[r] for r in problem.datacenters),
            'available_PBRs')
        master.addConstrs(
            (sample_w.sum('*', s) <= sigma[s] for s in problem.supplies),
            'supply')

        master.optimize()


        delta_s = {k: v.X for k, v in delta.items()}
        rho_s = {k: v.X for k, v in rho.items()}
        sigma_s = {k: v.X for k, v in sigma.items()}
        x_s = {k: v.X for k, v in x.items()}
        w_s = {k: v.X for k, v in w.items()}
        z_s = {k: v.X for k, v in z.items()}
        os.makedirs("det_out", exist_ok=True)
        pickle.dump([delta_s, rho_s, sigma_s, x_s, w_s, z_s, 
                     master.getObjective().getValue()],
                     open(f"det_out/det_{sample_index}.pickle", "wb"))
        
        ### Evaluation

        # Run all subproblems with these values of rho, sigma, delta
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

            sample_problems[i].optimize()

        for i in range(problem.sample_size):
            sheet1.write(i + 2, sample_index + 2,
                         (sample_problems[i].getObjective().getValue() +
                          problem.sample_undock_costs_sum[i]))
            problem.log(sample_problems[i].getObjective().getValue() +
                        problem.sample_undock_costs_sum[i])

        avg_scenarios = mean((sample_problems[i].getObjective().getValue() +
                              problem.sample_undock_costs_sum[i])
                             for i in range(problem.sample_size))
        sheet1.write(problem.sample_size + 4, sample_index + 2, avg_scenarios)
        sheet1.write(problem.sample_size + 6, sample_index + 2,
                     (master.getObjective().getValue() + problem.sum_undock -
                      sample_problems[j].getObjective().getValue()))
        sheet1.write(problem.sample_size + 7, sample_index + 2,
                     (master.getObjective().getValue() + problem.sum_undock +
                      problem.sample_undock_costs_sum[j]))
        expected_cost_fixing_scenario = (
            master.getObjective().getValue() + problem.sum_undock -
            sample_problems[j].getObjective().getValue() + avg_scenarios)
        sheet1.write(problem.sample_size + 10, sample_index + 2,
                     expected_cost_fixing_scenario)
        results.append(expected_cost_fixing_scenario)

        problem.log("Master:")
        problem.log(master.getObjective().getValue() + problem.sum_undock +
                    problem.sample_undock_costs_sum[j])
        master.dispose()

    sheet1.write(problem.sample_size + 13, 0, 'Expected result det')
    det_obj = mean(results)
    sheet1.write(problem.sample_size + 14, 0, det_obj)

    problem.log("Starting stochastic")
    master_stoch, _rho, _sample_z, _x, _z = single_stage_model(
        first_stage_model(env, problem), problem)
    master_stoch.optimize()
    stoch_obj = master_stoch.getObjective().getValue(
    ) + problem.sum_undock + sum(
        problem.sample_undock_costs_sum[i]
        for i in range(problem.sample_size)) / problem.sample_size
    sheet1.write(problem.sample_size + 13, 1, 'Expected result stoch')
    sheet1.write(problem.sample_size + 14, 1, stoch_obj)

    sheet1.write(problem.sample_size + 13, 2, 'Percentage improvement')
    sheet1.write(problem.sample_size + 14, 2, (det_obj - stoch_obj) / det_obj)

    return master, rho, sample_z, x, z, wb
