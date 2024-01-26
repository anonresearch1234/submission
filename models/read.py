# used within eval
import datetime
from bisect import bisect_left
from collections import Counter, defaultdict
from contextlib import suppress
from os import environ, scandir
from pathlib import Path
from pickle import dump, load

from more_itertools import only, unique_everseen

path = environ.get('SAMPLE_ROOT') or 'data'
samples_subpath = 'samples'
data_filename = 'data.pkl'

base_input_names = (
    'throughput_capacity',
    'dates',
    'undock_costs',
    'dock_costs',
    'supplies',
    'shipping_costs',
    'demand_centers',
    'completion_date',
    'existing_rows',
    'row_costs',
)

sample_base_input_names = (
    'undock_costs',
    'dock_costs',
    'shipping_costs',
)


def base_inputs(path=path, names=base_input_names):
    return validate_base_inputs(
        [eval(Path(path, f'{i}.py').read_text()) for i in names])


def sample_base_inputs(path=path, names=sample_base_input_names):
    samples_path = Path(path, samples_subpath)
    return [[
        eval(Path(sample, f'{i}.py').read_text())
        for sample in scandir(samples_path)
    ]
            for i in names]


def validate_base_inputs(inputs):
    throughput_capacity, *_ = inputs
    # may have multiple capacity restrictions for same or nested facility sets
    assert is_hierarchical(facilities for facilities, _ in throughput_capacity)
    return inputs


def is_hierarchical(subsets):
    container = {}
    for subset in sorted(subsets, key=len, reverse=True):
        try:
            only(unique_everseen(container.get(element) for element in subset))
        except ValueError:
            return False
        for element in subset:
            container[element] = subset
    return True


def build_inputs(path):
    return derived_inputs(base_inputs(path), sample_base_inputs(path))


def inputs(path=path, build=build_inputs):
    return cache_data(build, path, data_filename)


def cache_data(build, path, filename):
    data_path = Path(path, filename)
    with suppress(OSError), open(data_path, 'rb') as store:
        return load(store)
    data = build(path)
    with suppress(OSError), open(data_path, 'wb') as store:
        dump(data, store)
    return data


def derived_inputs(base_inputs, sample_base_inputs):
    (
        throughput_capacity,
        dates,
        undock_costs,
        dock_costs,
        supplies,
        shipping_costs,
        demand_centers,
        completion_date,
        existing_rows,
        row_costs,
    ) = base_inputs
    (
        sample_undock_costs,
        sample_dock_costs,
        sample_shipping_costs,
    ) = sample_base_inputs

    capacity_indices = range(len(throughput_capacity))
    datacenters = list(existing_rows.keys())
    demands = list(undock_costs.keys())
    sample_demands = [list(sample.keys()) for sample in sample_undock_costs]

    # construct the correspondence between throughput and datacenters
    center_throughput = defaultdict(list)
    for n, (p, _) in enumerate(throughput_capacity):
        for r in p:
            center_throughput[r].append(n)

    # construct demand costs and potential dock dates
    sum_undock = sum(undock_costs.values())
    demand_costs = {}
    potential_center_dates = set()
    for d in demands:
        for r in demand_centers[d]:
            for t, c in dock_costs[d]:
                demand_costs[d, r, t] = c - undock_costs[d]
                potential_center_dates.add((r, t))

    # find the index of the completion date in the sorted dates
    completion_date_index = bisect_left(dates, completion_date)

    sample_size = len(sample_undock_costs)

    # construct sample demand costs and potential dock dates
    sum_sample_undock = sum(
        sum(sample_undock_costs[i].values()) for i in range(sample_size))
    sample_undock_costs_sum = [
        sum(sample_undock_costs[i].values()) for i in range(sample_size)
    ]
    sample_demand_costs = []
    sample_potential_center_dates = []
    for i in range(sample_size):
        sample_demand_costs.append({})
        sample_potential_center_dates.append(set())
        for d in sample_demands[i]:
            for r in demand_centers[d]:
                for t, c in sample_dock_costs[i][d]:
                    sample_demand_costs[i][d, r,
                                           t] = c - sample_undock_costs[i][d]
                    sample_potential_center_dates[i].add((r, t))

    return (
        demand_costs,
        shipping_costs,
        row_costs,
        sum_undock,
        demands,
        supplies,
        existing_rows,
        datacenters,
        throughput_capacity,
        capacity_indices,
        center_throughput,
        dates,
        potential_center_dates,
        completion_date,
        completion_date_index,
        sample_size,
        sample_demand_costs,
        sample_shipping_costs,
        sum_sample_undock,
        sample_undock_costs_sum,
        sample_demands,
        sample_potential_center_dates,
    )
