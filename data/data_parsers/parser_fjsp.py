import re
from pathlib import Path

from scheduling_environment.job import Job
from scheduling_environment.jobShop import JobShop
from scheduling_environment.machine import Machine
from scheduling_environment.operation import Operation


def parse_fjsp(instance_file_name, from_absolute_path=False) ->JobShop:
    job_shop = JobShop()
    job_shop.set_instance_name(instance_file_name)

    if not from_absolute_path:
        base_path = Path(__file__).resolve().parents[2]
        data_path = base_path.joinpath('data' + instance_file_name)
    else:
        data_path = instance_file_name

    with open(data_path, "r") as data:
        total_jobs, total_machines, max_operations = re.findall(
            '\\S+', data.readline())
        number_total_jobs, number_total_machines, number_max_operations = int(
            total_jobs), int(total_machines), int(float(max_operations))

        job_shop.set_nr_of_jobs(number_total_jobs)
        job_shop.set_nr_of_machines(number_total_machines)

        precedence_relations = {}
        job_id = 0
        operation_id = 0

        for key, line in enumerate(data):
            if key >= number_total_jobs:
                break
            # Split data with multiple spaces as separator
            parsed_line = re.findall('\\S+', line)

            # Current item of the parsed line
            i = 1
            job = Job(job_id)

            while i < len(parsed_line):
                # Total number of operation options for the operation
                operation_options = int(parsed_line[i])
                # Current activity
                operation = Operation(job, job_id, operation_id)

                for operation_options_id in range(operation_options):
                    machine_id = int(parsed_line[i + 1 + 2 *
                                                 operation_options_id]) - 1
                    duration = int(
                        parsed_line[i + 2 + 2 * operation_options_id])
                    operation.add_operation_option(machine_id, duration)
                job.add_operation(operation)
                job_shop.add_operation(operation)
                if i != 1:
                    precedence_relations[operation_id] = [
                        job_shop.get_operation(operation_id - 1)]

                i += 1 + 2 * operation_options
                operation_id += 1

            job_shop.add_job(job)
            job_id += 1

    # add also the operations without precedence operations to the precedence relations dictionary
    for operation in job_shop.operations:
        if operation.operation_id not in precedence_relations.keys():
            precedence_relations[operation.operation_id] = []
        operation.add_predecessors(
            precedence_relations[operation.operation_id])

    sequence_dependent_setup_times = [[[0 for r in range(len(job_shop.operations))] for t in range(len(job_shop.operations))] for
                                      m in range(number_total_machines)]

    # Precedence Relations & sequence dependent setup times
    job_shop.add_precedence_relations_operations(precedence_relations)
    job_shop.add_sequence_dependent_setup_times(sequence_dependent_setup_times)

    # Machines
    for id_machine in range(0, number_total_machines):
        job_shop.add_machine((Machine(id_machine)))

    return job_shop


# if __name__ == "__main__":
#     from scheduling_environment.jobShop import JobShop
#     jobShopEnv = JobShop()
#     jobShopEnv = parse_fjsp(jobShopEnv, '/fjsp/1_brandimarte/Mk01.fjs')
#     jobShopEnv.update_operations_available_for_scheduling()
#     print(jobShopEnv.operations_available_for_scheduling)