import random
from pathlib import Path


def generate_fjsp_dataset(n_machines, n_jobs, min_op_job, max_op_job, min_time_op, max_time_op, seed=3):
    random.seed(seed)
    dataset_lines = []
    ops_per_job = [
        random.randint(min_op_job, max_op_job)
        for _ in range(n_jobs)
    ]
    max_operations = max(ops_per_job)

    header = f"{n_jobs} {n_machines} {max_operations}"
    dataset_lines.append(header)

    for job_idx in range(n_jobs):
        num_ops = ops_per_job[job_idx]
        tokens = [str(num_ops)]

        for op in range(num_ops):
            nr_options = random.randint(3, n_machines)
            tokens.append(str(nr_options))

            for _ in range(nr_options):
                machine = random.randint(1, n_machines)
                duration = random.randint(min_time_op, max_time_op)
                tokens.append(str(machine))
                tokens.append(str(duration))

        dataset_lines.append(" ".join(tokens))

    return "\n".join(dataset_lines)


def save_fjsp_file(content: str, file_path: Path):
    file_path = Path(file_path)

    if file_path.suffix != ".fjs":
        file_path = file_path.with_suffix(".fjs")

    with open(file_path, "w") as f:
        f.write(content)

