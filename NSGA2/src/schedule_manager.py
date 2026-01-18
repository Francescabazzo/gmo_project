class ScheduleManager:
    """
    Manager class for handling and storing the original schedule information.
    Used to compare new schedules against the original baseline.
    """

    def __init__(self, jobshop):
        """
        Initialize the ScheduleManager with a JobShop instance.

        Args:
            jobshop: JobShop environment instance containing the original schedule
        """
        self.jobshop = jobshop
        self.original_schedule = self._capture_original_schedule()

    def _capture_original_schedule(self):
        """
        Captures the original schedule information from the JobShop environment.
        Stores job completion times and machine assignments for later comparison.

        Returns:
            dict: Dictionary containing:
                - 'job_completion_times': Mapping from job_id to completion time
                - 'machine_assignments': Mapping from operation_id to machine_id
        """
        original = {
            'job_completion_times': {},
            'machine_assignments': {}
        }

        # Capture job completion times from the last operation of each job
        for job in self.jobshop.jobs:
            if job.operations and job.operations[-1].scheduling_information:
                original['job_completion_times'][job.job_id] = job.operations[-1].scheduled_end_time

        # Capture machine assignments for all operations
        for op in self.jobshop.operations:
            if op.scheduling_information:
                original['machine_assignments'][op.operation_id] = op.scheduled_machine

        return original


    def get_original_schedule(self):
        """
        Returns the captured original schedule information.

        Returns:
            dict: Original schedule containing job completion times and machine assignments
        """
        return self.original_schedule