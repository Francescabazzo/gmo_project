"""
Generic utilities to save the logging of the simulations.
They are independent on the module implemented. 
"""

import logging
import sys
import pickle
import os
from datetime import datetime
from pathlib import Path


def setup_logging_directory(base_dir, log_dir_name="experiment_logs"):
    """
    Creates the directory for the logs.
    
    Args:
        base_dir: "base" directory
        log_dir_name: name of the folder for the logs 
    
    Returns:
        Path: Logs' directory 
    """
    log_dir = os.path.join(base_dir, log_dir_name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def setup_experiment_logging(log_dir, exp_num, log_level=logging.INFO, 
                           log_to_console=True, log_filename=None):
    """
    Setup of the logging for a pecific experiment
    
    Args:
        log_dir: Logs' directory 
        exp_num: Number of the experiment 
        log_level: Logging level
        log_to_console: If True, shows the logs also on the console 
        log_filename: Name of the logs' file 
    
    Returns:
        logging.Logger: Logger 
    """
    if log_filename is None:
        log_filename = f"experiment_{exp_num}.txt"
    
    log_file = os.path.join(log_dir,f"EX{exp_num}", log_filename)
    
    logger = logging.getLogger(f'experiment_{exp_num}')
    logger.setLevel(log_level)
    
    # Remove existing handlers 
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Handler files 
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(log_level)
    
    # Handler stream for terminal 
    if log_to_console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    if log_to_console:
        stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    if log_to_console:
        logger.addHandler(stream_handler)
    
    return logger


def setup_gurobi_logging(log_dir, exp_num, log_to_console=True):
    """
    Prepares a dedicated logger for Gurobi output.

    Args:
        log_dir (Path): Directory where logs are saved
        exp_num (int): Experiment number
        log_to_console (bool): If True, print the path in console

    Returns:
        Path: Path to the Gurobi log file
    """
    # Construct the log file path
    log_file = os.path.join(log_dir,f"EX{exp_num}", f"gurobi_exp{exp_num}.log")

    # Create an empty file for Gurobi (Gurobi will write directly here)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if log_to_console:
        print(f"Gurobi log for experiment {exp_num} will be saved in: {log_file}")

    return log_file


def close_experiment_logging(logger):
    """Correctly closes the logging of the experiment."""
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def log_experiment_start(logger, exp_num, params=None):
    """Log at the beginning of the experiment."""
    logger.info(f"{'='*60}")
    logger.info(f"START EXPERIMENT {exp_num}")
    logger.info(f"{'='*60}")
    logger.info(f"Starting time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if params:
        logger.info("Parameters of the experiment:")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")


def log_experiment_end(logger, exp_num, success=True, message=None):
    """Log at the end of the experiment."""
    status = "COMPLETED" if success else "FAILED"
    logger.info(f"{'='*60}")
    logger.info(f"END OF THE EXPERIMENT {exp_num} - {status}")
    if message:
        logger.info(f"Message: {message}")
    logger.info(f"{'='*60}")


def load_environment(file_path, logger=None):
    """Load the environment from a pickle file."""
    try:
        with open(file_path, "rb") as file:
            environment = pickle.load(file)
        if logger:
            logger.info(f"Environment loaded from: {file_path}")
        return environment
    except Exception as e:
        if logger:
            logger.error(f"Error in loading the environment: {str(e)}")
        raise


def log_environment_info(logger, environment):
    """Log of the informations of the environment."""
    logger.info(f"Job number: {len(environment.jobs)}")
    logger.info(f"Machine number: {len(environment.machines)}")
    
    # Additional information, if available
    if hasattr(environment, 'unique_machine_ids'):
        unique_machines = environment.unique_machine_ids
        logger.info(f"Unique machines: {unique_machines}")
    
    if hasattr(environment, 'operations_count'):
        logger.info(f"Number of operations: {environment.operations_count}")