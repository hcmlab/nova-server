"""Backend for using virtual environments as a backend for nova server

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    18.8.2023

"""

import os
import shutil
import subprocess

import sys
from threading import Thread
from subprocess import Popen, PIPE
from pathlib import Path
from nova_server.utils import venv_utils as vu
from dotenv import load_dotenv
from logging import Logger


class VenvHandler:
    """
    Handles the creation and management of a virtual environment and running scripts within it.

    Args:
        module_dir (Path, optional): The path to the nova server module directory for the environtment.
        logger (Logger, optional): The logger instance for logging.
        log_verbose (bool, optional): If True, log verbose output.

    Attributes:
        venv_dir (Path): The path to the virtual environment.
        log_verbose (bool): If True, log verbose output.
        module_dir (Path): The path to the module directory.
        logger (Logger): The logger instance.

    Example:
        >>> import logging
        >>> load_dotenv("../.env")
        >>> log_dir = Path(os.getenv('NOVA_LOG_DIR', '.'))
        >>> log_file = log_dir / 'test.log'
        >>> logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.DEBUG)
        >>> logger = logging.getLogger('test_logger')
        >>> module_path = Path(os.getenv("NOVA_CML_DIR")) / "test"
        >>> venv_handler = VenvHandler(module_path, logger=logger, log_verbose=True)
        >>> venv_handler.run_python_script_from_file(
        ...     module_path / "test.py",
        ...     script_args=["pos_1", "pos_2"],
        ...     script_kwargs={"-k1": "k1", "--keyword_three": "k3"},
        ... )
    """

    def _reader(self, stream, context=None):
        """
        Reads and logs output from a stream.

        Args:
            stream: The stream to read.
            context (str, optional): The context of the stream (stdout or stderr).
        """
        while True:
            s = stream.readline()
            if not s:
                break
            if self.logger is None:
                if not self.log_verbose:
                    sys.stderr.write(".")
                else:
                    sys.stderr.write(s)
            else:
                if context == "stderr":
                    self.logger.error(s)
                else:
                    self.logger.info(s)
            sys.stderr.flush()
        stream.close()

    def _run_cmd(self, cmd: str, wait: bool = True) -> int:
        """
        Executes a command in a subprocess and logs the output.

        Args:
           cmd (str): The command to execute.
           wait (bool, optional): If True, wait for the command to complete.

        Returns:
            int: Return code of the executed command
        """
        p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, universal_newlines=True)
        t1 = Thread(target=self._reader, args=(p.stdout, "stdout"))
        t1.start()
        t2 = Thread(target=self._reader, args=(p.stderr, "stderr"))
        t2.start()
        if wait:
            p.wait()
            t1.join()
            t2.join()

        return p.returncode

    def _get_or_create_venv(self):
        """
        Gets or creates a virtual environment and returns its path.

        Returns:
            Path: The path to the virtual environment.
        """
        venv_dir = vu.venv_dir_from_mod(self.module_dir)
        if not venv_dir.is_dir():
            try:
                run_cmd = f"{sys.executable} -m venv {venv_dir}"
                self._run_cmd(run_cmd)
            except Exception as e:
                shutil.rmtree(venv_dir)
                raise e
        return venv_dir

    def _upgrade_pip(self):
        """
        Upgrades the `pip` package within the virtual environment.
        """
        run_cmd = vu.get_module_run_cmd(
            self.venv_dir, "pip", args=["install"], kwargs={"--upgrade": "pip"}
        )
        self._run_cmd(run_cmd)

    def _install_requirements(self):
        """
        Installs requirements from a `requirements.txt` file within the virtual environment.
        """
        req_txt = self.module_dir / "requirements.txt"
        if not req_txt.is_file():
            return
        else:
            # pip
            self._upgrade_pip()
            # TODO: replace with custom import utils for better parsing
            # requirements.txt
            run_cmd = vu.get_module_run_cmd(
                self.venv_dir,
                "pip",
                args=["install"],
                kwargs={"-r": str(req_txt.resolve())},
            )
            self._run_cmd(run_cmd)

    def __init__(
        self, module_dir: Path = None, logger: Logger = None, log_verbose: bool = False
    ):
        """
        Initializes the VenvHandler instance.

        Args:
            module_dir (Path, optional): The path to the module directory.
            logger (Logger, optional): The logger instance for logging.
            log_verbose (bool, optional): If True, log verbose output.
        """
        self.venv_dir = None
        self.log_verbose = log_verbose
        self.module_dir = module_dir
        self.logger = logger if logger is not None else Logger(__name__)
        if module_dir is not None:
            self.init_venv()

    def init_venv(self):
        """
        Initializes the virtual environment and installs requirements.
        """
        self.venv_dir = self._get_or_create_venv()
        self._install_requirements()

    def run_python_script_from_file(
        self, script_fp: Path, script_args: list = None, script_kwargs: dict = None
    ):
        """
        Runs a Python script within the virtual environment.

        Args:
            script_fp (Path): The path to the script to run.
            script_args (list, optional): List of positional arguments to pass to the script.
            script_kwargs (dict, optional): Dictionary of keyword arguments to pass to the script.

        Raises:
            ValueError: If the virtual environment has not been initialized. Call `init_venv()` first.
            subprocess.CalledProcessError: If the executed command exits with a return value other than 0
        """
        if self.venv_dir is None:
            raise ValueError(
                "Virtual environment has not been initialized. Call <init_venv()> first."
            )
        run_cmd = vu.get_python_script_run_cmd(
            self.venv_dir, script_fp, script_args, script_kwargs
        )
        return_code = self._run_cmd(run_cmd)
        if not return_code == 0:
            raise subprocess.CalledProcessError(returncode=return_code, cmd=run_cmd)

    def run_shell_script(self, script: str, script_args: list = None, script_kwargs: dict = None):
        """
        Runs a command in the respective os shell with an activated virtual environment


        Args:
            script (Path): The path to the script to run.
            script_args (list, optional): List of positional arguments to pass to the script.
            script_kwargs (dict, optional): Dictionary of keyword arguments to pass to the script.

        Raises:
            ValueError: If the virtual environment has not been initialized. Call `init_venv()` first.
            subprocess.CalledProcessError: If the executed command exits with a return value other than 0
        """
        if self.venv_dir is None:
            raise ValueError(
                "Virtual environment has not been initialized. Call <init_venv()> first."
            )
        run_cmd = vu.get_shell_script_run_cmd(
            self.venv_dir, script, script_args, script_kwargs
        )
        return_code = self._run_cmd(run_cmd)
        if not return_code == 0:
            raise subprocess.CalledProcessError(returncode=return_code, cmd=run_cmd)


if __name__ == "__main__":
    import logging

    load_dotenv("../.env")

    log_dir = Path(os.getenv("NOVA_LOG_DIR", "."))
    log_file = log_dir / "test.log"
    logging.basicConfig(filename=log_file, encoding="utf-8", level=logging.DEBUG)
    logger = logging.getLogger("test_logger")

    module_path = Path(os.getenv("NOVA_CML_DIR")) / "test"
    venv_handler = VenvHandler(module_path, logger=logger, log_verbose=True)
    venv_handler.run_python_script_from_file(
        module_path / "test.py",
        script_args=["pos_1", "pos_2"],
        script_kwargs={"-k1": "k1", "--keyword_three": "k3"},
    )
