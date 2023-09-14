""" Class definition for th execution handler. The execution handler combines uses data from the request form to execute
the correct command using the specified backends.

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    06.09.2023
"""

import os
import re
from logging import Logger
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from pathlib import PureWindowsPath
from nova_server.utils import env


class Action(Enum):
    PREDICT = "nu-predict"
    EXTRACT = "nu-extract"
    TRAIN = "nu-train"


class Backend(Enum):
    VENV = 0


class ExecutionHandler(ABC):
    @property
    def script_arguments(self):
        return self._script_arguments

    @script_arguments.setter
    def script_arguments(self, value):
        # convert dictionary keys from camelCase to snake_case
        self._script_arguments = {
            "--" + re.sub(r"(?<!^)(?=[A-Z])", "_", k).lower(): v
            for k, v in value.items()
        }

    def __init__(
        self, request_form: dict, backend: Backend = Backend.VENV, logger: Logger = None
    ):
        self.backend = backend
        self.script_arguments = request_form
        self.logger = logger

    def _nova_server_env_to_arg(self):
        env_vars = [env.NOVA_SERVER_CML_DIR, env.NOVA_SERVER_DATA_DIR, env.NOVA_SERVER_LOG_DIR, env.NOVA_SERVER_CACHE_DIR, env.NOVA_SERVER_TMP_DIR]
        arg_vars = {}
        prefix = 'NOVA_SERVER_'
        for var in env_vars:
            k = "--" + var[len(prefix):].lower()
            v = os.getenv(var)
            arg_vars[k] = v
        return arg_vars

    def run(self):

        # run with selected backend
        if self.backend == Backend.VENV:
            from nova_server.backend import virtual_environment as backend

            # setup virtual environment
            cml_dir = os.getenv(env.NOVA_SERVER_CML_DIR)
            if cml_dir is None:
                raise ValueError(f"NOVA_CML_DIR not et")

            module_dir = Path(cml_dir) / self.module_name
            if not module_dir.is_dir():
                raise NotADirectoryError(
                    f"NOVA_CML_DIR {module_dir} is not a valid directory"
                )
            backend_handler = backend.VenvHandler(
                module_dir, logger=self.logger, log_verbose=True
            )

            # add dotenv variables to arguments for script
            self._script_arguments |= self._nova_server_env_to_arg()

            backend_handler.run_shell_script(
                self.run_script,
                script_kwargs=self._script_arguments,
            )

        else:
            raise ValueError(f"Unknown backend {self.backend}")

    @property
    @abstractmethod
    def run_script(self):
        pass

    @property
    @abstractmethod
    def module_name(self):
        pass


class NovaPredictHandler(ExecutionHandler):
    @property
    def module_name(self):
        tfp = self.script_arguments.get("--trainer_file_path")
        if tfp is None:
            raise ValueError("trainerFilePath not specified in request.")
        else:
            return PureWindowsPath(tfp).parent

    @property
    def run_script(self):
        return self.action.value

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action = Action.PREDICT

class NovaExtractHandler(ExecutionHandler):
    @property
    def module_name(self):
        cfp = self.script_arguments.get("--chain_file_path")
        if cfp is None:
            raise ValueError("chainFilePath not specified in request.")
        else:
            return PureWindowsPath(cfp).parent

    @property
    def run_script(self):
        return self.action.value

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action = Action.EXTRACT

class NovaTrainHandler(ExecutionHandler):
    @property
    def module_name(self):
        tfp = self.script_arguments.get("--trainer_file_path")
        if tfp is None:
            raise ValueError("trainerFilePath not specified in request.")
        else:
            return PureWindowsPath(tfp).parent

    @property
    def run_script(self):
        return self.action.value

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action = Action.TRAIN