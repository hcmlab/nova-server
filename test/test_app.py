from unittest.mock import patch

import os
import unittest
from pathlib import Path

from nova_server.app import setup_env_variable, build_server, main


class TestSetupEnvVariable(unittest.TestCase):
    def test_valid_input(self):
        key = "TEST_KEY"
        value = "test_value"
        default = "default_value"
        with patch.dict(os.environ, {key: default}, clear=True):
            setup_env_variable(key, value, default)
            self.assertEqual(os.environ[key], value)
            self.assertTrue(Path(value).exists())

    def test_no_value_input(self):
        key = "TEST_KEY"
        value = None
        default = "default_value"
        with patch.dict(os.environ, clear=True):
            setup_env_variable(key, value, default)
            self.assertEqual(os.environ[key], default)
            self.assertTrue(Path(default).exists())

    def test_no_env_var_or_value_input(self):
        key = "TEST_KEY"
        value = None
        default = "default_value"
        with patch.dict(os.environ, clear=True):
            setup_env_variable(key, value, default)
            self.assertEqual(os.environ[key], default)
            self.assertTrue(Path(default).exists())

    def test_invalid_input(self):
        key = 1234
        value = "test_value"
        default = "default_value"
        with self.assertRaises(TypeError):
            setup_env_variable(key, value, default)

    def test_build_server(self):
        app = build_server()
        self.assertIsNotNone(app)
        self.assertEqual(len(app.blueprints), 7)
