import os
import shutil
import tempfile
import unittest

from nova_server.app import setup_env_variable, build_server


class TestExtract(unittest.TestCase):
    def setUp(self):
        self.app = build_server()
        self.client = self.app.test_client()
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        # Set environment variables
        setup_env_variable('NOVA_CML_DIR', self.test_dir, 'cml')
        setup_env_variable('NOVA_DATA_DIR', self.test_dir, 'data')
        setup_env_variable('NOVA_CACHE_DIR', None, 'cache')
        setup_env_variable('NOVA_TMP_DIR', None, 'tmp')
        setup_env_variable('NOVA_LOG_DIR', None, 'log')
        # Create test chain file
        self.chain_file_name = "test_chain.cml"
        with open(os.path.join(self.test_dir, self.chain_file_name), "w") as f:
            f.write("test content")

    def tearDown(self):
        # Remove temporary directory after testing
        shutil.rmtree(self.test_dir)

    def test_extract_thread_post_request(self):
        with self.app.app_context():
            response = self.client.post("/extract", data={
                "chainFilePath": self.chain_file_name,
                "dataset": "test_dataset",
                "sessions": "test_session",
                "roles": "test_role",
                "username": "test_user",
                "password": "test_password"
            })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["success"], "true")

    def test_extract_thread_invalid_request(self):
        with self.app.app_context():
            response = self.client.get("/extract")
            self.assertEqual(response.status_code, 405)
