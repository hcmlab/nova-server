import unittest

from nova_server.app import build_server, setup_env_variable
from nova_server.route.predict import predict_thread


class TestPredict(unittest.TestCase):
    def setUp(self):
        setup_env_variable('NOVA_CML_DIR', None, 'cml')
        setup_env_variable('NOVA_DATA_DIR', None, 'data')
        setup_env_variable('NOVA_CACHE_DIR', None, 'cache')
        setup_env_variable('NOVA_TMP_DIR', None, 'tmp')
        setup_env_variable('NOVA_LOG_DIR', None, 'log')
        self.app = build_server()
        self.client = self.app.test_client()
        self.request_form = {
            "trainerFilePath": "test_trainer_file_path",
            "sessions": "session1;session2",
            "roles": "role1;role2",
            "password": "test_password"
        }

    def test_predict_thread_post(self):
        with self.app.test_request_context('/predict', method='POST', data=self.request_form):
            response = predict_thread()
            self.assertEqual(response.status_code, 200)
            self.assertIn("success", response.json)
            self.assertEqual(response.json["success"], "true")

    def test_predict_thread_invalid_method(self):
        with self.app.test_request_context('/predict', method='GET', data=self.request_form):
            response = predict_thread()
            self.assertIsNone(response)
