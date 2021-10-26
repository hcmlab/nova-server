from nova_server.nova_backend import app
from flask import json

def example_train():

    response = app.test_client().post(
        '/train',
        data={
            'file': 'a',
        },
        content_type='multipart/form-data'
    )
    data = json.loads(response.get_data(as_text=True))
    print(data)

if __name__ == '__main__':
    example_train()
