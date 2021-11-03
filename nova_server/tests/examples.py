from nova_server.nova_backend import create_server
from flask import json

def example_train():
    app = create_server()
    response = app.test_client().post(
        '/train',
        data={
            'file': 'a',
        },
        content_type='multipart/form-data'
    )
    data = json.loads(response.get_data(as_text=True))
    print(data)

def example_thread():
    app = create_server()

    def thread_test():
        response = app.test_client().post(
            '/thread',
            data={
                'file': 'a',
            },
            content_type='multipart/form-data'
        )
        data = json.loads(response.get_data(as_text=True))
        print(data)

    thread_test()

if __name__ == '__main__':
    example_train()
