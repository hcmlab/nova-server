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
    from waitress import serve
    app = create_server()

    def thread_test():
        response = app.test_client().post(
            '/test',
            data={
                'file': 'a',
            },
            content_type='multipart/form-data'
        )
        data = json.loads(response.get_data(as_text=True))
        print(data)
        return data

    def status_test(job_id):
        response = app.test_client().post(
            '/jobstatus',
            data=job_id,
            content_type='multipart/form-data'
        )
        data = json.loads(response.get_data(as_text=True))
        print(data)
        return data

    id_1 = thread_test()
    id_2 = thread_test()
    id_3 = thread_test()
    id_4 = thread_test()
    id_5 = thread_test()

    status_1 = status_test(id_1)
    status_2 = status_test(id_2)
    status_3 = status_test(id_3)
    status_4 = status_test(id_4)
    status_5 = status_test(id_5)

    serve(app, host="127.0.0.1", port=8080)



if __name__ == '__main__':
    #example_train()
    example_thread()
