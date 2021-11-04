from nova_server.nova_backend import create_server
from flask import json
from waitress import serve


def example_train(data):
    app = create_server()
    response = app.test_client().post(
        '/train',
        data=data,
        content_type='multipart/form-data'
    )
    data = json.loads(response.get_data(as_text=True))
    print(data)
    serve(app, host="127.0.0.1", port=8080)


def example_jobqueque():
    app = create_server()

    def add_job():
        response = app.test_client().post(
            '/testjobqueque',
            data={},
            content_type='multipart/form-data'
        )
        data = json.loads(response.get_data(as_text=True))
        print(data)
        return data

    def get_status(job_id):
        response = app.test_client().post(
            '/jobstatus',
            data=job_id,
            content_type='multipart/form-data'
        )
        data = json.loads(response.get_data(as_text=True))
        print(data)
        return data

    ret = []
    for i in range(10):
        ret.append(add_job())

    import time
    time.sleep(1)

    for i in ret:
        get_status(i)

    serve(app, host="127.0.0.1", port=8080)

def example_predict_to_nova():
    app = create_server()
    response = app.test_client().post(
        '/predict',
        data={
            'file': 'a',
        },
        content_type='multipart/form-data'
    )
    data = json.loads(response.get_data(as_text=True))
    print(data)

if __name__ == '__main__':

    # add your own request payload here
    from nova_server.tests.test_payloads import test_train as data

    example_train(data)
    #example_predict_to_nova()
    #example_jobqueque()
