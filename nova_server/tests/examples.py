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

def example_jobqueque():
    from waitress import serve
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

    id_1 = add_job()
    id_2 = add_job()
    id_3 = add_job()
    id_4 = add_job()
    id_5 = add_job()

    status_1 = get_status(id_1)
    status_2 = get_status(id_2)
    status_3 = get_status(id_3)
    status_4 = get_status(id_4)
    status_5 = get_status(id_5)

    serve(app, host="127.0.0.1", port=8080)

if __name__ == '__main__':
    #example_train()
    example_jobqueque()
