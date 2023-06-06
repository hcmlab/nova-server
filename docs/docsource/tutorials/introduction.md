# Introduction

## Installation

### Prerequesites

Before starting to install NOVA-Server you need to install Python and FFMPEG.
While other Python versions may work as well the module is only tested for the following versions:

* 3.9.x

You can download the current version of python for your system [here](https://www.python.org/downloads/).

Download the current version off FFMPEG binaries from [here](https://github.com/BtbN/FFmpeg-Builds/releases) for your system and make sure to extract them to a place that is in your system path.

### Setup

You can then install NOVA-Server using pip like this:

```pip install hcai-nova-server```

### Start the server

To start NOVA-Server you can just open a Terminal and type 

```nova-server```

NOVA-Server takes the following optional arguments as input:

```--host```: ```0.0.0.0``` : The IP for the Server to listen

```--port``` : ```8080``` : The port for the Server to be bound to

```--template_folder``` : ```templates``` : The Flask template folder to 

```--cml_dir``` : ```cml``` : The cooperative machine learning directory for Nova 

```--data_dir``` : ```data``` : Directory where the NOVA data resides

```--cache_dir``` : ```cache``` : Cache directory for Models and other downloadable content 

```--tmp_dir``` : ```tmp``` : Directory to store data for temporary usage  

```--log_dir``` : ```log``` : Directory to store logfiles.


All directory variables can be also set as system wide environment variables using the following variable names:

```NOVA_CML_DIR```, ```NOVA_DATA_DIR```, ```NOVA_CACHE_DIR```, ```NOVA_TMP_DIR```, ```NOVA_LOG_DIR```

If a directory is set via a system variable but also passed as command line argument when starting nova-server, the command line argument will overwrite the system variable.

If the server started successfully your console output should look like this: 

```
Starting nova-backend server...
HOST: 0.0.0.0
PORT: 8080
NOVA_CML_DIR : cml
NOVA_DATA_DIR : data
NOVA_CACHE_DIR : cache
NOVA_TMP_DIR : tmp
NOVA_LOG_DIR : log
...done
```