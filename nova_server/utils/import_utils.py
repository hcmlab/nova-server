import sys
from pathlib import Path
import subprocess
import site

def assert_or_install_dependencies(packages, trainer_name):
    #exec_path = Path(sys.executable)
    site_package_path = (Path(site.getsitepackages()[0]) / '..' / 'nova-server-site-packages' / trainer_name).resolve()
    site_package_path.mkdir(parents=True, exist_ok=True)

    for i,pkg in enumerate(packages):
        params = []
        # split on space while also removing double spaces
        pk = [x for x in pkg.split(' ') if x]
        if len(pk) > 1:
            params.extend(pk[1:])
        name = pk[0].split('==')

        params.append("--target={}".format(site_package_path))
        adjusted_name = str(name[0]).replace('-', '_')
        dirs = [1 if x.name.startswith(adjusted_name) else 0 for x in site_package_path.iterdir() if x.is_dir()]

        if sum(dirs) > 0:
            print(f'skip installation of {site_package_path}. Package already installed')
            pass
        else:
            install_package(pk[0], params)

    sys.path.insert(0, str(site_package_path.resolve()))

def install_package(pkg, params):
    call = [sys.executable, "-m", "pip", "install", pkg, *params]
    print(call)
    return subprocess.check_call(call)