import sys
from pathlib import Path
import subprocess

def assert_or_install_dependencies(packages, trainer_name):
    exec_path = Path(sys.executable)
    site_package_path = exec_path / '..' / ".." / 'Lib' / 'nova-server-site-packages'
    site_package_path.mkdir(parents=True, exist_ok=True)

    path = site_package_path / trainer_name
    for i,pkg in enumerate(packages):
        params = []
        pk = pkg.split(' ')
        if len(pk) > 1:
            params.extend(pk[1:])
        name = pk[0].split('==')

        params.append("--target={}".format(path))

        package_path = site_package_path / trainer_name

        adjusted_name = str(name[0]).replace('-', '_')
        dirs = [1 if x.name.startswith(adjusted_name) else 0 for x in package_path.iterdir() if x.is_dir()]

        if sum(dirs) > 0:
            print(f'skip installation of {package_path}. Package already installed')
            pass
        else:
            install_package(pk[0], params)

    sys.path.insert(0, str(path.resolve()))

def install_package(pkg, params):
    call = [sys.executable, "-m", "pip", "install", pkg, *params]
    print(call)
    return subprocess.check_call(call)