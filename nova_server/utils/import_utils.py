import sys
from pathlib import Path
import subprocess

def assert_or_install_dependencies(packages):
    exec_path = Path(sys.executable)
    site_package_path = exec_path / '..' / ".." / 'Lib' / 'nova-server-site-packages'
    site_package_path.mkdir(parents=True, exist_ok=True)

    for i,pkg in enumerate(packages):
        params = []
        pk = pkg.split(' ')
        if len(pk) > 1:
            params.extend(pk[1:])
        name = pk[0].split('==')
        ver = 'latest'
        if len(name) == 2:
            ver = name[1]
        path = site_package_path / name[0] / ver

        print(path)
        params.append("--target={}".format(path))

        if path.exists():
            print(f'skip installation of {path}. Package already installed')
            pass
        else:
            install_package(pk[0], params)

        sys.path.insert(i, str(path.resolve()))

def install_package(pkg, params):
    call = [sys.executable, "-m", "pip", "install", pkg, *params]
    print(call)
    return subprocess.check_call(call)