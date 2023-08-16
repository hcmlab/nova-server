import sys
from pathlib import Path
import subprocess
import site


def assert_or_install_dependencies(packages, trainer_name):
    site_package_path = (Path(site.getsitepackages()[0]) / '..' / 'nova-server-site-packages' / trainer_name).resolve()
    site_package_path.mkdir(parents=True, exist_ok=True)

    for i, pkg in enumerate(packages):
        params = []
        # split on space while also removing double spaces
        pk = [x for x in pkg.split(' ') if x]
        if len(pk) > 1:
            params.extend(pk[1:])
        # maybe add all VCS https://pip.pypa.io/en/stable/topics/vcs-support/
        if 'git+' in pk[0]:
            if '#egg=' in pk[0]:
                name = pk[0].split('#egg=')[-1]
            else:
                name = pk[0].split('/')[-1].split('.git')[0]
        else:
            # maybe support all specifiers https://peps.python.org/pep-0440/#version-specifiers
            name = pk[0].split('==')[0]

        # support systems without nvidia/cuda installed
        try:
            subprocess.check_output('nvidia-smi')
            cuda_available = True
        except:
            cuda_available = False

        if not cuda_available and 'torch' in name:
            pk[0] = pk[0][:pk[0].find('+cu')]
            for p in pk:
                if 'index-url' in p or 'download.pytorch.org/whl' in p:
                    params.remove(p)

        params.append("--target={}".format(site_package_path))
        adjusted_name = str(name).replace('-', '_')

        if Path(f'{site_package_path}/{adjusted_name}').exists():
            print(f'Skip installation of {site_package_path}/{name} - package already installed')
        else:
            install_package(pk[0], params)

    sys.path.insert(0, str(site_package_path.resolve()))


def install_package(pkg, params):
    call = [sys.executable, "-m", "pip", "install", pkg, *params]
    print(*call)
    return subprocess.check_call(call)
