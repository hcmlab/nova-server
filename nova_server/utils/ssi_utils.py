import xml.etree.ElementTree as ET
from pathlib import Path


class Trainer():

    def __init__(self,
                 model_script_path: str = '',
                 model_option_path: str = '',
                 model_weights_path: str = '',
                 model_stream: int = 0,
                 model_create: str = "PythonModel",
                 users: list = None,
                 classes: list = None,
                 streams: list = None,
                 register: list = None,
                 info_trained: bool = False,
                 meta_right_ctx: int = 0,
                 meta_left_ctx: int = 0,
                 meta_balance: str = 'none',
                 meta_backend: str = 'Python',
                 ssi_v="5",
                 xml_version="1.0"
                 ):

        self.model_script_path = model_script_path
        self.model_option_path = model_option_path
        self.model_weights_path = model_weights_path
        self.model_stream = model_stream
        self.model_create = model_create
        self.users = users if users is not None else []
        self.classes = classes if classes is not None else []
        self.streams = streams if streams is not None else []
        self.register = register if register is not None else []
        self.info_trained = info_trained
        self.meta_right_ctx = meta_right_ctx
        self.meta_left_ctx = meta_left_ctx
        self.meta_balance = meta_balance
        self.meta_backend = meta_backend
        self.ssi_v = ssi_v
        self.xml_version = xml_version

    def load_from_file(self, fp):
        root = ET.parse(Path(fp))
        info = root.find('info')
        meta = root.find('meta')
        register = root.find('register')
        streams = root.find('streams')
        classes = root.find('classes')
        users = root.find('users')
        model = root.find('model')

        if info is not None:
            self.info_trained = info.get('trained')
        if meta is not None:
            self.meta_left_ctx = int(meta.get('leftContext', default='0'))
            self.meta_right_ctx = int(meta.get('rightContext', default='0'))
            self.meta_balance = meta.get('balance', default='none')
            self.meta_backend = meta.get('backend', default='Python')
        if register is not None:
            for r in register:
                self.register.append(r.attrib)
        if streams is not None:
            for s in streams:
                self.streams.append(s.attrib)
        if classes is not None:
            for c in classes:
                self.classes.append(c.attrib)
        if users is not None:
            for u in users:
                self.users.append(u.attrib)
        if model is not None:
            self.model_stream = model.get('stream', default='0')
            self.model_create = model.get('create', default='PythonModel')
            self.model_option_path = model.get('option', default='')
            self.model_script_path = model.get('script', default='')
            self.model_weights_path = model.get('path', default='')

    def write_trainer_to_file(self, fp):
        root = ET.Element('trainer')
        ET.SubElement(root, 'info', trained=str(self.info_trained))
        ET.SubElement(root, 'meta', leftContext=str(self.meta_left_ctx), rightContex=str(self.meta_right_ctx), balance=self.meta_balance, backend=self.meta_backend)
        register = ET.SubElement(root, 'register')
        for r in self.register:
            ET.SubElement(register, 'item', **r)
        streams = ET.SubElement(root, 'streams')
        for s in self.streams:
            ET.SubElement(streams, 'item', **s)
        classes = ET.SubElement(root, 'classes')
        for c in self.classes:
            ET.SubElement(classes, 'item', **c)
        users = ET.SubElement(root, 'users')
        for u in self.users:
            ET.SubElement(users, 'item', **u)
        ET.SubElement(root, 'model', create=self.model_create, stream=str(self.model_stream), path=self.model_weights_path, script=self.model_script_path, option=self.model_option_path)

        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ", level=0)

        if not fp.suffix:
            fp = fp.with_suffix('.trainer')
        tree.write(fp)


if __name__ == '__main__':
    trainer_in_fp = Path(
        r'Z:\nova\cml\models\trainer\discrete\base_emotions\feature{compare[480ms,40ms,480ms]}\linsvm\linsvm.compare.trainer')
    trainer_out_fp = Path('.test_trainer.trainer')

    trainer = Trainer()
    trainer.load_from_file(trainer_in_fp)
    trainer.write_trainer_to_file(trainer_out_fp)

