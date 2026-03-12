'''READ config file and get elements'''

import yaml
from icecream import ic
from typing import NamedTuple
from pprint import pprint

class Colours(NamedTuple):
    HEADER = "\033[95m"   # purple/magenta
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"       # reset to default

class YMLparser():

    def __init__(self, file_pth:str = 'config.yml'):

        self.file_ = file_pth
        # temporaly yml file removed from here and use _open
        self._print_config()


    def _open(self):
        with open(self.file_) as conf:
            return yaml.safe_load(conf) 


    def _print_config(self):
        """Print Config.yml"""
        file_ = self._open()
        print('\n')
        for section, values in file_.items():
            print(f"{Colours.HEADER}{section}:") 
            for key, value in values.items():
                print(f" {Colours.GREEN}{key}: {Colours.YELLOW}{value}{Colours.END}")
        print('\n')


    def get(self,ref = None):
        config = self._open()
        if not ref:
            return config
        else:
            return config[ref]
        

    def update_yml(self,ref,dict_:dict):

        YMLfile = self.get()
        YMLfile[ref].update(dict_)
        
        with open(self.file_, "w") as f:
            yaml.dump(YMLfile, f)




if __name__ == '__main__':
    print('what are you doing? you are doing nothing')
            



