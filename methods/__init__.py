
from .fedavg import *
from .fedper import *
from .lg_fedavg import *
from .local import *
from .fedpac import *
from .fedgpa import *

def local_update(rule):
    LocalUpdate = {'FedAvg':LocalUpdate_FedAvg,
                   'FedPer':LocalUpdate_FedPer,
                   'LG_FedAvg':LocalUpdate_LG_FedAvg,
                   'Local':LocalUpdate_StandAlone,
                   'FedPAC': LocalUpdate_FedPAC,
                   'FedGPA':LocalUpdate_FedGPA,
    }

    return LocalUpdate[rule]