import os
import sys
import numpy as np
import pkg_resources
import json

here = os.path.dirname(os.path.abspath(__file__))

if "vulkanese" not in [pkg.key for pkg in pkg_resources.working_set]:
    sys.path = [
        os.path.join(here, "..", "..")
    ] + sys.path

import vulkanese as ve

# get vulkanese instance
instance_inst = ve.instance.Instance(verbose=False)
print(json.dumps(instance_inst.getDeviceList(), indent=2))
instance_inst.release()
#
