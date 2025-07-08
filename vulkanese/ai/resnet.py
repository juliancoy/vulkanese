import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vulkanese as ve


def test(device):
    print("Testing RESNET50")


if __name__ == "__main__":
    # get the Vulkan instance
    instance = ve.instance.Instance(verbose=False)
    # get the default device
    device = instance.getDevice(0)
    test(device)
    instance.release()
