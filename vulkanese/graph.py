# include the Vulkanese directory
# (in case you're doing local development)
import sys
import os
import time

here = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(
    0, os.path.join(here, "..", "..", "sinode"))
sys.path.insert(0, os.path.join(here, ".."))
import vulkanese as ve
import numpy as np
import sinode.sinode as sinode

"""the compute graph representing
 result=(v+w)*(x+y)
"""


class Graph(sinode.Sinode):
    def __init__(self, **kwargs):
        sinode.Sinode.__init__(self, **kwargs)

        # first, declare all input buffers
        self.v = self.device.getStorageBuffer(name="v", shape=[128])
        self.w = self.device.getStorageBuffer(name="w", shape=[128])
        self.x = self.device.getStorageBuffer(name="x", shape=[128])
        self.y = self.device.getStorageBuffer(name="y", shape=[128])

        # then declare all shaders
        # shaders create their own output buffers
        # (typically called "result")
        self.addShader0 = ve.math.arith.add(
            name="add0", x=self.v, y=self.w, device=self.device
        )
        self.addShader1 = ve.math.arith.add(
            name="add1", x=self.x, y=self.y, device=self.device
        )
        self.multiplyShader = ve.math.arith.multiply(
            name="multiply",
            x=self.addShader0.gpuBuffers.result,
            y=self.addShader1.gpuBuffers.result,
            device=self.device,
            depends=[self.addShader0, self.addShader1],
        )
        self.result = self.multiplyShader.gpuBuffers.result
        self.shaders = [self.addShader0, self.addShader1, self.multiplyShader]
        for shader in self.shaders:
            shader.finalize()

    def run(self):
        for shader in self.shaders:
            shader.run()


def test(device):

    # begin GPU test
    simpleGraph = Graph(device=device, )
    # set the inputs
    simpleGraph.v.set(np.arange(128))
    simpleGraph.w.set(np.arange(128))
    simpleGraph.x.set(np.arange(128))
    simpleGraph.y.set(np.arange(128))
    simpleGraph.run()

    print(simpleGraph.result.get())


if __name__ == "__main__":
    instance = ve.instance.Instance(verbose=False)
    device = instance.getDevice(0)
    test(device=device)
    instance.dump()
    instance.release()
