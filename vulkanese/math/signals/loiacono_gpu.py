import os
import sys
import pkg_resources
import time
from scipy.signal import get_window

gpuhere = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
import vulkanese as ve
import vulkan as vk
import numpy as np

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "sinode")
    ),
)
import sinode.sinode as sinode

loiacono_home = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt

# Create a compute shader
class Loiacono_GPU(ve.shader.Shader):
    def __init__(self, **kwargs):
        sinode.Sinode.__init__(self, **kwargs)
        self.proc_kwargs(
            signalLength=2 ** 15,
            constantsDict={},
            DEBUG=False,
            buffType="float",
            memProperties=0
            | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        )

        # the constants will be placed into the shader.comp file,
        # and also available in Python
        constantsDict = {}
        constantsDict["multiple"] = self.multiple
        constantsDict["SIGNAL_LENGTH"] = self.signalLength
        constantsDict["PROCTYPE"] = self.buffType
        constantsDict["TOTAL_THREAD_COUNT"] = self.signalLength * len(self.fprime)
        constantsDict["LG_WG_SIZE"] = 7
        constantsDict["THREADS_PER_WORKGROUP"] = 1 << constantsDict["LG_WG_SIZE"]
        constantsDict["windowed"] = 0

        # device selection and instantiation
        self.instance = self.device.instance
        self.constantsDict = constantsDict
        self.numSubgroups = (
            self.signalLength * len(self.fprime) / self.device.subgroupSize
        )
        self.numSubgroupsPerFprime = int(self.numSubgroups / len(self.fprime))
        self.spectrum = np.zeros((len(self.fprime)))

        # declare buffers. they will be in GPU memory, but visible from the host (!)
        buffers = [
            # x is the input signal
            ve.buffer.StorageBuffer(
                device=self.device,
                name="x",
                memtype=self.buffType,
                qualifier="readonly",
                shape=[2 ** 15],  # always 32**3
                memProperties=self.memProperties,
            ),
            # The following 4 are reduction buffers
            # Intermediate buffers for computing the sum
            ve.buffer.DebugBuffer(
                device=self.device,
                name="Li1",
                memtype=self.buffType,
                shape=[len(self.fprime), self.device.subgroupSize ** 2],
                dimIndexNames=["frequency_ix", "sg"],
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="Lr1",
                memtype=self.buffType,
                shape=[len(self.fprime), self.device.subgroupSize ** 2],
                dimIndexNames=["F", "sg"],
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="Li0",
                memtype=self.buffType,
                shape=[len(self.fprime), self.device.subgroupSize],
                dimIndexNames=["F", "sg"],
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="Lr0",
                memtype=self.buffType,
                shape=[len(self.fprime), self.device.subgroupSize],
                dimIndexNames=["F", "sg"],
            ),
            # L is the final output
            ve.buffer.StorageBuffer(
                device=self.device,
                name="L",
                memtype=self.buffType,
                qualifier="writeonly",
                shape=[len(self.fprime)],
                dimIndexNames=["F"],
                memProperties=self.memProperties,
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="f",
                memtype=self.buffType,
                qualifier="readonly",
                shape=[len(self.fprime)],
                dimIndexNames=["F"],
                memProperties=self.memProperties,
            ),
            ve.buffer.StorageBuffer(
                device=self.device,
                name="offset",
                memtype="uint",
                qualifier="readonly",
                shape=[16],
                memProperties=self.memProperties,
            ),
            # StorageBuffer(
            #    device=self.device,
            #    name="allShaders",
            #    memtype=buffType,
            #    shape=[constantsDict["TOTAL_THREAD_COUNT"]],
            # ),
        ]

        if constantsDict["windowed"]:
            buffers += [
                ve.buffer.StorageBuffer(
                    device=self.device,
                    name="window",
                    memtype=self.buffType,
                    qualifier="readonly",
                    shape=[1024],  # always 32**3
                    memProperties=self.memProperties,
                )
            ]

        # self.descriptorPool.finalize()

        # Create a compute shader
        # Compute Stage: the only stage
        ve.shader.Shader.__init__(
            self,
            sourceFilename=os.path.join(
                loiacono_home, "shaders/loiacono.template.comp"
            ),
            constantsDict=self.constantsDict,
            device=self.device,
            name="loiacono",
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            buffers=buffers,
            DEBUG=self.DEBUG,
            workgroupCount=[
                int(
                    self.signalLength
                    * len(self.fprime)
                    / (constantsDict["THREADS_PER_WORKGROUP"])
                ),
                1,
                1,
            ],
            useFence=True,
        )

        self.gpuBuffers.f.set(self.fprime)
        self.gpuBuffers.offset.zeroInitialize()
        self.offset = 0
        if constantsDict["windowed"]:
            self.gpuBuffers.window.set(get_window("hamming", 1024))

        self.finalize()

    def debugRun(self, z):
        linst_gpu.gpuBuffers.x.set(z)
        vstart = time.time()
        self.run()
        vlen = time.time() - vstart
        self.spectrum = self.gpuBuffers.L
        print("vlen " + str(vlen))
        # return self.sumOut.get()

    def feed(self, newData, blocking=True):
        self.gpuBuffers.x.setByIndexStart(self.offset, newData)
        self.offset = (self.offset + len(newData)) % self.signalLength
        self.gpuBuffers.offset.setByIndex(index=0, data=[self.offset])
        self.run(blocking)

    def getSpectrum(self):
        self.spectrum = self.gpuBuffers.L.get()
        return self.spectrum


if __name__ == "__main__":
    # generate a sine wave at A440, SR=48000
    sr = 48000
    A4 = 440
    z = np.sin(np.arange(2 ** 15) * 2 * np.pi * A4 / sr)
    z += np.sin(2 * np.arange(2 ** 15) * 2 * np.pi * A4 / sr)
    z += np.sin(3 * np.arange(2 ** 15) * 2 * np.pi * A4 / sr)
    z += np.sin(4 * np.arange(2 ** 15) * 2 * np.pi * A4 / sr)
    # Ensure z is float32 to match GPU buffer type
    z = z.astype(np.float32)
    
    multiple = 40
    num_frequencies = 1024  # or any fixed length you prefer
    min_freq_hz = 100
    max_freq_hz = 3000

    # Create a fixed-length frequency array (in Hz)
    fprime_hz = np.linspace(min_freq_hz, max_freq_hz, num_frequencies)

    # Normalize by the sampling rate for internal use
    fprime = fprime_hz / sr

    

    # generate a Loiacono based on this SR
    # (this one runs in CPU. reference only)
    linst = ve.math.signals.loiacono.Loiacono(
        fprime=fprime, multiple=multiple, dtftlen=2 ** 15
    )
    # begin GPU test
    instance = ve.instance.Instance(verbose=True)
    device = instance.getDevice(0)
    # Convert fprime to float32 to match GPU buffer type
    fprime_float32 = fprime.astype(np.float32)
    linst_gpu = Loiacono_GPU(
        device=device, parent=device, fprime=fprime_float32, multiple=linst.multiple
    )
    print("--- Running CPU Test ---")
    for i in range(10):
        linst.debugRun(z)
    print("--- Running GPU Test ---")
    for i in range(10):
        linst_gpu.debugRun(z)
    # linst_gpu.dumpMemory()
    readstart = time.time()
    linst_gpu.spectrum = linst_gpu.gpuBuffers.L.get()
    print("Readtime " + str(time.time() - readstart))

    graph = True
    if graph:
        fig, ((ax1, ax2)) = plt.subplots(1, 2)
        ax1.plot(linst.fprime * sr, linst_gpu.spectrum)
        ax1.set_title("GPU Result")
        ax2.plot(linst.fprime * sr, linst.spectrum)
        ax2.set_title("CPU Result")

        plt.show()

    instance.dump()
    instance.release()
