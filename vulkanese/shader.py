import json
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sinode"))
)
import sinode.sinode as sinode
import re
import vulkan as vk
from . import buffer
from . import compute_pipeline
import platform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import vulkanese as ve

from pathlib import Path

here = os.path.dirname(os.path.abspath(__file__))


class Empty:
    def __init__(self):
        pass


class Shader(sinode.Sinode):
    def __init__(self, **kwargs):
        sinode.Sinode.__init__(self, parent=kwargs["device"], **kwargs)

        if self not in self.device.shaders:
            self.device.shaders += [self]

        self.proc_kwargs(
            stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
            DEBUG=False,
            workgroupCount=[1, 1, 1],
            compressBuffers=True,
            waitSemaphores=[],
            depends=[],
            waitStages=None,
            signalSemaphores=[],  # these only used for compute shaders
            sourceFilename = ""
        )

        for shader in self.depends:
            newSemaphore = self.device.getSemaphore()
            shader.signalSemaphores += [newSemaphore]
            self.waitSemaphores += [newSemaphore]

        self.descriptorPool = ve.descriptor_pool.DescriptorPool(
            device=self.device, parent=self
        )

        self.gpuBuffers = Empty()
        self.basename = self.sourceFilename.replace(".template", "")
        #if not self.basename.endswith(".comp"):
        #    self.basename += ".comp"

        self.debugBuffers = []
        for buffer in self.buffers:
            # add the buffer to the descriptor pool
            self.descriptorPool.addBuffer(buffer)

            # make the buffer accessable as a local attribute
            exec("self.gpuBuffers." + buffer.name + "= buffer")

            # keep the debug buffers separately
            if buffer.DEBUG:
                self.debugBuffers += [buffer]

        # Desc Pools belong to the shader
        self.descriptorPool.finalize()

        outfilename = self.basename + ".spv"
        # if its the empty string "", just read it
        if self.sourceFilename == "":
            self.glslCode = self.sourceText
            self.compile()

        # if its spv (compiled), just run it
        elif self.sourceFilename.endswith(".spv"):
            with open(self.sourceFilename, "rb") as f:
                self.spirv = f.read()

        # if its not an spv, compile it
        elif ".template" in self.sourceFilename:
            with open(self.sourceFilename, "r") as f:
                self.glslCode = f.read()
            self.compile()
        else:
            raise Exception(
                "source template filename "
                + self.sourceFilename
                + " must end with .template"
            )

        # Create Stage
        self.vkShaderModuleCreateInfo = vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            # flags=0,
            codeSize=len(self.spirv),
            pCode=self.spirv,
        )

        self.vkShaderModule = vk.vkCreateShaderModule(
            self.device.vkDevice, self.vkShaderModuleCreateInfo, None
        )

        # Create Shader stage
        self.vkPipelineShaderStageCreateInfo = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=self.stage,
            module=self.vkShaderModule,
            flags=0,
            pSpecializationInfo=None,
            pName="main",
        )
        self.debug("creating Stage " + str(self.stage))

    def finalize(self):
        # if this is a compute shader, it corresponds with a single pipeline. we create that here
        if self.stage == vk.VK_SHADER_STAGE_COMPUTE_BIT:
            # generate a compute cmd buffer
            self.computePipeline = compute_pipeline.ComputePipeline(
                parent=self,
                computeShader=self,
                device=self.device,
                constantsDict=self.constantsDict,
                workgroupCount=self.workgroupCount,
                waitSemaphores=self.waitSemaphores,
                signalSemaphores=self.signalSemaphores,
            )

        # first run is always slow
        # run once in init so people dont judge the first run
        # self.run()

    def release(self):
        self.debug("destroying descriptor Pool")
        self.descriptorPool.release()
        if hasattr(self, "computePipeline"):
            self.debug("destroying Pipeline")
            self.computePipeline.release()
        self.debug("destroying shader")
        vk.vkDestroyShaderModule(self.device.vkDevice, self.vkShaderModule, None)

    def compile(self):

        # PREPROCESS THE SHADER CODE
        # RELATIVE TO DEFINED BUFFERS

        BUFFERS_STRING = ""
        # novel INPUT buffers belong to THIS Stage (others are linked)
        if self.stage == vk.VK_SHADER_STAGE_VERTEX_BIT:

            for b in self.buffers:
                if self.stage == vk.VK_SHADER_STAGE_FRAGMENT_BIT and b.name == "fragColor":
                    b.qualifier = "in"
                    BUFFERS_STRING += b.getDeclaration()
        else:
            BUFFERS_STRING += self.descriptorPool.getComputeDeclaration()

        if self.DEBUG:
            self.glslCode = self.addIndicesToOutputs(self.glslCode)

        # put structs and buffers into the code
        self.glslCode = self.glslCode.replace("BUFFERS_STRING", BUFFERS_STRING)
        

        # add definitions from constants dict
        DEFINE_STRING = ""
        for k, v in self.constantsDict.items():
            DEFINE_STRING += "#define " + k + " " + str(v) + "\n"
        self.glslCode = self.glslCode.replace("DEFINE_STRING", DEFINE_STRING)

        # COMPILE GLSL TO SPIR-V
        self.debug("compiling Stage")
        glslFilename = self.basename

        with open(glslFilename, "w+") as f:
            f.write(self.glslCode)

        # POS always outputs to "a.spv"
        compiledFilename = "a.spv"

        # delete the old one
        if os.path.exists(compiledFilename):
            os.remove(compiledFilename)

        self.debug("running " + glslFilename)
        # os.system("glslc --scalar-block-layout " + glslFilename)
        if platform.system() == "Windows":
            glslcbin = "glslc"
        else:
            glslcbin = os.path.join(here, "glslc")
        os.system(glslcbin + " --target-env=vulkan1.1 " + glslFilename)
        with open(compiledFilename, "rb") as f:
            self.spirv = f.read()

        # delete it after reading
        if os.path.exists(compiledFilename):
            os.remove(compiledFilename)

        return self.spirv

    def run(self, blocking=True):
        #self.dump()
        self.computePipeline.run(blocking=blocking)

    def wait(self):
        self.computePipeline.wait()

    def getVertexBuffers(self):
        allVertexBuffers = []
        for b in self.buffers:
            if b.usage == vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT:
                allVertexBuffers += [b]
        return allVertexBuffers

    def dumpMemory(self, filename="debug.json"):
        outdict = {}
        for b in self.buffers:
            if not b.DEBUG:
                continue
            rcvdArray = b.getAsNumpyArray()[0]
            # convert to list to make it JSON serializable
            outdict[b.name] = rcvdArray.tolist()  # nested lists with same data, indices
        with open(filename, "w+") as f:
            self.debug("dumping to " + filename)
            json.dump(outdict, f, indent=4)

    # take the template GLSL file
    # and add output indices
    # mostly for debugging
    def addIndicesToOutputs(self, shaderGLSL):
        indexedVarStrings = []
        # if this is debug mode, all vars have already been declared as output buffers
        # intermediate variables must be indexed
        for b in self.debugBuffers:
            indexedVarString = b.name + "["
            for i, d in enumerate(b.dimIndexNames):
                indexedVarString += d
                # since GLSL doesnt allow multidim arrays (kinda, see gl_arb_arrays_of_arrays)
                # ok i dont understand Vulkan multidim
                for j, rd in enumerate(b.shape[i + 1 :]):
                    indexedVarString += "*" + str(rd)
                if i < (len(b.dimIndexNames) - 1):
                    indexedVarString += "+"
            indexedVarString += "]"
            indexedVarStrings += [(b.name, indexedVarString)]

        outShaderGLSL = ""
        for line in shaderGLSL.split("\n"):
            # replace all non-comments
            if not line.strip().startswith("//") and not line.strip().endswith("debug"):
                # whole-word replacement
                for iv in indexedVarStrings:
                    line = re.sub(r"\b{}\b".format(iv[0]), iv[1], line)
            # keep the comments
            outShaderGLSL += line + "\n"

        return outShaderGLSL


class VertexStage(Shader):
    def __init__(
        self,
        **kwargs
    ):
        Shader.__init__(
            self,
            stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
            **kwargs
        )


class FragmentStage(Shader):
    def __init__(
        self,
        device,
        buffers,
        constantsDict,
        sourceFilename,
        name="mandlebrot",
        DEBUG=False,
    ):
        Shader.__init__(
            self,
            device=device,
            buffers=buffers,
            constantsDict=constantsDict,
            sourceFilename=sourceFilename,
            stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            name=name,
            DEBUG=False,
        )
