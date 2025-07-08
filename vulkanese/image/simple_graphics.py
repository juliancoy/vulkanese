#!/bin/env python
import os
import time
import sys
import numpy as np
import json

# import open3d as o3d
import copy
import sdl2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import vulkanese as ve
import vulkan as vk

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sinode"))
)
import sinode.sinode as sinode

htHere = os.path.dirname(os.path.abspath(__file__))


class SimpleGraphicsPipeline(ve.graphics_pipeline.GraphicsPipeline):
    def __init__(self, device, surface, **kwargs):
        self.width = 700
        self.height = 700
        self.device = device
        self.setDefaults(**kwargs)

        FragBuffer = ve.buffer.VertexBuffer(
            device=self.device,
            name="fragColor",
            memtype="vec3",
            qualifier="out",
            location=3,
            dimensionVals=[self.VERTEX_COUNT, self.SPATIAL_DIMENSIONS],
            # dimensionVals=[self.width, self.height],
            stride=12,
            compress=False,
        )

        self.indexBuffer = ve.buffer.IndexBuffer(
            device=self.device,
            dimensionVals=[self.TRIANGLE_COUNT, self.VERTS_PER_TRIANGLE],
            stride=4,
        )

        # Input buffers to the shader
        self.vertexBuffers = [
            ve.buffer.VertexBuffer(
                device=self.device,
                name="position",
                memtype="vec3",
                qualifier="in",
                location=0,
                dimensionVals=[self.VERTEX_COUNT, self.SPATIAL_DIMENSIONS],
                stride=12,
                compress=False,
            ),
            ve.buffer.VertexBuffer(
                device=self.device,
                name="normal",
                memtype="vec3",
                qualifier="in",
                location=1,
                dimensionVals=[self.VERTEX_COUNT, self.SPATIAL_DIMENSIONS],
                stride=12,
                compress=False,
            ),
            ve.buffer.VertexBuffer(
                device=self.device,
                name="color",
                memtype="vec3",
                qualifier="in",
                location=2,
                dimensionVals=[self.VERTEX_COUNT, self.SPATIAL_DIMENSIONS],
                stride=12,
                compress=False,
            ),
            FragBuffer,
        ]

        self.fragmentBuffers = [
            ve.buffer.VertexBuffer(
                device=self.device,
                name="outColor",
                memtype="vec4",
                location=0,
                qualifier="out",
                dimensionVals=[self.width, self.height],
                stride=16,
                compress=False,
            ),
            FragBuffer,
        ]

        # finalize the buffers
        # device.descriptorPool.finalize()

        # (vertex -> tesselate -> fragment)
        # Vertex Stage
        self.vertexStage = ve.shader.VertexStage(
            device=self.device,
            constantsDict=self.constantsDict,
            buffers=self.vertexBuffers,
            name="vertexStage",
            sourceFilename=os.path.join(htHere, "shaders", "passthrough_vert.c"),
        )

        FragBuffer.qualifier = "in"
        # fragment stage
        self.fragmentStage = ve.shader.FragmentStage(
            device=self.device,
            buffers=self.fragmentBuffers,
            constantsDict=self.constantsDict,
            name="fragmentStage",
            sourceFilename=os.path.join(htHere, "shaders", "passthrough_frag.c"),
        )

        # create the standard graphics pipeline
        ve.graphics_pipeline.GraphicsPipeline.__init__(
            self,
            device=self.device,
            buffers=self.vertexBuffers + self.fragmentBuffers,
            shaders=[self.vertexStage, self.fragmentStage],
            indexBuffer=self.indexBuffer,
            constantsDict=self.constantsDict,
            surface=surface,
            outputWidthPixels=700,
            outputHeightPixels=700,
        )


if __name__ == "__main__":

    # constants declared here will be visible in
    # this class as an attribute,
    # and in the shader as a #define
    # we need 12 verts for a pyramid
    constantsDict = {
        "VERTEX_COUNT": 12,  # there are redundant verts
        "TRIANGLE_COUNT": 4,
        "VERTS_PER_TRIANGLE": 3,
        "SPATIAL_DIMENSIONS": 3,
        "COLOR_DIMENSIONS": 3,
    }

    # device selection and instantiation
    instance = ve.instance.Instance(verbose=True)
    print("naively choosing device 0")
    device = instance.getDevice(0)

    ht = SimpleGraphicsPipeline(device, constantsDict)
    ht.run()

    # elegantly free all memory
    instance.release()
