import ctypes
import os
import sys
import time
import json
import vulkan as vk

here = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.abspath(os.path.join(here, "..")))
import vulkanese as ve
from vulkanese import synchronization as synchronization
from vulkanese import renderpass as renderpass
from vulkanese import graphics_command_buffer as graphics_command_buffer

import numpy as np
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sinode"))
)
import sinode.sinode as sinode


class GraphicsPipeline(sinode.Sinode):
    def __init__(
        self,
        **kwargs
    ):

        self.setDefaults(
            **kwargs
        )        
        
        sinode.Sinode.__init__(self, parent = self.device)

        self.setDefaults(
            outputWidthPixels=700,
            outputHeightPixels=700,
            culling=vk.VK_CULL_MODE_BACK_BIT,
            oversample=vk.VK_SAMPLE_COUNT_1_BIT,
            waitSemaphores=[],
        )

        # synchronization is owned by the pipeline (command buffer?)
        self.renderFence = synchronization.Fence(device=self.device)
        self.acquireFence = synchronization.Fence(device=self.device)
        self.fences = [self.renderFence]
        self.renderSemaphore = synchronization.Semaphore(device=self.device)
        self.presentSemaphore = synchronization.Semaphore(device=self.device)

        push_constant_ranges = vk.VkPushConstantRange(stageFlags=0, offset=0, size=0)
        # The pipeline layout allows the pipeline to access descriptor sets.
        # So we just specify the established descriptor set
        self.vkPipelineLayoutCreateInfo = vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            flags=0,
            # setLayoutCount=len(device.descriptorPool.descSets),
            # pSetLayouts=[
            #    d.vkDescriptorSetLayout for d in device.descriptorPool.descSets
            # ],
            pushConstantRangeCount=0,
            pPushConstantRanges=[push_constant_ranges],
        )

        self.vkPipelineLayout = vk.vkCreatePipelineLayout(
            device=self.device.vkDevice,
            pCreateInfo=[self.vkPipelineLayoutCreateInfo],
            pAllocator=None,
        )

        for shader in self.shaders:
            # make the buffer accessable as a local attribute
            exec("self." + shader.name + "= shader")

        # get global lists
        self.allVertexBuffers = []
        for b in self.vertexStage.buffers:
            if b.usage == vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT or b.name == "index":
                self.allVertexBuffers += [b]
        # self.allVertexBuffers += [self.indexBuffer]
        self.allVertexBuffers = list(set(self.allVertexBuffers))

        self.vkAcquireNextImageKHR = vk.vkGetInstanceProcAddr(
            self.device.instance.vkInstance, "vkAcquireNextImageKHR"
        )

        # Create a generic render pass
        self.renderpass = renderpass.RenderPass(
            pipeline=self,
            device=self.device,
            oversample=oversample,
            surface=self.surface,
        )
        self.children += [self.vkPipelineLayout]

        self.createGraphicsPipeline()
        self.recordCommandBuffers()
        self.frameNumber = 0

        self.vkQueuePresentKHR = vk.vkGetInstanceProcAddr(
            self.device.instance.vkInstance, "vkQueuePresentKHR"
        )

        print(self.surface.vkSwapchain)
        # presentation creator
        self.vkPresentInfoKHR = vk.VkPresentInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            waitSemaphoreCount=1,
            pWaitSemaphores=[
                self.renderSemaphore.vkSemaphore
            ],  # wait on the render before presenting
            swapchainCount=1,
            pSwapchains=[self.surface.vkSwapchain],
            pImageIndices=[0],
            pResults=None,
        )

        self.last_time = time.time()
        self.fps_last = 60
        self.fps = 0
        self.running = True

    def draw_frame(self):

        # timing
        self.fps += 1
        if time.time() - self.last_time >= 1:
            self.last_time = time.time()
            self.fps_last = self.fps
            self.fps = 0
            self.device.debug("FPS: %s" % self.fps)

        # acquire a writeable image
        self.device.debug("getting current GCB")
        image_index = self.vkAcquireNextImageKHR(
            device=self.device.vkDevice,
            swapchain=self.surface.vkSwapchain,
            timeout=vk.UINT64_MAX,
            # semaphore = None,
            semaphore=self.presentSemaphore.vkSemaphore,
            # fence = self.acquireFence.vkFence,
            fence=None,
        )
        self.device.debug("acquired image " + str(image_index))
        thisGCB = self.GraphicsCommandBuffers[image_index]
        # self.acquireFence.wait()

        # submit the appropriate queue
        self.device.debug("submitting queue")
        vk.vkQueueSubmit(
            self.device.graphic_queue, 1, [thisGCB.vkSubmitInfo], fence=None
        )  # self.renderFence.vkFence)
        # self.renderFence.wait()
        self.device.debug("presenting")

        # present it when finished
        self.vkPresentInfoKHR.pImageIndices[0] = image_index
        self.vkQueuePresentKHR(self.device.presentation_queue, self.vkPresentInfoKHR)

        vk.vkQueueWaitIdle(self.device.presentation_queue)

        # self.frameNumber = (self.frameNumber + 1) % 3

    def getAllBuffers(self):
        self.device.debug("Getting all buffers")
        allBuffers = []
        for shader in self.shaders:
            allBuffers += shader.buffers

        return list(set(allBuffers))

    def createGraphicsPipeline(self):
        self.device.debug("Creating graphics pipeline")

        # print([b.name for b in self.allVertexBuffers])
        # print([b.location for b in self.allVertexBuffers])
        # print(len(self.allVertexBuffers))

        # Create graphic Pipeline
        self.vertex_input_create = vk.VkPipelineVertexInputStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            flags=0,
            vertexBindingDescriptionCount=len(self.allVertexBuffers),
            pVertexBindingDescriptions=[
                b.bindingDescription for b in self.allVertexBuffers
            ],
            vertexAttributeDescriptionCount=len(self.allVertexBuffers),
            pVertexAttributeDescriptions=[
                b.attributeDescription for b in self.allVertexBuffers
            ],
        )

        input_assembly_create = vk.VkPipelineInputAssemblyStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            flags=0,
            topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitiveRestartEnable=vk.VK_FALSE,
        )
        viewport = vk.VkViewport(
            x=0.0,
            y=0.0,
            width=float(self.outputWidthPixels),
            height=float(self.outputHeightPixels),
            minDepth=0.0,
            maxDepth=1.0,
        )

        scissor_offset = vk.VkOffset2D(x=0, y=0)
        self.extent = vk.VkExtent2D(
            width=self.outputWidthPixels, height=self.outputHeightPixels
        )
        scissor = vk.VkRect2D(offset=scissor_offset, extent=self.extent)
        viewport_state_create = vk.VkPipelineViewportStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            flags=0,
            viewportCount=1,
            pViewports=[viewport],
            scissorCount=1,
            pScissors=[scissor],
        )

        rasterizer_create = vk.VkPipelineRasterizationStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            flags=0,
            depthClampEnable=vk.VK_FALSE,
            rasterizerDiscardEnable=vk.VK_FALSE,
            polygonMode=vk.VK_POLYGON_MODE_FILL,
            lineWidth=1,
            cullMode=self.culling,
            frontFace=vk.VK_FRONT_FACE_CLOCKWISE,
            depthBiasEnable=vk.VK_FALSE,
            depthBiasConstantFactor=0.0,
            depthBiasClamp=0.0,
            depthBiasSlopeFactor=0.0,
        )

        multisample_create = vk.VkPipelineMultisampleStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            flags=0,
            sampleShadingEnable=vk.VK_FALSE,
            rasterizationSamples=self.oversample,
            minSampleShading=1,
            pSampleMask=None,
            alphaToCoverageEnable=vk.VK_FALSE,
            alphaToOneEnable=vk.VK_FALSE,
        )

        color_blend_attachement = vk.VkPipelineColorBlendAttachmentState(
            colorWriteMask=vk.VK_COLOR_COMPONENT_R_BIT
            | vk.VK_COLOR_COMPONENT_G_BIT
            | vk.VK_COLOR_COMPONENT_B_BIT
            | vk.VK_COLOR_COMPONENT_A_BIT,
            blendEnable=vk.VK_FALSE,
            srcColorBlendFactor=vk.VK_BLEND_FACTOR_ONE,
            dstColorBlendFactor=vk.VK_BLEND_FACTOR_ZERO,
            colorBlendOp=vk.VK_BLEND_OP_ADD,
            srcAlphaBlendFactor=vk.VK_BLEND_FACTOR_ONE,
            dstAlphaBlendFactor=vk.VK_BLEND_FACTOR_ZERO,
            alphaBlendOp=vk.VK_BLEND_OP_ADD,
        )

        color_blend_create = vk.VkPipelineColorBlendStateCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            flags=0,
            logicOpEnable=vk.VK_FALSE,
            logicOp=vk.VK_LOGIC_OP_COPY,
            attachmentCount=1,
            pAttachments=[color_blend_attachement],
            blendConstants=[0, 0, 0, 0],
        )

        # Finally create graphics Pipeline
        self.vkGraphicsPipelineCreateInfo = vk.VkGraphicsPipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            flags=0,
            stageCount=len(self.shaders),
            pStages=[s.vkPipelineShaderStageCreateInfo for s in self.shaders],
            pVertexInputState=self.vertex_input_create,
            pInputAssemblyState=input_assembly_create,
            pTessellationState=None,
            pViewportState=viewport_state_create,
            pRasterizationState=rasterizer_create,
            pMultisampleState=multisample_create,
            pDepthStencilState=None,
            pColorBlendState=color_blend_create,
            pDynamicState=None,
            layout=self.vkPipelineLayout,
            renderPass=self.renderpass.vkRenderPass,
            subpass=0,
            basePipelineHandle=None,
            basePipelineIndex=-1,
        )

        pipelines = vk.vkCreateGraphicsPipelines(
            self.device.vkDevice, None, 1, [self.vkGraphicsPipelineCreateInfo], None
        )
        self.children += [pipelines]
        self.vkPipeline = pipelines[0]

    def recordCommandBuffers(self):
        self.device.debug("Creating command buffers")

        # Create command buffers, one for each image in the triple-buffer (swapchain + framebuffer)
        # OR one for each non-surface pass
        self.vkCommandBufferAllocateInfo = vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.device.vkGraphicsCommandPool,
            level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=self.surface.imageCount,
        )

        self.vkCommandBuffers = vk.vkAllocateCommandBuffers(
            self.device.vkDevice, self.vkCommandBufferAllocateInfo
        )

        self.device.debug("Recording command buffers")

        self.GraphicsCommandBuffers = []
        # Record command buffer
        for i, vkCommandBuffer in enumerate(self.vkCommandBuffers):
            self.GraphicsCommandBuffers += [
                graphics_command_buffer.GraphicsCommandBuffer(
                    device=self.device,
                    pipeline=self,
                    renderpass=self.renderpass,
                    surface=self.surface,
                    vkCommandBuffer=vkCommandBuffer,
                    index=i,
                )
            ]

    def release(self):
        self.device.debug("graphics pipeline release")

        for g in self.GraphicsCommandBuffers:
            g.release()

        self.renderpass.release()

        self.renderFence.release()
        self.acquireFence.release()
        self.renderSemaphore.release()
        self.presentSemaphore.release()

        vk.vkDestroyPipeline(self.device.vkDevice, self.vkPipeline, None)
        vk.vkDestroyPipelineLayout(self.device.vkDevice, self.vkPipelineLayout, None)
