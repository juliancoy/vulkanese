#!/bin/env python
import ctypes
import os
import time
import sys
import numpy as np
import json
import cv2
import math

here = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(here, "..", ".."))
sys.path.append(os.path.join(here, "..", "..", "..", "sinode"))
import sinode.sinode as sinode
import vulkanese as ve
import vulkan as vk

# Assuming sinode, vulkanese, and vulkan libraries are correctly imported and set up


class Transform(ve.shader.Shader):
    def __init__(self, device, WIDTH, HEIGHT, CHANNELS=3, **kwargs):
        self.WIDTH = int(WIDTH)
        self.HEIGHT = int(HEIGHT)
        self.CHANNELS = int(CHANNELS)

        self.device = device  # Make sure the device is passed and stored

        print((WIDTH, HEIGHT))
        workgroups = [
                math.ceil(float(WIDTH) / 16),
                math.ceil(float(HEIGHT) / 16),
                1,
            ]
        print(workgroups)

        # Set shader properties, including the path to the 'transform.template.comp'
        self.setDefaults(
            memProperties=0
            | vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            name="Transform",
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            sourceFilename=os.path.join(
                here, "shaders", "transform.template.comp"
            ), 
            constantsDict=dict(
                LOCAL_SIZE=16,  # Workgroup size in compute shader
            ),
            workgroupCount=workgroups
        )

        # Input image setup (assuming it's loaded or captured elsewhere)
        self.inputImage = ve.buffer.StorageBuffer(
            device=self.device,
            name="inputImage",
            memtype="vec4",
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            shape=[WIDTH,HEIGHT,4],
            compress=False
        )

        # Output image setup
        self.outputImage = ve.buffer.StorageBuffer(
            device=self.device,
            name="outputImage",
            memtype="vec4", 
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            shape=[WIDTH, HEIGHT,4],
            compress=False
        )

        # Transformation parameters buffer setup
        self.paramsBuffer = ve.buffer.UniformBuffer(
            device=self.device,
            name="paramsBuffer",
            memtype="float",  # Assuming your transformation parameters are floats
            shape=[128],  # Scale, rotation angle, translation (x, y)
            params={
                "width": "float",
                "height": "float",
                "scaleX": "float",
                "scaleY": "float",
                "rotationAngle": "float",
                "translationX": "float",
                "translationY": "float",
            },
        )

        self.buffers = [self.paramsBuffer, self.inputImage, self.outputImage]

        # Initialize the base shader
        ve.shader.Shader.__init__(self, device=self.device)
        self.finalize()

    def run(self, input_img, scale, rotation_angle, translation):
        # Set transformation parameters
        self.paramsBuffer.setByIndex(
            0,
            np.array(
                [
                    self.WIDTH,
                    self.HEIGHT,
                    scale[0],
                    scale[1],
                    rotation_angle,
                    translation[0],
                    translation[1],
                ],
                dtype=np.float32,
            ),
        )

        # Execute the shader
        ve.shader.Shader.run(self)

        # Retrieve and return the transformed image
        return self.outputImage.get()

# Example usage
def runDemo():

    # device selection and instantiation
    instance_inst = ve.instance.Instance(verbose=False)
    instance_inst.DEBUG = False
    device = instance_inst.getDevice(0)

    # Load or capture an input image...
    # OPENCV objects are HEIGHT, WIDTH, CHANNELS for some reason
    # Vk is WIDTH, HEIGHT, CHANNELS
    input_img = cv2.imread("example.png", cv2.IMREAD_UNCHANGED)  # Example input image
    input_img = np.transpose(input_img, (1, 0, 2))
    input_img = input_img.astype(np.float32)/255.0
    width = input_img.shape[0]
    height = input_img.shape[1]

    # Create a Transform object
    transform = Transform(
        device=device, WIDTH=width, HEIGHT=height
    )

    turns = 24.0
    # Upload the input image to GPU
    transform.inputImage.set(input_img)

    for i in range(int(turns)):
        
        # Run the transformation (example parameters)
        output_img = transform.run(
            #input_img, scale=1.0, rotation_angle=math.pi / 4, translation=[0.1, 0.1]
            input_img, scale=[2.0,2.0], rotation_angle=math.pi * 2*  (i/turns), translation=[0,0]
        )
        #cv2.imshow('Output Image', np.transpose(output_img, (1,0,2)))
        cv2.imshow('Output Image', output_img)
        cv2.waitKey(1)

        print(i)



# Example usage
def runWebcamDemo():

    # device selection and instantiation
    instance_inst = ve.instance.Instance(verbose=False)
    instance_inst.DEBUG = False
    device = instance_inst.getDevice(0)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    ret, input_img = cap.read()
    width = input_img.shape[1]
    height = input_img.shape[0]

    # Create a Transform object
    transform = Transform(
        device=device, WIDTH=width, HEIGHT=height
    )
    turns=24
    i=0
    while(1):

        # Read one frame from the webcam
        ret, input_img = cap.read()

        if not ret:
            print("Error: Could not read frame from webcam")
            cap.release()
            return

        # Process the captured frame...
        # OpenCV captures in HEIGHT, WIDTH, CHANNELS format but we need WIDTH, HEIGHT, CHANNELS
        input_img = input_img.astype(np.float32) / 255.0
        # Convert BGR to BGRA by adding an alpha channel
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2BGRA)
        input_img = np.transpose(input_img, (1, 0, 2))

        # Upload the input image to GPU
        #transform.inputImage.set(input_img)
        input_img_bytes = input_img.tobytes()
        transform.inputImage.pmap[0:len(input_img_bytes)] = input_img_bytes

        i+=1
        # Run the transformation with example parameters
        output_img = transform.run(
            input_img, scale=[3.0, 3.0], rotation_angle=math.pi * 2 * (i / turns), translation=[0, 0]
        )
        
        # Display the transformed image
        cv2.imshow('Output Image', cv2.resize(output_img, (int(height/2), int(width/2))))
        cv2.waitKey(1)

    # Close all OpenCV windows
    cv2.destroyAllWindows()
    # Close the webcam
    cap.release()



if __name__ == "__main__":
    runWebcamDemo()
