# vulkanese  
## Installation  
```pip install git+https://github.com/julianfl0w/vulkanese```

Note: if you're on Windows, you will also need 
The Vulkan SDK
https://vulkan.lunarg.com/sdk/home

## It's Vulkan-Ease!  

This repository  
* Imposes a hierarchical structure on Vulkan
* Dramatically simplifies Vulkan usage
* Is pure python
* Runs SPIR-V compute shaders efficiently, across all modern GPUs
* Makes compute shader debugging easy
* Easily integrates with Numpy
* Easily chain up GPU operations using semaphores

It is comparable to
* Nvidia's CUDA
* [Linux's Kompute](https://github.com/KomputeProject/kompute)
* [Alibaba's MNN](https://github.com/alibaba/MNN)

## The Topology
For simplicity, the Python Classes contain one another in the following topology  
<img src="https://github.com/julianfl0w/vulkanese/assets/8158655/7f45a7b3-f642-4e8a-b106-8bf85cf8aae3" width="60%">

## Sponsors
![image](https://github.com/julianfl0w/vulkanese/assets/8158655/c980a734-61f4-41f3-94b6-52c56b5fd2f7)

## Mandlebrot Example
```python
import vulkanese as ve
# device selection and instantiation
instance_inst = ve.instance.Instance(verbose=True)
# choose a device
print("naively choosing device 0")
device = instance_inst.getDevice(0)
screen = ve.screen.FullScreen()

ve.image.Mandlebrot(
    device=device,
    instance=instance_inst,
    parent=device,
    WIDTH=screen.display.width,
    HEIGHT=screen.display.height
    ).runDemo()
```
<img src="https://github.com/julianfl0w/vulkanese/assets/8158655/b0180362-bf91-4698-a4cc-48d8a08c71e1" width="60%">

## GPGPU Example: Pitch Detection 
I've implemented a world class pitch detector in GPU, based on the Loiacono Transform. 
Here is a snapshot of that code, which shows how to use Vulkanese to manage compute shaders:

```python
import vulkanese as ve

# generate a sine wave at A440, SR=48000
sr = 48000
A4 = 440
z = np.sin(np.arange(2**15)*2*np.pi*A4/sr)

multiple = 40
normalizedStep = 5.0/sr
# create a linear distribution of desired frequencies
fprime = np.arange(100/sr,3000/sr,normalizedStep)

# generate a Loiacono based on this SR
# (this one runs in CPU. reference only)
linst = Loiacono(
    fprime = fprime,
    multiple=multiple,
    dtftlen=2**15
)
linst.debugRun(z)

# begin GPU test
instance = ve.instance.Instance(verbose=False)
device = instance.getDevice(0)
linst_gpu = Loiacono_GPU(
    device = device,
    fprime = fprime,
    multiple = linst.multiple,
)
linst_gpu.gpuBuffers.x.set(z)
for i in range(10):
    linst_gpu.debugRun()
#linst_gpu.dumpMemory()
readstart = time.time()
linst_gpu.spectrum = linst_gpu.gpuBuffers.L.getAsNumpyArray()
print("Readtime " + str(time.time()- readstart))

fig, ((ax1, ax2)) = plt.subplots(1, 2)
ax1.plot(linst.fprime*sr, linst_gpu.spectrum)
ax1.set_title("GPU Result")
ax2.plot(linst.fprime*sr, linst.spectrum)
ax2.set_title("CPU Result")

plt.show()
```

We have sucessfully detected the 440Hz signal in this simple example:
![image](https://user-images.githubusercontent.com/8158655/205408263-2ab2236b-1b76-4f7d-9f6c-e4813ccb12d7.png)

## Future Development
Vulkanese has potential to simplify the fields of
* High-power Computing
* Gaming
* Graphics Rendering

Possible avenues for development include
* A guide for Distributed High-power Computing (HPC) using Docker and Kubernetes **(ACTIVE PROJECT)**
* A more complete math library
* Machine Learning / AI (Tensorflow Backend)
* Mobile (Android) port for gaming 
* Mobile (Android) port for embedded devices 
* No-code GUI for pipeline development
* Raytracing support
* Virtual Reality 
