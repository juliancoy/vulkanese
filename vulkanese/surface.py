import vulkan as vk
from . import device as dd

import sdl2
import sdl2.ext
import ctypes
import json
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "sinode"))
)
import sinode.sinode as sinode


class Surface(sinode.Sinode):
    def __init__(self, instance, device, width, height):
        sinode.Sinode.__init__(self, parent =instance)
        self.running = True
        self.instance = instance
        self.device = device

        self.WIDTH = width
        self.HEIGHT = height
        self.extent = vk.VkExtent2D(width=self.WIDTH, height=self.HEIGHT)

        # ----------
        # Init sdl2
        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
            raise Exception(sdl2.SDL_GetError())

        self.window = sdl2.SDL_CreateWindow(
            "test".encode("ascii"),
            sdl2.SDL_WINDOWPOS_UNDEFINED,
            sdl2.SDL_WINDOWPOS_UNDEFINED,
            self.WIDTH,
            self.HEIGHT,
            0,
        )

        if not self.window:
            raise Exception(sdl2.SDL_GetError())

        self.wm_info = sdl2.SDL_SysWMinfo()
        sdl2.SDL_VERSION(self.wm_info.version)

        sdl2.SDL_GetWindowWMInfo(self.window, ctypes.byref(self.wm_info))

        # extensions = ["vk.VK_KHR_surface", "vk.VK_EXT_debug_report"]
        # if self.wm_info.subsystem == sdl2.SDL_SYSWM_WINDOWS:
        #    extensions.append("vk.VK_KHR_win32_surface")
        # elif self.wm_info.subsystem == sdl2.SDL_SYSWM_X11:
        #    extensions.append("vk.VK_KHR_xlib_surface")
        # elif self.wm_info.subsystem == sdl2.SDL_SYSWM_WAYLAND:
        #    extensions.append("vk.VK_KHR_wayland_surface")
        # else:
        #    raise Exception("Platform not supported: " + str(self.wm_info.subsystem))

        self.surface_mapping_dict = {
            sdl2.SDL_SYSWM_UNKNOWN: self.surface_xlib,
            sdl2.SDL_SYSWM_X11: self.surface_xlib,
            sdl2.SDL_SYSWM_WAYLAND: self.surface_wayland,
            sdl2.SDL_SYSWM_WINDOWS: self.surface_win32,
                }
        self.surface_mapping =  list(self.surface_mapping_dict.values())
        print(self.wm_info.subsystem)
        self.vkSurface = self.surface_mapping[self.wm_info.subsystem]()
        print(self.vkSurface)

        vkGetPhysicalDeviceSurfaceSupportKHR = vk.vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkGetPhysicalDeviceSurfaceSupportKHR"
        )
        for i, queue_family in enumerate(device.queueFamilies):
            print(queue_family)

            queue_familyPre = dd.ctypes2dict(queue_family)
            print(json.dumps(queue_familyPre, indent=2))
            support_present = vkGetPhysicalDeviceSurfaceSupportKHR(
                physicalDevice=device.physical_device,
                queueFamilyIndex=i,
                surface=self.vkSurface,
            )
            print(support_present)

        # ----------
        # Create swapchain
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR = vk.vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"
        )
        vkGetPhysicalDeviceSurfaceFormatsKHR = vk.vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkGetPhysicalDeviceSurfaceFormatsKHR"
        )
        vkGetPhysicalDeviceSurfacePresentModesKHR = vk.vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkGetPhysicalDeviceSurfacePresentModesKHR"
        )

        self.capabilities = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            physicalDevice=device.physical_device, surface=self.vkSurface
        )
        print(self.capabilities)
        self.formats = vkGetPhysicalDeviceSurfaceFormatsKHR(
            physicalDevice=device.physical_device, surface=self.vkSurface
        )
        self.present_modes = vkGetPhysicalDeviceSurfacePresentModesKHR(
            physicalDevice=device.physical_device, surface=self.vkSurface
        )

        if not self.formats or not self.present_modes:
            raise Exception("No available swapchain")

            width = max(
                capabilities.minImageExtent.width,
                min(capabilities.maxImageExtent.width, WIDTH),
            )
            height = max(
                capabilities.minImageExtent.height,
                min(capabilities.maxImageExtent.height, HEIGHT),
            )
            actualExtent = VkExtent2D(width=width, height=height)
            return actualExtent

        self.surface_format = self.get_surface_format()
        present_mode = self.get_surface_present_mode()
        self.extent = self.get_swap_extent()
        self.imageCount = self.capabilities.minImageCount + 1
        if (
            self.capabilities.maxImageCount > 0
            and self.imageCount > self.capabilities.maxImageCount
        ):
            self.imageCount = self.capabilities.maxImageCount

        print("selected format: %s" % self.surface_format.format)
        print("selected colorspace: %s" % self.surface_format.colorSpace)
        print("%s available swapchain present modes" % len(self.present_modes))
        print("image count " + str(self.imageCount))

        self.swapchain_create = vk.VkSwapchainCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            flags=0,
            surface=self.vkSurface,
            minImageCount=self.imageCount,
            imageFormat=self.surface_format.format,
            imageColorSpace=self.surface_format.colorSpace,
            imageExtent=self.extent,
            imageArrayLayers=1,
            imageUsage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            imageSharingMode=self.device.imageSharingMode,
            queueFamilyIndexCount=self.device.queueFamilyIndexCount,
            pQueueFamilyIndices=self.device.pQueueFamilyIndices,
            compositeAlpha=vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode=present_mode,
            clipped=vk.VK_TRUE,
            oldSwapchain=None,
            preTransform=self.capabilities.currentTransform,
        )

        vkCreateSwapchainKHR = vk.vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkCreateSwapchainKHR"
        )
        self.vkSwapchain = vkCreateSwapchainKHR(
            self.device.vkDevice, self.swapchain_create, None
        )
        vkGetSwapchainImagesKHR = vk.vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkGetSwapchainImagesKHR"
        )
        self.vkSwapchainImages = vkGetSwapchainImagesKHR(
            self.device.vkDevice, self.vkSwapchain
        )
        print("swapchain images " + str(self.vkSwapchainImages))
        self.children += [self.vkSwapchainImages]

    def getEvents(self):
        return sdl2.ext.get_events()

    def get_surface_format(self):
        for f in self.formats:
            print("FORMAT")
            print(f.format)
            if f.format == vk.VK_FORMAT_UNDEFINED:
                print("FORMAT UNDEFINED")
                return f
            if (
                f.format == vk.VK_FORMAT_B8G8R8A8_UNORM
                and f.colorSpace == vk.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
            ):
                print("FORMAT vk.VK_FORMAT_B8G8R8A8_UNORM")
                return f
        #die
        return self.formats[0]

    def get_surface_present_mode(self):
        for p in self.present_modes:
            if p == vk.VK_PRESENT_MODE_MAILBOX_KHR:
                return p
        return vk.VK_PRESENT_MODE_FIFO_KHR

    def get_swap_extent(self):
        uint32_max = 4294967295
        if self.capabilities.currentExtent.width != uint32_max:
            return vk.VkExtent2D(
                width=self.capabilities.currentExtent.width,
                height=self.capabilities.currentExtent.height,
            )

    def surface_xlib(self):
        print("Create Xlib surface")
        vkCreateXlibSurfaceKHR = vk.vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkCreateXlibSurfaceKHR"
        )
        surface_create = vk.VkXlibSurfaceCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
            dpy=self.wm_info.info.x11.display,
            window=self.wm_info.info.x11.window,
            flags=0,
        )
        return vkCreateXlibSurfaceKHR(self.instance.vkInstance, surface_create, None)

    def surface_wayland(self):
        print("Create wayland surface")
        vkCreateWaylandSurfaceKHR = vk.vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkCreateWaylandSurfaceKHR"
        )
        surface_create = vk.VkWaylandSurfaceCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
            display=self.wm_info.info.wl.display,
            surface=self.wm_info.info.wl.surface,
            flags=0,
        )
        return vkCreateWaylandSurfaceKHR(self.instance.vkInstance, surface_create, None)

    def surface_win32(self):
        def get_instance(hWnd):
            """Hack needed before SDL 2.0.6"""
            from cffi import FFI

            _ffi = FFI()
            _ffi.cdef("long __stdcall GetWindowLongA(void* hWnd, int nIndex);")
            _lib = _ffi.dlopen("User32.dll")
            return _lib.GetWindowLongA(_ffi.cast("void*", hWnd), -6)

        print("Create windows surface")
        vkCreateWin32SurfaceKHR = vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkCreateWin32SurfaceKHR"
        )
        surface_create = VkWin32SurfaceCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
            hinstance=get_instance(self.wm_info.info.win.window),
            hwnd=self.wm_info.info.win.window,
            flags=0,
        )
        return vkCreateWin32SurfaceKHR(self.instance.vkInstance, surface_create, None)

    def release(self):
        print("destroying surface")
        self.vkDestroySwapchainKHR = vk.vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkDestroySwapchainKHR"
        )
        self.vkDestroySurfaceKHR = vk.vkGetInstanceProcAddr(
            self.instance.vkInstance, "vkDestroySurfaceKHR"
        )
        self.vkDestroySwapchainKHR(self.device.vkDevice, self.swapchain, None)
        self.vkDestroySurfaceKHR(self.instance.vkInstance, self.vkSurface, None)
