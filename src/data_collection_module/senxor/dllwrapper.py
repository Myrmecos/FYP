#  Copyright (C) Meridian Innovation Ltd. 2024. All rights reserved.
#
import ctypes as ct
from pathlib import Path
import platform
import numpy as np

file_path = Path(__file__).resolve()
dll_dir  = file_path.parent.parent / 'dll'

# Convert python float to C float 32
def toCtypeFloat32(pData):
    return np.ctypeslib.as_ctypes(pData.astype(np.float32))

# Convert python uint16 to C uint16
def toCtypeUint16(data:np.uint16):
    """Return a pointer to an ctype array of the correct type and size"""
    return np.ctypeslib.as_ctypes(pData.astype(np.uint16))

class DistanceCompensationModel_1:
    if platform.system() == 'Windows':
        if platform.machine() == 'i386':
            dll_file = dll_dir / 'mi48_filters_Win32.dll'
        else:
            dll_file = dll_dir / 'mi48_filters_x64.dll'
    elif platform.system() == "Darwin":
        # Dynamic library for macOS is universal (compatible with x64, and aarch64)
        # Modern macOS had dropped the support of 32bit machine
        dll_file = dll_dir / 'mi48_filters.dylib'
    else:
        if platform.machine() == 'x86_64':
            dll_file = dll_dir / 'mi48_filters_x86.so'
        elif platform.machine() == 'x64':
            dll_file = dll_dir / 'mi48_filters_x64.so'
        elif platform.machine() == 'aarch64':
            # Replace mi48_filters_ARM64 with mi48_filters_pi_ARM64
            # if this script is deployed in Raspberry Pi
            dll_file = dll_dir / 'mi48_filters_pi_ARM64.so'
        else:
            # Replace mi48_filters_ARM64 with mi48_filters_pi_ARM
            # if this script is deployed in Raspberry Pi
            dll_file = dll_dir / 'mi48_filters_pi_ARM.so'

    print("Loading SenXor DLL file: ", dll_file)
    _dll = ct.CDLL(str(dll_file))

    # distance compensation model methods definitions
    _dll.disCompModel_new.argtypes = (ct.POINTER(ct.c_uint8),)
    _dll.disCompModel_new.restype = ct.POINTER(ct.c_void_p)

    _dll.disCompModel_delete.argtypes = (ct.POINTER(ct.c_void_p),)
    _dll.disCompModel_delete.restype = None

    _dll.DisCompModel_paramSettings.argtypes = (ct.POINTER(ct.c_void_p),\
                                ct.c_float, ct.c_float, ct.c_float, ct.c_float)
    _dll.DisCompModel_paramSettings.restype = None

    _dll.getCompRst.argtypes = (ct.POINTER(ct.c_void_p),ct.c_float, ct.c_float, ct.c_float)
    _dll.getCompRst.restype = ct.c_float

    _dll.getInfo.argtypes = (ct.POINTER(ct.c_void_p),)
    _dll.getInfo.restype = None


    def __init__(self, module_type, parameters=None):
        """
        Initialize the class through the constructor API function, and set
        model parameters (instance parameters) also via an API call
        """
        # call the constructor API function
        self.model = self._dll.disCompModel_new(ct.c_uint8(module_type))
        if parameters is not None:
            self.set_parameters(parameters)

    def set_parameters(self, parameters):
        """
        Set model parameters through the C API.
        """
        a, b, c, d = [ct.c_float(p) for p in parameters]
        self._dll.DisCompModel_paramSettings(self.model, a, b, c, d)
        self._dll.getInfo(self.model)

    def print_info(self):
        self._dll.getInfo(self.model)

    def get_result(self, Tmax_degC, Tbackground_degC, distance_m):
        """
        Obtain compensated value of Tmax, given Tbackground and Distance
        """
        return self._dll.getCompRst(self.model, Tmax_degC, Tbackground_degC, distance_m)

    def __call__(self, Tmax_degC, Tbackground_degC, distance_m):
        return self.get_result(Tmax_degC, Tbackground_degC, distance_m)

    def __del__(self):
        """
        Clean up memory by calling the destuctor API function
        Recall that Python does not manage C memory space.
        So the calling routing using this class must do:
            del class_instance
        """
        self._dll.disCompModel_delete(self.model)


# C++ filters

class C_Filter:
    if platform.system() == 'Windows':
        if platform.machine() == 'i386':
            dll_file = dll_dir / 'mi48_filters_Win32.dll'
        else:
            dll_file = dll_dir / 'mi48_filters_x64.dll'
    elif platform.system() == "Darwin":
        # Dynamic library for macOS is universal (compatible with x64, and aarch64)
        # Modern macOS had dropped the support of 32bit machine
        dll_file = dll_dir / 'mi48_filters.dylib'
    else:
        if platform.machine() == 'x86_64':
            dll_file = dll_dir / 'mi48_filters_x86.so'
        elif platform.machine() == 'x64':
            dll_file = dll_dir / 'mi48_filters_x64.so'
        elif platform.machine() == 'aarch64':
            # Replace mi48_filters_ARM64 with mi48_filters_pi_ARM64
            # if this script is deployed in Raspberry Pi
            dll_file = dll_dir / 'mi48_filters_pi_ARM64.so'
        else:
            # Replace mi48_filters_ARM64 with mi48_filters_pi_ARM
            # if this script is deployed in Raspberry Pi
            dll_file = dll_dir / 'mi48_filters_pi_ARM.so'

    print("Loading SenXor DLL file: ", dll_file)
    _dll = ct.CDLL(str(dll_file))

    _dll.clib_new.argtypes = None
    _dll.clib_new.restype = ct.POINTER(ct.c_void_p)

    _dll.clib_delete.argtypes = ct.POINTER(ct.c_void_p),
    _dll.clib_delete.restype = None

    _dll.Dnlce_settings.argtypes = ct.POINTER(ct.c_void_p), ct.c_bool
    _dll.Dnlce_settings.restype = None
    
    _dll.dnlce.argtypes = ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_uint16), ct.POINTER(ct.c_int)
    _dll.dnlce.restype = ct.POINTER(ct.c_uint16)

    _dll.Stark_settings.argtypes = ct.POINTER(ct.c_void_p), ct.c_bool, ct.c_bool,  ct.c_uint, ct.c_bool, ct.c_bool, ct.c_uint8, ct.c_uint8, ct.c_uint8
    _dll.Stark_settings.restype = None

    _dll.STARK_Filter.argtypes = ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_uint16), ct.POINTER(ct.c_int)
    _dll.STARK_Filter.restype = ct.POINTER(ct.c_uint16)

    _dll.Median_settings.argtypes = ct.POINTER(ct.c_void_p), ct.c_bool, ct.c_bool
    _dll.Median_settings.restype = None

    _dll.median_filter.argtypes = ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_uint16), ct.POINTER(ct.c_int)
    _dll.median_filter.restype = ct.POINTER(ct.c_uint16)

    _dll.Kxms_settings.argtypes = ct.POINTER(ct.c_void_p), ct.c_bool, ct.c_bool
    _dll.Kxms_settings.restype = None

    _dll.KXMS_stabilizer.argtypes = ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_uint16), ct.POINTER(ct.c_int)
    _dll.KXMS_stabilizer.restype = ct.POINTER(ct.c_uint16)

    _dll.validateFrame.argtypes = ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_uint16), ct.c_uint16
    _dll.validateFrame.restype = None

    _dll.setFilterParms.argtypes = ct.POINTER(ct.c_void_p), ct.c_int, ct.c_int, ct.c_uint8, ct.c_int, ct.c_int,
    _dll.setFilterParms.restype = None

    _dll.Customer_imageprocessing.argtypes = ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_uint16), ct.POINTER(ct.c_int)
    _dll.Customer_imageprocessing.restype = ct.POINTER(ct.c_uint16)

    _dll.HasanFilter_Settings.argtypes = ct.POINTER(ct.c_void_p), ct.c_bool
    _dll.HasanFilter_Settings.restype = None

    _dll.Hasan_Filter.argtypes = ct.POINTER(ct.c_void_p),ct.POINTER(ct.c_uint16), ct.POINTER(ct.c_int)
    _dll.Hasan_Filter.restype = ct.POINTER(ct.c_uint16)

    # _dll.Get_Min.argtypes = PC_FILTER
    _dll.Get_Min.restype = ct.c_uint16

    # _dll.Get_Max.argtypes = PC_FILTER
    _dll.Get_Max.restype = ct.c_uint16

    # Testing the STARK filter


    def __init__(self):
        self.obj = self._dll.clib_new()

    def __del__(self):
        self._dll.clib_delete(self.obj)

    def Filter_setParams(self, pCols, pRows, pGain, pModType, pFps_Divisor):
        self._dll.setFilterParms(self.obj, pCols, pRows, pGain, pModType, pFps_Divisor)

    def Dnlce_settings(self, pIsDnlceEn):
        self._dll.Dnlce_settings(self.obj, pIsDnlceEn)

    def Dnlce(self, buffer):
        size = ct.c_int()
        m = self._dll.dnlce(self.obj, buffer, ct.byref(size))
        return np.array(m[:size.value])

    # STARK
    def STARK_Settings(self, StarkEnable, StarkAuto, pSTARKRev, StarkBackgroundSmooth,
                       StarkKernelSize, StarkCutoff, StarkGradient, StarkScale):
        self._dll.Stark_settings(self.obj, StarkEnable, StarkAuto, pSTARKRev,
                                 StarkBackgroundSmooth, StarkKernelSize, StarkCutoff,
                                 StarkGradient, StarkScale)
    def STARKFilter(self, buffer):
        size = ct.c_int()
        m = self._dll.STARK_Filter(self.obj, buffer, ct.byref(size))
        return np.array(m[:size.value])

    # KXMS
    def KXMS_Settings(self, KxmsEnable, KxmsRa):
        self._dll.Kxms_settings(self.obj, KxmsEnable, KxmsRa)

    def Kxms_stabliser(self, buffer):
        size = ct.c_int()
        m = self._dll.KXMS_stabilizer(self.obj, buffer, ct.byref(size))
        return np.array(m[:size.value])

    # Median
    def Median_Settings(self, medianEnable, medianKernelSize):
        self._dll.Median_settings(self.obj, medianEnable, medianKernelSize)

    def MedianFilter(self, buffer):
        size = ct.c_int()
        m = self._dll.median_filter(self.obj, buffer, ct.byref(size))
        return np.array(m[:size.value])

    def Customer_imageprocessing(self, frame):
        shape = frame.shape
        pointer_to_array = frame.ctypes.data_as(ct.POINTER(ct.c_uint16))
        size = ct.c_int()
        pointer_to_array = self._dll.Customer_imageprocessing(self.obj, pointer_to_array,
                                                              ct.byref(size))
        processed = np.ctypeslib.as_array(pointer_to_array, shape=shape)
        return processed

    def HasanFilter_Settings(self, enable):
        self._dll.HasanFilter_Settings(self.obj, enable)

    def HasanFilter(self, buffer):
        size = ct.c_int()
        m = self._dll.Hasan_Filter(self.obj, buffer, ct.byref(size))
        return np.array(m[:size.value])

    def Get_Min(self):
        return self._dll.Get_Min(self.obj)

    def Get_Max(self):
        return self._dll.Get_Max(self.obj)

    def CheckSenxor(self, buffer, headerContent):
        self._dll.validateFrame(self.obj, np.ctypeslib.as_ctypes(buffer.astype(np.uint16)), np.ctypeslib.as_ctypes(headerContent.astype(np.uint16)))

    def loadOptimalSettings(self):
        self.KXMS_Settings(True, 2)
        self.STARK_Settings(True, False, 2, False, True, 10, 20, 80)
        self.Dnlce_settings(True)
        #self.Median_Settings(False, 1)
        #self.HasanFilter_Settings(False)

    def clearSettings(self):
        self.KXMS_Settings(False, 1)
        self.STARK_Settings(False, False, 2, False, True, 10, 20, 80)
        self.Dnlce_settings(False)
        self.Median_Settings(False, 1)
        self.HasanFilter_Settings(False)


