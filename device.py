# device.py
from enum import Enum, auto

class DeviceType(Enum):
    CPU = auto()
    CUDA = auto()

class Device:
    def __init__(self, device_type=DeviceType.CPU, device_id=0):
        self.device_type = device_type
        self.device_id = device_id
        
        # Check if CUDA is available (stub implementation)
        if device_type == DeviceType.CUDA:
            try:
                # This would be an actual check in a real implementation
                cuda_available = False  # Placeholder
                if not cuda_available:
                    print("CUDA not available. Falling back to CPU.")
                    self.device_type = DeviceType.CPU
            except:
                print("Error checking CUDA availability. Falling back to CPU.")
                self.device_type = DeviceType.CPU
    
    def __str__(self):
        if self.device_type == DeviceType.CPU:
            return "cpu"
        else:
            return f"cuda:{self.device_id}"
    
    def __eq__(self, other):
        if not isinstance(other, Device):
            return False
        return (self.device_type == other.device_type and 
                self.device_id == other.device_id)

# Default devices
cpu = Device(DeviceType.CPU)
cuda = Device(DeviceType.CUDA) if DeviceType.CUDA else cpu

def get_device(device_str=None):
    """Parse a device string to a Device object."""
    if device_str is None:
        return cpu
    if isinstance(device_str, Device):
        return device_str
    
    if device_str.lower() == "cpu":
        return cpu
    elif device_str.lower().startswith("cuda"):
        try:
            device_id = int(device_str.split(":")[-1]) if ":" in device_str else 0
            return Device(DeviceType.CUDA, device_id)
        except:
            print("Invalid CUDA device specification. Falling back to CPU.")
            return cpu
    else:
        print(f"Unknown device '{device_str}'. Falling back to CPU.")
        return cpu