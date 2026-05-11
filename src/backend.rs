use crate::compute_graph::ComputeGraph;
use crate::defs::Status;
use crate::memory_manager::MemoryBlock;
use crate::tensor::Tensor;

pub enum BackendDeviceType {
    Cpu,
    Gpu,
    ACCEL,
}

pub struct BackendDeviceCaps {
    pub aysnc: bool,
    pub host_buffer: bool,
    pub buffer_from_host_ptr: bool,
    pub events: bool,
}

pub struct BackendDeviceProps {
    pub name: &'static str,
    pub description: &'static str,
    pub memory_free: usize,
    pub memory_total: usize,
    pub device_type: BackendDeviceType,
    pub caps: BackendDeviceCaps,
}

// BackendBuffer = ggml_backend_buffer_type + ggml_backend_buffer
pub trait BackendBuffer: Send + Sync {
    pub fn as_ptr(&self) -> *mut u8;

    pub fn device(&self) -> Box<dyn BackendDevice>;

    pub fn get_base(&self) -> MemoryBlock;

    pub fn clear(&self, value: u8);

    pub fn reset(&self);

    pub fn allocate_tensor(&self, tensor: Tensor, memory_block: MemoryBlock) {}

    pub fn init_tensor(&self, tensor: Tensor);

    pub fn memset_tensor(&self, tensor: Tensor, value: u8, offset: usize, size: usize);

    pub fn set_tensor(&self, tensor: Tensor, data: *mut u8, offset: usize, size: usize);

    pub fn get_tensor(&self, tensor: Tensor, data: *mut u8, offset: usize, size: usize);

    pub fn copy_tensor(&self, src: Tensor, dst: Tensor);

    pub fn buffer_allocator(&self) -> Box<dyn BackendBufferAllocator>;
}

pub trait BackendBufferAllocator {
    pub fn allocate_buffer(&self, size: usize) -> Box<dyn BackendBuffer>;

    pub fn alignment(&self) -> usize {
        std::mem::align_of::<usize>()
    }

    pub fn max_size(&self) -> usize {
        usize::MAX
    }

    pub fn alloc_size(&self, tensor: Tensor) -> usize;

    pub fn size(&self) -> usize;
}

pub trait Backend {
    type Device: BackendDevice;

    pub fn get_name(&self) -> &str;

    fn synchronize(&self);

    fn graph_compute(&self, graph: &mut ComputeGraph) -> Status;

    fn memcpy_async(&self, dst: *mut u8, src: *const u8, size: usize);

    fn set_tensor_async(&self, tensor: Tensor, data: *mut u8, offset: usize, size: usize);

    fn get_tensor_async(&self, tensor: Tensor, data: *mut u8, offset: usize, size: usize);

    fn copy_tensor_async(&self, src: Tensor, dst: Tensor);
}
pub trait BackendDevice: Send + Sync {
    fn name(&self) -> &str;

    fn memory(&self) -> (usize, usize);

    fn description(&self) -> &str;

    fn device_type(&self) -> BackendDeviceType;

    fn props(&self) -> BackendDeviceProps;

    fn init(&self, params: *mut u8);

    fn supports_op(&self, tensor: Tensor) -> bool;

    fn supports_buffer_allocator(&self, buffer_allocator: &Box<dyn BackendBufferAllocator>)
        -> bool;

    fn offload_op(&self, tensor: Tensor) -> bool;
}

pub trait BackendRegister: Send + Sync {
    fn name(&self) -> &str;

    fn device_count(&self) -> usize;

    fn device(&self, index: usize) -> Box<dyn BackendDevice>;
}
