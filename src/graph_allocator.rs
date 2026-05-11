use crate::backend::{Backend, BackendBuffer, BackendBufferAllocator};

struct GraphAllocator<B: BackendBuffer> {
    buffer: B,
    alignment: usize,
    offset: usze,
}

impl<B: BackendBuffer> GraphAllocator<B> {
    pub fn new(buffer: B) -> Self {
        let alignment = buffer.buffer_allocator().alignment();
        Self { buffer, alignment, offset: 0 }
    }

    pub fn allocate(&mut self, tensor: Tensor) {
        let size = self.buffer.buffer_allocator().alloc_size(tensor);

        if self.offset + size > self.buffer.buffer_allocator().size() {
            tracing::error!(
                "GraphAllocator: Out of memory. Requested size: {}, available size: {}",
                size,
                self.buffer.buffer_allocator().size() - self.offset
            );
        }
    }
}
