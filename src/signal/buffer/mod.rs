#[repr(transparent)]
pub struct Buffer<T> {
    data: Vec<T>
}