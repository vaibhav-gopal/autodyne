/// N-Dimensional Buffer Array

pub struct NDArray<T> {
    data: Vec<T>,       // flat buffer
    shape: Vec<usize>,  // size of each dimension
    strides: Vec<usize> // stride for each dimension
}