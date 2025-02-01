pub struct AudioBuffer {
    data: Vec<f32>
}

impl AudioBuffer {
    pub fn new() -> Self {
        Self {
            data: Vec::new()
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn zero(&mut self) {
        for s in &mut self.data {
            *s = 0.0;
        }
    }
}


pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

