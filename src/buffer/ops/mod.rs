/// where buffer operations go
use super::*;
use delegate::delegate;

impl<T: PrimitiveUnit, B: Buffer> BufferOps<B> for Vec<T> {
    fn conv(input: &B, impulse: &B, output: &mut B, state: &mut B) {
        let mut bruh = input.as_ref().iter();
        let huh = bruh.next();
        todo!()
    }
}