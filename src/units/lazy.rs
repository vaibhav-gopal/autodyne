//! [`Lazy`]: a value computed on first use and shared from a `static`, without the standard library
//! (`std::sync::OnceLock` needs it). Threads that race to initialize each compute a value and one
//! of them is kept; the others are dropped, so initializers must be pure. Used for lookup tables
//! (ziggurat layers, interpolation kernels, shared wavetables).
//!
//! tend: no_std

use alloc::boxed::Box;
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

/// A value computed by the first [`get_or_init`](Self::get_or_init) and kept for the program's
/// life (statics are never dropped).
pub(crate) struct Lazy<T> {
    ptr: AtomicPtr<T>,
}

impl<T> Lazy<T> {
    /// Empty: the value is made on first use.
    pub(crate) const fn new() -> Self {
        Lazy { ptr: AtomicPtr::new(ptr::null_mut()) }
    }

    /// The value, made by `init` if no thread has published one yet. Once made, one atomic load
    /// (inlined into hot loops such as the ziggurat's).
    #[inline]
    pub(crate) fn get_or_init(&self, init: impl FnOnce() -> T) -> &T {
        let current = self.ptr.load(Ordering::Acquire);
        if !current.is_null() {
            // SAFETY: a non-null pointer was published by `compare_exchange` in `init_slow` from
            // `Box::into_raw` and is never freed or replaced, so it stays valid for `&self`.
            return unsafe { &*current };
        }
        self.init_slow(init)
    }

    #[cold]
    fn init_slow(&self, init: impl FnOnce() -> T) -> &T {
        let made = Box::into_raw(Box::new(init()));
        match self.ptr.compare_exchange(ptr::null_mut(), made, Ordering::AcqRel, Ordering::Acquire) {
            // SAFETY: just published from `Box::into_raw`; never freed (as above).
            Ok(_) => unsafe { &*made },
            Err(published) => {
                // SAFETY: `made` came from `Box::into_raw` and lost the race, so nothing else has it.
                drop(unsafe { Box::from_raw(made) });
                // SAFETY: published by the winning thread, never freed (as above).
                unsafe { &*published }
            }
        }
    }
}

// SAFETY: the value is shared by reference across threads once published, so it must be Sync (and
// Send, since the publishing thread may not be the one that made it).
unsafe impl<T: Send + Sync> Sync for Lazy<T> {}
// SAFETY: moving the cell moves ownership of the boxed value, which is Send.
unsafe impl<T: Send> Send for Lazy<T> {}

#[cfg(test)]
mod tests {
    use super::Lazy;

    #[test]
    fn computes_once_and_shares_across_threads() {
        static CELL: Lazy<[u64; 4]> = Lazy::new();
        let addresses: alloc::vec::Vec<usize> = (0..8)
            .map(|_| std::thread::spawn(|| CELL.get_or_init(|| [1, 2, 3, 4]) as *const _ as usize))
            .collect::<alloc::vec::Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("thread"))
            .collect();
        assert!(addresses.windows(2).all(|w| w[0] == w[1]), "one value is kept");
        assert_eq!(CELL.get_or_init(|| [0; 4]), &[1, 2, 3, 4]);
    }
}
