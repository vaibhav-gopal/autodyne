use std::cmp::Ordering;
use std::fmt::Formatter;
use super::*;
use super::markers::*;

use delegate::delegate;

mod real_fixed_new;

// don't use newtype just use extension traits!!
// avoids inheritance (via reducing code duplication ; avoid the situation entirely)
