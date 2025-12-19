use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

#[derive(Eq, PartialEq)]
struct HeapItem {
    freq: i32,
    b1: Vec<u8>,
    b2: Vec<u8>,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.freq.cmp(&other.freq) {
            Ordering::Equal => match self.b1.cmp(&other.b1) {
                Ordering::Equal => self.b2.cmp(&other.b2),
                o => o,
            },
            o => o,
        }
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[pyclass]
pub struct RustMaxHeap {
    freqs: HashMap<(Vec<u8>, Vec<u8>), i32>,
    heap: BinaryHeap<HeapItem>,
}

#[pymethods]
impl RustMaxHeap {
    #[new]
    pub fn new() -> Self {
        Self {
            freqs: HashMap::new(),
            heap: BinaryHeap::new(),
        }
    }

    pub fn clear(&mut self) {
        self.freqs.clear();
        self.heap.clear();
    }

    pub fn reset(&mut self) {
        self.clear();
    }

    pub fn add(&mut self, key: (Vec<u8>, Vec<u8>), count: i32) {
        let total = self.freqs.entry(key.clone()).or_insert(0);
        *total += count;

        if *total <= 0 {
            self.freqs.remove(&key);
        } else {
            self.heap.push(HeapItem {
                freq: *total,
                b1: key.0,
                b2: key.1,
            });
        }
    }

    pub fn remove(&mut self, key: (Vec<u8>, Vec<u8>)) {
        if let Some(total) = self.freqs.get_mut(&key) {
            self.freqs.remove(&key);
        }
    }

    pub fn find_max<'py>(
        &mut self,
        py: Python<'py>,
    ) -> Option<((&'py PyBytes, &'py PyBytes), i32)> {
        while let Some(item) = self.heap.pop() {
            let key = (item.b1.clone(), item.b2.clone());
            if let Some(&actual) = self.freqs.get(&key) {
                if actual == item.freq {
                    let py_b1 = PyBytes::new(py, &item.b1);
                    let py_b2 = PyBytes::new(py, &item.b2);
                    return Some(((py_b1, py_b2), actual));
                }
            }
        }
        None
    }

    pub fn get_freq(&self, key: (Vec<u8>, Vec<u8>)) -> i32 {
        *self.freqs.get(&key).unwrap_or(&0)
    }
}

#[pymodule]
fn rust_max_heap(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustMaxHeap>()?;
    Ok(())
}
