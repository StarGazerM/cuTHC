
use crate::rmm;

#[test]
fn test_init_rmm() {
    rmm::memory_pool_init();
}

#[test]
fn test_pinned_vec() {
    // rmm::memory_pool_init();
    let mut vec = rmm::PinnedVec::new(10);
    assert_eq!(vec.size(), 10);
    vec.resize(20);
    assert_eq!(vec.size(), 20);
    vec.resize(30);
    assert_eq!(vec.size(), 30);
    for i in 0..10 {
        vec.set(i, 1);
    }
    for i in 0..10 {
        assert_eq!(vec.get(i), 1);
    }
}

#[test]
fn test_device_vec() {
    // rmm::memory_pool_init();
    let mut vec = rmm::DeviceVec::new(10);
    assert_eq!(vec.size(), 10);
    vec.resize(20);
    assert_eq!(vec.size(), 20);
    vec.resize(30);
    assert_eq!(vec.size(), 30);
    for i in 0..10 {
        vec.set(i, 1);
    }
    for i in 0..10 {
        assert_eq!(vec.get(i), 1);
    }
}
