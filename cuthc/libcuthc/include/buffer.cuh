
#pragma once

#include <thrust/device_vector.h>

// Buffer class, a static buffer that can be used to store data on the device
// internally, it uses a thrust::device_vector to store the data
// a singlton class
template <typename T>
class CuTHCBuffer {
private:
    // the data vector
    thrust::device_vector<T> m_data;

    // private constructor
    CuTHCBuffer() {}

    // private copy constructor
    CuTHCBuffer(const CuTHCBuffer&) {}

    // private assignment operator
    CuTHCBuffer& operator=(const CuTHCBuffer&) {}

public:
    // get the buffer instance
    static CuTHCBuffer<T>& getInstance() {
        static CuTHCBuffer<T> instance;
        return instance;
    }

    // get the device vector
    thrust::device_vector<T>& getDeviceVector() {
        return m_data;
    }

    // get the pointer to the data
    T* getPointer() {
        return thrust::raw_pointer_cast(m_data.data());
    }

    // get the size of the buffer
    size_t size() {
        return m_data.size();
    }

    // resize the buffer
    void resize(size_t size) {
        m_data.resize(size);
    }

    // clear the buffer
    void clear() {
        m_data.clear();
    }
};

