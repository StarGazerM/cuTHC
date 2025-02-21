
#include <cuco/static_map.cuh>
#include <cstdint>

// using cuthc_static_map_t = cuco::static_map <
//                            int,
//       uint64_t, cuco::extent<std::size_t>,
//       cuda::thread_scope_device, thrust::equal_to<int>,
//       cuco::linear_probing<4, cuco::default_hash_function<int>>;
// using cuthc_static_map_pair_t = cuco::pair<int, uint64_t>;

// void *cuthc_mk_static_map(size_t capacity)
// {
//     return new cuthc_static_map_t(capacity);
// }

// size_t cuthc_static_map_size(void *map)
// {
//     cuthc_static_map_t *map_ = static_cast<cuthc_static_map_t *>(map);
//     return map_->size();
// }

// size_t cuthc_static_map_clear(void *map)
// {
//     cuthc_static_map_t *map_ = static_cast<cuthc_static_map_t *>(map);
//     return map_->clear();
// }

// void cuthc_static_map_find(void *map, int *key_start, int key_end, uint64_t *value)
// {
//     cuthc_static_map_t *map_ = static_cast<cuthc_static_map_t *>(map);
//     map_->find(cuthc_static_map_pair_t(*key_start), cuthc_static_map_pair_t(key_end), value);
// }

// void cuthc_static_map_contains(void *map, int *key_start, int key_end, bool *contains)
// {
//     cuthc_static_map_t *map_ = static_cast<cuthc_static_map_t *>(map);
//     *contains = map_->contains(cuthc_static_map_pair_t(*key_start, key_end));
// }
