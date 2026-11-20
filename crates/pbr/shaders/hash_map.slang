module hash_map;

// A hash map from 64-bit keys to slots, shared by every thread of a dispatch.
//
//     HashMap map = HashMap(entries, capacity);
//     if (let slot = map.findOrInsert(key)) {
//         InterlockedAdd(sampleCounts[slot], 1u);
//     }
//
// It maps keys to slot indices, not to values: callers keep their values in
// buffers of their own, one element per slot, and index them with the slot that
// `find` or `findOrInsert` returns. That keeps value types out of the map and
// lets each value buffer have the access it needs (atomic adds for one, packed
// reads for another).
//
// The map is one buffer of `capacity` uints, zeroed before first use. It doesn't
// store keys: each slot holds its key's tag, a nonzero 32-bit hash, or 0 while
// the slot is empty. Inserts claim a slot with a 32-bit compare-exchange on it,
// so the map needs no 64-bit atomics, which Apple GPUs lack.
//
// A key is stored in one of the `kBucketSize` slots from its base slot, its
// bucket. Buckets overlap and stop at the end of the table rather than wrap, so
// `capacity` must be at least `kBucketSize`. A key whose bucket is full can't be
// inserted.
//
// As the map keeps only tags, two keys whose tags match and whose buckets
// overlap share a slot. The tag and the base slot are independent hashes of the
// key, so a given pair of keys shares one about 2 * kBucketSize / 2^32 /
// capacity of the time: with a million slots, about once in 2^47, so not
// likely even once with all of them in use.
//
// `find` and `findOrInsert` can run at the same time from any number of threads.
// `remove` can't run alongside them: call it from its own pass, with a barrier
// between it and the passes that look keys up.
public struct HashMap {
    public static const uint kBucketSize = 16;

    RWStructuredBuffer<uint> entries;
    public uint capacity;

    public __init(RWStructuredBuffer<uint> entries, uint capacity) {
        this.entries = entries;
        this.capacity = capacity;
    }

    // The slot that holds `key`, if any.
    public Optional<uint> find(uint64_t key) {
        uint base = baseSlot(key);
        uint t = tag(key);
        [loop]
        for (uint i = 0; i < kBucketSize; i++) {
            if (entries[base + i] == t) {
                return base + i;
            }
        }
        return none;
    }

    // The slot that holds `key`, claiming an empty one for it if none does.
    // None when the key's bucket is full. Threads inserting the same key at the
    // same time get the same slot.
    public Optional<uint> findOrInsert(uint64_t key) {
        uint base = baseSlot(key);
        uint t = tag(key);

        // Search the whole bucket before claiming a slot: `remove` can empty a
        // slot ahead of the key's, and claiming that slot would store the key
        // twice.
        uint firstEmpty = kBucketSize;
        [loop]
        for (uint i = 0; i < kBucketSize; i++) {
            uint stored = entries[base + i];
            if (stored == t) {
                return base + i;
            }
            if (stored == 0 && firstEmpty == kBucketSize) {
                firstEmpty = i;
            }
        }

        // Claim the first empty slot. A thread inserting another key can take
        // it first, so go on down the bucket until a claim succeeds or finds a
        // slot another thread claimed for this key.
        [loop]
        for (uint i = firstEmpty; i < kBucketSize; i++) {
            uint slot = base + i;
            uint prev;
            InterlockedCompareExchange(entries[slot], 0u, t, prev);
            if (prev == 0 || prev == t) {
                return slot;
            }
        }
        return none;
    }

    // Whether a key is stored in `slot`.
    public bool isOccupied(uint slot) {
        return entries[slot] != 0;
    }

    // Empties `slot`. Can't run alongside `find` or `findOrInsert`.
    public void remove(uint slot) {
        entries[slot] = 0;
    }

    // How far into its bucket `key` is stored: 0 at its base slot, `kBucketSize`
    // when it isn't in the map. For debug views of hash collisions.
    public uint probeLength(uint64_t key) {
        uint base = baseSlot(key);
        if (let slot = find(key)) {
            return slot - base;
        }
        return kBucketSize;
    }

    // The 32-bit hash of `key` that picks its base slot.
    public static uint hash(uint64_t key) {
        return fmix32(uint(key) ^ fmix32(uint(key >> 32)));
    }

    uint baseSlot(uint64_t key) {
        return hash(key) % (capacity - kBucketSize + 1);
    }

    // The 32-bit hash of `key` stored in its slot. Independent of `hash`, so
    // that keys with nearby base slots rarely share it. Never 0, which marks an
    // empty slot.
    static uint tag(uint64_t key) {
        return max(fmix32(uint(key >> 32) ^ fmix32(uint(key) + 0x9e3779b9u)), 1u);
    }
}

// MurmurHash3's 32-bit finalizer: a bijection that spreads every input bit over
// every output bit.
uint fmix32(uint h) {
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}
