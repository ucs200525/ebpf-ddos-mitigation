/*
 * eBPF Map Definitions for DDoS Mitigation System
 * Shared between kernel XDP program and user-space control plane
 */

#ifndef __XDP_MAPS_H__
#define __XDP_MAPS_H__

#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

/* Maximum values */
#define MAX_FLOWS 262144
#define MAX_IPS 131072
#define MAX_BLACKLIST 10000
#define MAX_CPUS 128

/* Flow tracking structure */
struct flow_key {
    __u32 src_ip;
    __u32 dst_ip;
    __u16 src_port;
    __u16 dst_port;
    __u8 protocol;
    __u8 pad[3];
} __attribute__((packed));

/* Per-flow counters for the level-3 flow classifier (src/ml/flow_features.py).
 * User space reads and clears the map every 10 s, so one entry = one flow in one
 * window. Lengths are IP total lengths; times are bpf_ktime ns; inter-arrival
 * times are in units of 1024 ns (shift instead of division). */
struct flow_stats {
    __u64 packets;
    __u64 bytes;          /* sum of IP lengths */
    __u64 first_seen;
    __u64 last_seen;
    __u64 len_sumsq;      /* sum of squared IP lengths */
    __u64 iat_sumsq;      /* sum of squared inter-arrival times (1024-ns units) */
    __u32 iat_max;        /* largest inter-arrival time (1024-ns units, saturating) */
    __u16 len_min;
    __u16 len_max;
    __u32 syn;            /* packets with SYN set */
    __u32 ack;
    __u32 fin;
    __u32 rst;
    __u32 psh;
    __u8 flags;           /* TCP flags OR'd together */
    __u8 pad[3];
};

/* Per-IP tracking structure */
struct ip_stats {
    __u64 packets;
    __u64 bytes;
    __u64 last_seen;
    __u32 flow_count;  // Number of flows from this IP
    __u32 syn_count;   // SYN packets (for SYN flood detection) - must be 32-bit for atomic ops
    __u32 udp_count;   // UDP packets - must be 32-bit for atomic ops
    __u32 syn_in_window;     // SYNs in the current one-second window (SYN guard)
    __u64 syn_window_start;
};

/* Attack signature structure */
struct attack_signature {
    __u32 enabled;
    __u32 signature_type;  // 1=IP, 2=Port, 3=Protocol
    __u32 value;           // IP/Port/Protocol value
    __u64 blocked_packets;
    __u64 blocked_bytes;
} __attribute__((packed));

/* Statistics structure (per-CPU) */
struct stats {
    __u64 total_packets;
    __u64 total_bytes;
    __u64 dropped_packets;
    __u64 dropped_bytes;
    __u64 passed_packets;
    __u64 passed_bytes;
    __u64 tcp_packets;
    __u64 udp_packets;
    __u64 icmp_packets;
    __u64 other_packets;
};

/*
 * BPF Maps
 */

/* Flow statistics - LRU hash map for automatic eviction */
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, MAX_FLOWS);
    __type(key, struct flow_key);
    __type(value, struct flow_stats);
} flow_map SEC(".maps");

/* Per-source IP tracking */
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, MAX_IPS);
    __type(key, __u32);  // Source IP
    __type(value, struct ip_stats);
} ip_tracking_map SEC(".maps");

/* Blacklist - blocked IPs */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_BLACKLIST);
    __type(key, __u32);  // IP address
    __type(value, __u64);  // Timestamp when blacklisted
} blacklist_map SEC(".maps");

/* Attack signatures */
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1000);
    __type(key, __u32);  // Signature ID
    __type(value, struct attack_signature);
} signature_map SEC(".maps");

/* Per-CPU statistics array for lock-free updates */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct stats);
} stats_map SEC(".maps");

#endif /* __XDP_MAPS_H__ */
