/*
 * XDP DDoS Mitigation Filter - libbpf Compatible Version
 * High-performance packet filtering at NIC driver level
 * 
 * This file is for standalone compilation with clang/libbpf
 * Use xdp_filter.c for BCC runtime compilation
 */

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/in.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

/* Include shared map definitions */
#include "xdp_maps.h"

#ifndef NULL
#define NULL ((void *)0)
#endif

/* Helper to get current time */
static __always_inline __u64 get_time_ns(void) {
    return bpf_ktime_get_ns();
}

/* Update per-CPU statistics */
static __always_inline void update_stats(struct stats *s, __u64 bytes, __u8 protocol, int action) {
    if (!s) return;
    
    __sync_fetch_and_add(&s->total_packets, 1);
    __sync_fetch_and_add(&s->total_bytes, bytes);
    
    if (action == XDP_DROP) {
        __sync_fetch_and_add(&s->dropped_packets, 1);
        __sync_fetch_and_add(&s->dropped_bytes, bytes);
    } else if (action == XDP_PASS) {
        __sync_fetch_and_add(&s->passed_packets, 1);
        __sync_fetch_and_add(&s->passed_bytes, bytes);
    }
    
    switch (protocol) {
        case IPPROTO_TCP:
            __sync_fetch_and_add(&s->tcp_packets, 1);
            break;
        case IPPROTO_UDP:
            __sync_fetch_and_add(&s->udp_packets, 1);
            break;
        case IPPROTO_ICMP:
            __sync_fetch_and_add(&s->icmp_packets, 1);
            break;
        default:
            __sync_fetch_and_add(&s->other_packets, 1);
            break;
    }
}

/* Check if IP is blacklisted */
static __always_inline int is_blacklisted(__u32 src_ip) {
    __u64 *timestamp = bpf_map_lookup_elem(&blacklist_map, &src_ip);
    return (timestamp != NULL);
}

/* Update IP tracking statistics */
/* SYN guard: SYNs per source in a one-second window. syn_count stays a lifetime counter
 * (user space takes deltas); a lifetime limit would cut off any busy client eventually. */
#define SYN_GUARD_PER_SECOND 1000
#define NS_PER_SECOND 1000000000ULL

static __always_inline void count_syn_window(struct ip_stats *st, __u64 now) {
    if (now - st->syn_window_start >= NS_PER_SECOND) {
        st->syn_window_start = now;
        st->syn_in_window = 1;
    } else {
        __sync_fetch_and_add(&st->syn_in_window, 1);
    }
}

static __always_inline int over_syn_guard(struct ip_stats *st, __u64 now) {
    return st && now - st->syn_window_start < NS_PER_SECOND &&
           st->syn_in_window > SYN_GUARD_PER_SECOND;
}

static __always_inline void update_ip_stats(__u32 src_ip, __u64 bytes, __u8 protocol, __u8 tcp_flags) {
    struct ip_stats *ip_stat = bpf_map_lookup_elem(&ip_tracking_map, &src_ip);
    
    if (ip_stat) {
        __sync_fetch_and_add(&ip_stat->packets, 1);
        __sync_fetch_and_add(&ip_stat->bytes, bytes);
        ip_stat->last_seen = get_time_ns();
        
        if (protocol == IPPROTO_TCP && (tcp_flags & 0x12) == 0x02) {
            __sync_fetch_and_add(&ip_stat->syn_count, 1);
            count_syn_window(ip_stat, get_time_ns());
        } else if (protocol == IPPROTO_UDP) {
            __sync_fetch_and_add(&ip_stat->udp_count, 1);
        }
    } else {
        struct ip_stats new_stat = {
            .packets = 1,
            .bytes = bytes,
            .last_seen = get_time_ns(),
            .flow_count = 1,
            .syn_count = (protocol == IPPROTO_TCP && (tcp_flags & 0x12) == 0x02) ? 1 : 0,
            .syn_in_window = (protocol == IPPROTO_TCP && (tcp_flags & 0x12) == 0x02) ? 1 : 0,
            .syn_window_start = get_time_ns(),
            .udp_count = (protocol == IPPROTO_UDP) ? 1 : 0,
        };
        bpf_map_update_elem(&ip_tracking_map, &src_ip, &new_stat, BPF_ANY);
    }
}

/* Update flow statistics */
static __always_inline void update_flow_stats(struct flow_key *key, __u16 ip_len, __u8 tcp_flags) {
    struct flow_stats *flow = bpf_map_lookup_elem(&flow_map, key);
    __u64 now = get_time_ns();
    __u64 len = ip_len;
    if (flow) {
        __u64 last = flow->last_seen;
        __u64 gap = now > last ? (now - last) >> 10 : 0;
        __sync_fetch_and_add(&flow->packets, 1);
        __sync_fetch_and_add(&flow->bytes, len);
        __sync_fetch_and_add(&flow->len_sumsq, len * len);
        __sync_fetch_and_add(&flow->iat_sumsq, gap * gap);
        if (gap > flow->iat_max)
            flow->iat_max = gap > 0xFFFFFFFFULL ? 0xFFFFFFFF : (__u32)gap;
        if (ip_len < flow->len_min)
            flow->len_min = ip_len;
        if (ip_len > flow->len_max)
            flow->len_max = ip_len;
        flow->last_seen = now;
        if (tcp_flags & 0x02) __sync_fetch_and_add(&flow->syn, 1);
        if (tcp_flags & 0x10) __sync_fetch_and_add(&flow->ack, 1);
        if (tcp_flags & 0x01) __sync_fetch_and_add(&flow->fin, 1);
        if (tcp_flags & 0x04) __sync_fetch_and_add(&flow->rst, 1);
        if (tcp_flags & 0x08) __sync_fetch_and_add(&flow->psh, 1);
        flow->flags |= tcp_flags;
    } else {
        struct flow_stats new_flow = {};
        new_flow.packets = 1;
        new_flow.bytes = len;
        new_flow.first_seen = now;
        new_flow.last_seen = now;
        new_flow.len_sumsq = len * len;
        new_flow.len_min = ip_len;
        new_flow.len_max = ip_len;
        new_flow.syn = (tcp_flags & 0x02) ? 1 : 0;
        new_flow.ack = (tcp_flags & 0x10) ? 1 : 0;
        new_flow.fin = (tcp_flags & 0x01) ? 1 : 0;
        new_flow.rst = (tcp_flags & 0x04) ? 1 : 0;
        new_flow.psh = (tcp_flags & 0x08) ? 1 : 0;
        new_flow.flags = tcp_flags;
        bpf_map_update_elem(&flow_map, key, &new_flow, BPF_ANY);
    }
}

/* Main XDP program */
SEC("xdp")
int xdp_ddos_filter(struct xdp_md *ctx) {
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;
    
    struct ethhdr *eth = data;
    struct iphdr *iph;
    struct tcphdr *tcph;
    struct udphdr *udph;
    
    __u32 action = XDP_PASS;
    
    // Get stats structure
    __u32 stats_key = 0;
    struct stats *s = bpf_map_lookup_elem(&stats_map, &stats_key);
    
    // Parse Ethernet header
    if (data + sizeof(*eth) > data_end)
        return XDP_DROP;
    
    // Only process IPv4
    if (eth->h_proto != bpf_htons(ETH_P_IP))
        return XDP_PASS;
    
    // Parse IP header
    iph = data + sizeof(*eth);
    if ((void *)(iph + 1) > data_end)
        return XDP_DROP;
    
    __u32 src_ip = iph->saddr;
    __u32 dst_ip = iph->daddr;
    __u8 protocol = iph->protocol;
    // Byte counters use the IP total length (the unit of the training data)
    __u16 ip_len = bpf_ntohs(iph->tot_len);
    
    // Check blacklist
    if (is_blacklisted(src_ip)) {
        action = XDP_DROP;
        update_stats(s, ip_len, protocol, action);
        return action;
    }
    
    // Parse transport layer and create flow key
    struct flow_key fkey = {
        .src_ip = src_ip,
        .dst_ip = dst_ip,
        .protocol = protocol,
        .src_port = 0,
        .dst_port = 0,
    };
    
    __u8 tcp_flags = 0;
    // Later IP fragments carry no transport header (their ports stay 0, as offline)
    int first_fragment = (iph->frag_off & bpf_htons(0x1FFF)) == 0;
    
    if (protocol == IPPROTO_TCP && first_fragment) {
        tcph = (void *)iph + (iph->ihl * 4);
        if ((void *)(tcph + 1) > data_end)
            goto update_tracking;
        
        fkey.src_port = bpf_ntohs(tcph->source);
        fkey.dst_port = bpf_ntohs(tcph->dest);
        
        // Extract TCP flags
        tcp_flags = ((unsigned char *)tcph)[13];
        
        // Simple SYN flood guard: too many SYNs from this source in the current second
        if ((tcp_flags & 0x12) == 0x02) {
            struct ip_stats *ip_stat = bpf_map_lookup_elem(&ip_tracking_map, &src_ip);
            if (over_syn_guard(ip_stat, get_time_ns())) {
                action = XDP_DROP;
            }
        }
        
    } else if (protocol == IPPROTO_UDP && first_fragment) {
        udph = (void *)iph + (iph->ihl * 4);
        if ((void *)(udph + 1) > data_end)
            goto update_tracking;
        
        fkey.src_port = bpf_ntohs(udph->source);
        fkey.dst_port = bpf_ntohs(udph->dest);
    }
    
update_tracking:
    // Update tracking maps
    update_ip_stats(src_ip, ip_len, protocol, tcp_flags);
    update_flow_stats(&fkey, ip_len, tcp_flags);
    update_stats(s, ip_len, protocol, action);
    
    return action;
}

char _license[] SEC("license") = "GPL";
