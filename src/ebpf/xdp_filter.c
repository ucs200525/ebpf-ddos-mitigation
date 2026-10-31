/*
 * XDP DDoS Mitigation Filter - BCC Compatible Version
 * High-performance packet filtering at NIC driver level
 * 
 * This file is compiled at runtime by BCC (BPF Compiler Collection)
 * It uses BCC-specific macros and syntax
 */

#include <uapi/linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/icmp.h>
#include <linux/in.h>

/* Flow tracking structure */
struct flow_key {
    __u32 src_ip;
    __u32 dst_ip;
    __u16 src_port;
    __u16 dst_port;
    __u8 protocol;
    __u8 pad[3];
};

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
    __u32 flow_count;
    __u32 syn_count;
    __u32 udp_count;
    __u32 syn_in_window;     /* SYNs in the current one-second window (SYN guard) */
    __u64 syn_window_start;
};

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

/* Per-source rate limit (token bucket) installed by user space */
struct rate_limit {
    __u64 rate_pps;      /* packets per second allowed */
    __u64 expires_ns;    /* bpf_ktime_get_ns() deadline; 0 = never */
    __u64 tokens_milli;  /* bucket level, 1000 = one packet */
    __u64 last_ns;       /* last refill */
};

/* Spoofing guard: while active, sources not known from normal times share this
 * budget (per CPU; user space splits the total budget across CPUs) */
struct spoof_guard {
    __u64 rate_pps;
    __u64 expires_ns;
    __u64 tokens_milli;
    __u64 last_ns;
};

/* BPF Maps - BCC style declarations */
/* LRU: during a flood the map keeps the most recent flows instead of refusing new ones;
 * user space reads and empties it every 10 s (batched lookup-and-delete) */
BPF_TABLE("lru_hash", struct flow_key, struct flow_stats, flow_map, 262144);
BPF_HASH(ip_tracking_map, __u32, struct ip_stats, 131072);
BPF_HASH(blacklist_map, __u32, __u64, 10000);          /* value: expiry (ktime ns), 0 = permanent */
BPF_HASH(ratelimit_map, __u32, struct rate_limit, 10000);
BPF_HASH(known_map, __u32, __u8, 131072);               /* sources seen in normal intervals */
BPF_PERCPU_ARRAY(guard_map, struct spoof_guard, 1);
BPF_PERCPU_ARRAY(stats_map, struct stats, 1);

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

/* Take one packet's token from a bucket refilled at rate_pps; 1 = allowed */
static __always_inline int take_token(__u64 *tokens, __u64 *last, __u64 rate, __u64 now) {
    __u64 cap = rate * 1000;
    __u64 t;
    if (now - *last > 1000000000ULL)
        t = cap;                                     /* idle for > 1 s: bucket is full */
    else
        t = *tokens + (now - *last) * rate / 1000000;  /* ns x pkt/s -> milli-packets */
    if (t > cap)
        t = cap;
    *last = now;
    if (t >= 1000) {
        *tokens = t - 1000;
        return 1;
    }
    *tokens = t;
    return 0;
}

/* Blocked until the entry expires; expired entries are removed (blocks heal) */
static __always_inline int is_blacklisted(__u32 src_ip, __u64 now) {
    __u64 *expires = blacklist_map.lookup(&src_ip);
    if (!expires)
        return 0;
    if (*expires == 0 || now < *expires)
        return 1;
    blacklist_map.delete(&src_ip);
    return 0;
}

/* Source over its rate limit (the "slow down" action) */
static __always_inline int over_rate_limit(__u32 src_ip, __u64 now) {
    struct rate_limit *rl = ratelimit_map.lookup(&src_ip);
    if (!rl)
        return 0;
    if (rl->expires_ns && now >= rl->expires_ns) {
        ratelimit_map.delete(&src_ip);
        return 0;
    }
    return !take_token(&rl->tokens_milli, &rl->last_ns, rl->rate_pps, now);
}

/* Spoofing guard: unknown sources share a budget; known sources are not affected */
static __always_inline int spoof_guard_drop(__u32 src_ip, __u64 now) {
    __u32 key = 0;
    struct spoof_guard *g = guard_map.lookup(&key);
    if (!g || now >= g->expires_ns)
        return 0;
    if (known_map.lookup(&src_ip))
        return 0;
    return !take_token(&g->tokens_milli, &g->last_ns, g->rate_pps, now);
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
    struct ip_stats *ip_stat = ip_tracking_map.lookup(&src_ip);
    
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
        struct ip_stats new_stat = {};
        new_stat.packets = 1;
        new_stat.bytes = bytes;
        new_stat.last_seen = get_time_ns();
        new_stat.flow_count = 1;
        new_stat.syn_count = (protocol == IPPROTO_TCP && (tcp_flags & 0x12) == 0x02) ? 1 : 0;
        new_stat.syn_in_window = new_stat.syn_count;
        new_stat.syn_window_start = get_time_ns();
        new_stat.udp_count = (protocol == IPPROTO_UDP) ? 1 : 0;
        ip_tracking_map.update(&src_ip, &new_stat);
    }
}

/* Update flow statistics */
static __always_inline void update_flow_stats(struct flow_key *key, __u16 ip_len, __u8 tcp_flags) {
    struct flow_stats *flow = flow_map.lookup(key);
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
        flow_map.update(key, &new_flow);
    }
}

/* Main XDP program */
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
    struct stats *s = stats_map.lookup(&stats_key);
    
    // Parse Ethernet header
    if (data + sizeof(*eth) > data_end)
        return XDP_DROP;
    
    // Only process IPv4
    if (eth->h_proto != htons(ETH_P_IP))
        return XDP_PASS;
    
    // Parse IP header
    iph = data + sizeof(*eth);
    if ((void *)(iph + 1) > data_end)
        return XDP_DROP;
    
    __u32 src_ip = iph->saddr;
    __u32 dst_ip = iph->daddr;
    __u8 protocol = iph->protocol;
    // Byte counters use the IP total length (the unit of the training data: CIC and MAWI
    // captures), not the frame length
    __u16 ip_len = ntohs(iph->tot_len);
    
    // Parse transport layer and create flow key
    struct flow_key fkey = {};
    fkey.src_ip = src_ip;
    fkey.dst_ip = dst_ip;
    fkey.protocol = protocol;
    fkey.src_port = 0;
    fkey.dst_port = 0;
    
    __u8 tcp_flags = 0;
    // Later IP fragments carry no transport header (their ports stay 0, as offline)
    int first_fragment = (iph->frag_off & htons(0x1FFF)) == 0;
    
    if (protocol == IPPROTO_TCP && first_fragment) {
        tcph = (void *)iph + (iph->ihl * 4);
        if ((void *)(tcph + 1) > data_end)
            goto update_tracking;
        
        fkey.src_port = ntohs(tcph->source);
        fkey.dst_port = ntohs(tcph->dest);
        
        // Extract TCP flags
        tcp_flags = ((unsigned char *)tcph)[13];
        
    } else if (protocol == IPPROTO_UDP && first_fragment) {
        udph = (void *)iph + (iph->ihl * 4);
        if ((void *)(udph + 1) > data_end)
            goto update_tracking;
        
        fkey.src_port = ntohs(udph->source);
        fkey.dst_port = ntohs(udph->dest);
    }
    
update_tracking:
    // Update tracking maps regardless of blacklist status
    update_ip_stats(src_ip, ip_len, protocol, tcp_flags);
    update_flow_stats(&fkey, ip_len, tcp_flags);

process_action:
    // Mitigation AFTER updating stats so blocked sources still show in Top IPs.
    // Order: block (TTL) -> per-source rate limit -> spoofing guard -> SYN guard
    __u64 now = get_time_ns();
    if (is_blacklisted(src_ip, now)) {
        action = XDP_DROP;
    }
    else if (over_rate_limit(src_ip, now)) {
        action = XDP_DROP;
    }
    else if (spoof_guard_drop(src_ip, now)) {
        action = XDP_DROP;
    }
    else if (protocol == IPPROTO_TCP && (tcp_flags & 0x12) == 0x02) {
        // Simple SYN flood guard: too many SYNs from this source in the current second
        struct ip_stats *ip_stat = ip_tracking_map.lookup(&src_ip);
        if (over_syn_guard(ip_stat, now)) {
            action = XDP_DROP;
        }
    }

    update_stats(s, ip_len, protocol, action);
    return action;
}
