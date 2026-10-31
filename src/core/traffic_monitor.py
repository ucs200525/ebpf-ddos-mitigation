"""
Traffic Monitor - User-space controller for eBPF/XDP programs
Loads XDP programs and reads statistics from eBPF maps
"""

import ipaddress
import json
import platform
import logging
import struct
import socket
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from src.config import BLACKLIST_FILE, DATA_DIR

# Module logger (logging.warning() on the root logger at import time would
# auto-configure logging before main.py sets up its file handler)
logger = logging.getLogger(__name__)

# Platform-specific imports
PLATFORM = platform.system().lower()

if PLATFORM == 'linux':
    try:
        from bcc import BPF
        BCC_AVAILABLE = True
    except ImportError:
        BCC_AVAILABLE = False
        logger.warning("BCC not available. Install with: sudo apt-get install python3-bpfcc")
else:  # Windows
    # Microsoft eBPF bindings (placeholder - will need actual implementation)
    BCC_AVAILABLE = False
    logger.info("Microsoft eBPF support not yet implemented; simulation mode will be used")


def validate_blockable_ip(ip_address: str) -> Optional[str]:
    """
    Return the normalized IPv4 address if it is safe to blacklist, else None.
    Rejects malformed input and addresses that must never be blocked
    (loopback, unspecified, multicast, broadcast).
    """
    try:
        addr = ipaddress.IPv4Address(str(ip_address).strip())
    except (ipaddress.AddressValueError, ValueError):
        return None
    if addr.is_loopback or addr.is_unspecified or addr.is_multicast or addr == ipaddress.IPv4Address('255.255.255.255'):
        return None
    return str(addr)


def _ktime_ns() -> int:
    """Clock of bpf_ktime_get_ns() (CLOCK_MONOTONIC), for expiry times in kernel maps"""
    return time.monotonic_ns()


def _active(value, now_ns: int) -> bool:
    """A blacklist map value is an expiry time; 0 means permanent"""
    expires = getattr(value, 'value', value)
    return expires == 0 or expires > now_ns


def _np_dtype(cstruct) -> np.dtype:
    """numpy view of a BCC ctypes struct (array members such as padding are skipped)"""
    import ctypes as ct
    names, formats, offsets = [], [], []
    for name, ctype in cstruct._fields_:
        if issubclass(ctype, ct.Array):
            continue
        names.append(name)
        formats.append(np.dtype(ctype))
        offsets.append(getattr(cstruct, name).offset)
    return np.dtype({'names': names, 'formats': formats, 'offsets': offsets,
                     'itemsize': ct.sizeof(cstruct)})


def _drain_map(table, delete: bool = True):
    """
    All entries of a BCC hash map as two numpy record arrays (keys, values), read with
    the kernel's batch operation (BPF_MAP_LOOKUP[_AND_DELETE]_BATCH): a few syscalls for the
    whole map, and with delete=True nothing is lost between reading and clearing.
    """
    import ctypes as ct
    import errno
    import os
    from bcc.libbcc import lib
    fn = lib.bpf_lookup_and_delete_batch if delete else lib.bpf_lookup_batch
    n_max = table.max_entries
    keys = (table.Key * n_max)()
    vals = (table.Leaf * n_max)()
    in_b, out_b, cnt = ct.c_uint32(0), ct.c_uint32(0), ct.c_uint32(0)
    total, first = 0, True
    while total < n_max:
        cnt.value = n_max - total
        res = fn(table.map_fd, None if first else ct.byref(in_b), ct.byref(out_b),
                 ct.byref(keys, ct.sizeof(table.Key) * total), ct.byref(vals, ct.sizeof(table.Leaf) * total),
                 ct.byref(cnt))
        err = ct.get_errno()
        total += cnt.value
        if res != 0:
            if err != errno.ENOENT:          # ENOENT = no more entries
                raise OSError(err, os.strerror(err))
            break
        if cnt.value == 0:
            break
        in_b.value, first = out_b.value, False
    k = np.frombuffer(keys, dtype=_np_dtype(table.Key))[:total]
    v = np.frombuffer(vals, dtype=_np_dtype(table.Leaf))[:total]
    return k, v


class TrafficMonitor:
    """Monitor traffic using eBPF/XDP programs"""

    def __init__(self, interface: str, xdp_mode: str = 'native',
                 blacklist_file: Optional[str] = None):
        """
        Initialize traffic monitor

        Args:
            interface: Network interface name (e.g., 'eth0', 'Ethernet')
            xdp_mode: XDP mode - 'native', 'generic', or 'offload'
        """
        self.interface = interface
        self.xdp_mode = xdp_mode
        self.bpf = None
        self.loaded = False
        self.blacklist_file = blacklist_file or BLACKLIST_FILE
        # Simulation-mode copies of the kernel mitigation maps. Expiry times are in
        # the sample clock (sim_now, set by the system each interval); inf = permanent
        self._sim_blacklist: Dict[str, float] = {}
        self._sim_limits: Dict[str, Tuple[float, float]] = {}   # ip -> (pps, expiry)
        self._sim_known = set()
        self._sim_guard: Optional[Tuple[float, float]] = None   # (expiry, total pps)
        self.sim_now = 0.0
        self._permanent = set()     # permanent blocks are the only ones saved to disk
        # Previous cumulative per-IP counters, for per-interval deltas
        self._prev_ip_counters: Dict[str, Tuple[int, int, int, int]] = {}

        self.simulation_mode = not BCC_AVAILABLE

        if self.simulation_mode:
            logger.warning("Running in SIMULATION MODE (eBPF not available)")

        # Ensure data directory exists
        Path(self.blacklist_file).parent.mkdir(parents=True, exist_ok=True)

    def load_xdp_program(self, program_path: str) -> bool:
        """
        Load XDP program onto the network interface

        Args:
            program_path: Path to BPF C source file (not compiled)

        Returns:
            True if loaded successfully
        """
        if self.simulation_mode:
            self.loaded = True
            logger.info(f"[SIM] Dashboard available on {self.interface}")
            return True

        try:
            # Change path from .o to .c
            if program_path.endswith('.o'):
                c_source_path = program_path.replace('.o', '.c')
            else:
                c_source_path = program_path

            # Load the BPF program from C source
            with open(c_source_path, 'r') as f:
                bpf_source = f.read()

            self.bpf = BPF(text=bpf_source)

            # Get the XDP function
            fn = self.bpf.load_func("xdp_ddos_filter", BPF.XDP)

            # Attach to interface
            flags = 0
            if self.xdp_mode == 'generic':
                flags = 2  # XDP_FLAGS_SKB_MODE
            elif self.xdp_mode == 'native':
                flags = 1  # XDP_FLAGS_DRV_MODE
            elif self.xdp_mode == 'offload':
                flags = 4  # XDP_FLAGS_HW_MODE

            self.bpf.attach_xdp(self.interface, fn, flags)
            self.loaded = True

            # Load existing blacklist into eBPF map
            self._load_blacklist_from_disk()

            logger.info(f"XDP program loaded on {self.interface} (mode: {self.xdp_mode})")
            return True

        except Exception as e:
            logger.error(f"Failed to load XDP program: {e}")
            return False

    def unload_xdp_program(self):
        """Unload XDP program from interface"""
        # "is not None": BCC's BPF object defines __len__ (cached tables), so it is
        # falsy while loaded when maps are only read through get_table()
        if self.bpf is not None and self.loaded:
            try:
                self.bpf.remove_xdp(self.interface)
                logger.info(f"XDP program unloaded from {self.interface}")
            except Exception as e:
                logger.error(f"Failed to unload XDP program: {e}")
            finally:
                self.loaded = False

    def get_statistics(self) -> Dict:
        """
        Read overall statistics from eBPF maps

        Returns:
            Dictionary with aggregated statistics
        """
        if not self.loaded or self.simulation_mode:
            return {}

        try:
            stats_map = self.bpf.get_table("stats_map")

            # Aggregate per-CPU stats
            total_stats = {
                'total_packets': 0,
                'total_bytes': 0,
                'dropped_packets': 0,
                'dropped_bytes': 0,
                'passed_packets': 0,
                'passed_bytes': 0,
                'tcp_packets': 0,
                'udp_packets': 0,
                'icmp_packets': 0,
                'other_packets': 0,
            }

            # BCC returns a list of per-CPU values for PERCPU_ARRAY
            # We need to sum across all CPUs
            key = stats_map.Key(0)
            per_cpu_stats = stats_map[key]

            # per_cpu_stats is a list of stats structures, one per CPU
            for cpu_stats in per_cpu_stats:
                for field in total_stats.keys():
                    total_stats[field] += getattr(cpu_stats, field, 0)

            return total_stats

        except Exception as e:
            logger.error(f"Failed to read statistics: {e}")
            return {}

    def get_ip_statistics(self, limit: int = 100) -> List[Dict]:
        """
        Get per-IP statistics

        Args:
            limit: Maximum number of IPs to return

        Returns:
            List of dictionaries with IP statistics
        """
        if not self.loaded or self.simulation_mode:
            return []

        try:
            ip_map = self.bpf.get_table("ip_tracking_map")
            ip_stats = []


            # Fetch ALL IPs first (up to a reasonable safety max), then sort
            # Iterate through the map without an early break to find the true top talkers
            # BPF hash maps are not ordered, so the first 100 might be low-traffic IPs
            MAX_scan = 10000
            count = 0

            for key, value in ip_map.items():
                if count >= MAX_scan:
                    break

                # key is just a c_uint32 (the IP address)
                # In different BCC versions, key might be 'c_uint', 'c_int', or just an int.
                # safely cast it
                try:
                    ip_int = key.value
                except AttributeError:
                    ip_int = key

                ip_addr = socket.inet_ntoa(struct.pack('I', ip_int))

                ip_stats.append({
                    'ip': ip_addr,
                    'packets': value.packets,
                    'bytes': value.bytes,
                    'flow_count': value.flow_count,
                    'syn_count': value.syn_count,
                    'udp_count': value.udp_count,
                    'last_seen': value.last_seen,
                })
                count += 1

            # Sort by packet count (descending)
            ip_stats.sort(key=lambda x: x['packets'], reverse=True)

            # Return top 'limit'
            return ip_stats[:limit]

        except Exception as e:
            logger.error(f"Failed to read IP statistics: {e}")
            return []

    def get_ip_interval_statistics(self, limit: int = 1000) -> List[Dict]:
        """
        Per-IP counts for the interval since the previous call.

        The kernel map holds lifetime counters; detection needs what each
        source sent *recently*, otherwise a host that was busy an hour ago
        still looks like a top talker and SYN/packet thresholds never reset.
        """
        current = self.get_ip_statistics(limit=10000)
        deltas = []
        seen = {}
        for ip_stat in current:
            ip = ip_stat['ip']
            counters = (ip_stat['packets'], ip_stat['bytes'],
                        ip_stat.get('syn_count', 0), ip_stat.get('udp_count', 0))
            seen[ip] = counters
            prev = self._prev_ip_counters.get(ip)
            # A counter going backwards means the LRU entry was evicted and recreated
            if prev is None or counters[0] < prev[0]:
                prev = (0, 0, 0, 0)
            d_packets = counters[0] - prev[0]
            if d_packets <= 0:
                continue
            deltas.append({
                **ip_stat,
                'packets': d_packets,
                'bytes': max(0, counters[1] - prev[1]),
                'syn_count': max(0, counters[2] - prev[2]),
                'udp_count': max(0, counters[3] - prev[3]),
            })
        self._prev_ip_counters = seen
        deltas.sort(key=lambda x: x['packets'], reverse=True)
        return deltas[:limit]

    def get_flow_statistics(self, limit: int = 100) -> List[Dict]:
        """
        Get flow statistics

        Args:
            limit: Maximum number of flows to return

        Returns:
            List of dictionaries with flow statistics
        """
        if not self.loaded or self.simulation_mode:
            return []

        try:
            flow_map = self.bpf.get_table("flow_map")
            flows = []

            count = 0
            for key, value in flow_map.items():
                if count >= limit:
                    break

                src_ip = socket.inet_ntoa(struct.pack('I', key.src_ip))
                dst_ip = socket.inet_ntoa(struct.pack('I', key.dst_ip))

                flows.append({
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'src_port': key.src_port,
                    'dst_port': key.dst_port,
                    'protocol': key.protocol,
                    'packets': value.packets,
                    'bytes': value.bytes,
                    'last_seen': value.last_seen,
                })

                count += 1

            # Sort by packet count (descending)
            flows.sort(key=lambda x: x['packets'], reverse=True)
            return flows

        except Exception as e:
            logger.error(f"Failed to read flow statistics: {e}")
            return []

    def get_flow_records(self, clear: bool = True) -> Optional[Dict]:
        """
        Every flow in flow_map as counter columns for the level-3 flow classifier
        (src/ml/flow_features.FLOW_COUNTERS), then empty the map so the next read
        covers the next window (batched lookup-and-delete). Times are converted to
        seconds. The map is LRU: in a flood it keeps the most recent flows.
        Returns None when there is no kernel map (simulation mode).
        """
        if not self.loaded or self.simulation_mode:
            return None
        try:
            flow_map = self.bpf.get_table("flow_map")
            keys, vals = _drain_map(flow_map, delete=clear)
        except Exception as e:
            logger.error(f"Failed to read flow records: {e}")
            return None
        unit = 1024e-9          # inter-arrival times are kept in 1024-ns units
        src_u32 = keys['src_ip'].astype(np.uint32)
        uniq, inv = np.unique(src_u32, return_inverse=True)
        names = np.array([socket.inet_ntoa(struct.pack('I', int(x))) for x in uniq], dtype=object)
        f = lambda name: vals[name].astype(np.float64)  # noqa: E731
        return {
            'src': names[inv] if len(uniq) else np.zeros(0, dtype=object),
            'proto': keys['protocol'].astype(np.float64),
            'sport': keys['src_port'].astype(np.float64),
            'dport': keys['dst_port'].astype(np.float64),
            'packets': f('packets'), 'bytes': f('bytes'),
            'len_min': f('len_min'), 'len_max': f('len_max'), 'len_sumsq': f('len_sumsq'),
            't_first': f('first_seen') * 1e-9, 't_last': f('last_seen') * 1e-9,
            'iat_sumsq': f('iat_sumsq') * unit * unit, 'iat_max': f('iat_max') * unit,
            'syn': f('syn'), 'ack': f('ack'), 'fin': f('fin'), 'rst': f('rst'), 'psh': f('psh'),
        }

    def add_to_blacklist(self, ip_address: str, ttl_s: Optional[float] = None) -> bool:
        """
        Add IP to blacklist

        Args:
            ip_address: IP address to block (dotted decimal)
            ttl_s: block duration in seconds; None = permanent. The kernel map
                   stores the expiry time and removes the entry when it passes.

        Returns:
            True if added successfully
        """
        if not self.loaded:
            return False

        normalized = validate_blockable_ip(ip_address)
        if normalized is None:
            logger.warning(f"Refusing to blacklist invalid or protected address: {ip_address!r}")
            return False
        ip_address = normalized

        if ttl_s is None and self.is_blacklisted(ip_address) and ip_address in self._permanent:
            return True

        if self.simulation_mode:
            self._sim_blacklist[ip_address] = float('inf') if ttl_s is None else self.sim_now + ttl_s
            if ttl_s is None:
                self._permanent.add(ip_address)
            logger.info(f"[SIM] Added {ip_address} to blacklist" + (f" for {ttl_s:g}s" if ttl_s else ""))
            return True

        try:
            blacklist_map = self.bpf.get_table("blacklist_map")
            ip_int = struct.unpack('I', socket.inet_aton(ip_address))[0]
            expires = 0 if ttl_s is None else _ktime_ns() + int(ttl_s * 1e9)

            blacklist_map[blacklist_map.Key(ip_int)] = blacklist_map.Leaf(expires)
            logger.info(f"Added {ip_address} to blacklist" + (f" for {ttl_s:g}s" if ttl_s else ""))

            if ttl_s is None:
                self._permanent.add(ip_address)
                self._save_blacklist_to_disk()
            return True

        except Exception as e:
            logger.error(f"Failed to add {ip_address} to blacklist: {e}")
            return False

    def rate_limit_ip(self, ip_address: str, pps: float, ttl_s: float) -> bool:
        """Limit a source to `pps` packets per second for `ttl_s` seconds (token bucket in XDP)"""
        if not self.loaded or validate_blockable_ip(ip_address) is None:
            return False
        pps = max(1, int(round(pps)))
        if self.simulation_mode:
            self._sim_limits[ip_address] = (float(pps), self.sim_now + ttl_s)
            return True
        try:
            table = self.bpf.get_table("ratelimit_map")
            ip_int = struct.unpack('I', socket.inet_aton(ip_address))[0]
            now = _ktime_ns()
            table[table.Key(ip_int)] = table.Leaf(pps, now + int(ttl_s * 1e9), pps * 1000, now)
            return True
        except Exception as e:
            logger.error(f"Failed to rate-limit {ip_address}: {e}")
            return False

    def add_known_sources(self, ip_addresses) -> int:
        """Mark sources as known from normal traffic (exempt from the spoofing guard)"""
        ips = [ip for ip in ip_addresses if validate_blockable_ip(ip)]
        if not self.loaded or not ips:
            return 0
        if self.simulation_mode:
            self._sim_known.update(ips)
            return len(ips)
        try:
            table = self.bpf.get_table("known_map")
            for ip in ips:
                table[table.Key(struct.unpack('I', socket.inet_aton(ip))[0])] = table.Leaf(1)
            return len(ips)
        except Exception as e:
            logger.error(f"Failed to update known sources: {e}")
            return 0

    def set_spoof_guard(self, ttl_s: float, total_pps: float) -> bool:
        """
        For `ttl_s` seconds, sources not in the known set share `total_pps`
        packets per second. The kernel keeps one bucket per CPU, so the budget
        is split across the online CPUs.
        """
        if not self.loaded:
            return False
        if self.simulation_mode:
            self._sim_guard = (self.sim_now + ttl_s, float(total_pps))
            return True
        try:
            import os
            table = self.bpf.get_table("guard_map")
            ncpu = max(1, len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1)
            per_cpu = max(1, int(total_pps / ncpu))
            now = _ktime_ns()
            leaf = table.sLeaf(per_cpu, now + int(ttl_s * 1e9), per_cpu * 1000, now)
            table[table.Key(0)] = table.Leaf(*([leaf] * table.total_cpu))
            return True
        except Exception as e:
            logger.error(f"Failed to set spoofing guard: {e}")
            return False

    def get_rate_limited(self) -> List[str]:
        """Sources currently under a rate limit"""
        if self.simulation_mode or not self.loaded:
            return sorted(ip for ip, (_, exp) in self._sim_limits.items() if exp > self.sim_now)
        try:
            table, now = self.bpf.get_table("ratelimit_map"), _ktime_ns()
            return [socket.inet_ntoa(struct.pack('I', k.value)) for k, v in table.items()
                    if v.expires_ns == 0 or v.expires_ns > now]
        except Exception as e:
            logger.error(f"Failed to read rate limits: {e}")
            return []

    def spoof_guard_active(self) -> bool:
        if self.simulation_mode or not self.loaded:
            return self._sim_guard is not None and self._sim_guard[0] > self.sim_now
        try:
            values = self.bpf.get_table("guard_map")[0]
            return any(v.expires_ns > _ktime_ns() for v in values)
        except Exception:
            return False

    def simulate_drops(self, packets: Dict[str, int]) -> Dict[str, float]:
        """
        Simulation mode: packets the XDP program would drop this interval, per
        source, applying blocks, rate limits and the spoofing guard in the same
        order as the kernel (one-second interval, so a limit of r pps admits r).
        """
        now = self.sim_now
        dropped, unknown = {}, []
        for ip, c in packets.items():
            if self._sim_blacklist.get(ip, -1.0) > now:
                dropped[ip] = float(c)
                continue
            limit = self._sim_limits.get(ip)
            d = max(0.0, c - limit[0]) if limit and limit[1] > now else 0.0
            dropped[ip] = d
            if ip not in self._sim_known:
                unknown.append(ip)
        if self._sim_guard and self._sim_guard[0] > now and unknown:
            admitted = sum(packets[ip] - dropped[ip] for ip in unknown)
            budget = self._sim_guard[1]
            if admitted > budget:
                keep = budget / admitted
                for ip in unknown:
                    dropped[ip] += (packets[ip] - dropped[ip]) * (1 - keep)
        return dropped

    def remove_from_blacklist(self, ip_address: str) -> bool:
        """
        Remove IP from blacklist

        Args:
            ip_address: IP address to unblock

        Returns:
            True if removed successfully
        """
        if not self.loaded:
            return False

        self._permanent.discard(ip_address)
        if self.simulation_mode:
            if ip_address in self._sim_blacklist:
                del self._sim_blacklist[ip_address]
                logger.info(f"[SIM] Removed {ip_address} from blacklist")
                return True
            return False

        try:
            blacklist_map = self.bpf.get_table("blacklist_map")
            ip_int = struct.unpack('I', socket.inet_aton(ip_address))[0]

            del blacklist_map[blacklist_map.Key(ip_int)]
            logger.info(f"Removed {ip_address} from blacklist")

            # Save to disk
            self._save_blacklist_to_disk()
            return True

        except Exception as e:
            logger.error(f"Failed to remove {ip_address} from blacklist: {e}")
            return False

    def get_blacklist(self) -> List[str]:
        """
        Get current blacklist

        Returns:
            List of blacklisted IP addresses
        """
        if not self.loaded or self.simulation_mode:
            return sorted(ip for ip, exp in self._sim_blacklist.items() if exp > self.sim_now)

        try:
            blacklist_map = self.bpf.get_table("blacklist_map")
            now = _ktime_ns()
            # Expired entries stay in the map until the next packet from that source
            return [socket.inet_ntoa(struct.pack('I', k.value)) for k, v in blacklist_map.items()
                    if _active(v, now)]

        except Exception as e:
            logger.error(f"Failed to read blacklist: {e}")
            return []

    def is_blacklisted(self, ip_address: str) -> bool:
        """Check whether an IP is currently blacklisted"""
        if self.simulation_mode:
            return self._sim_blacklist.get(ip_address, -1.0) > self.sim_now
        if self.bpf is None:
            return False
        try:
            blacklist_map = self.bpf.get_table("blacklist_map")
            ip_int = struct.unpack('I', socket.inet_aton(ip_address))[0]
            value = blacklist_map.get(blacklist_map.Key(ip_int))
            return value is not None and _active(value, _ktime_ns())
        except Exception:
            return False

    def _save_blacklist_to_disk(self):
        """Save current blacklist to disk for persistence"""
        try:
            blacklist = sorted(self._permanent)
            with open(self.blacklist_file, 'w') as f:
                json.dump(blacklist, f)
            logger.debug(f"Saved {len(blacklist)} IPs to blacklist file")
        except Exception as e:
            logger.error(f"Failed to save blacklist to disk: {e}")

    def _load_blacklist_from_disk(self):
        """Load blacklist from disk into eBPF map"""
        if not Path(self.blacklist_file).exists():
            return

        try:
            with open(self.blacklist_file, 'r') as f:
                blacklist = json.load(f)

            count = 0
            for ip in blacklist:
                if self.add_to_blacklist(ip):
                    count += 1

            logger.info(f"Restored {count} IPs from blacklist file")
        except Exception as e:
            logger.error(f"Failed to load blacklist from disk: {e}")
