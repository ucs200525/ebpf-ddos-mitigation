# Attacker laptop — DDoS attack tool

Copy this `attacker/` folder to the **attacker laptop** (the second laptop). It floods the
protection server with real packets over the hotspot. The server detects the source and
drops its packets in the kernel with XDP.

Only ever point this at the lab server you are demonstrating (`10.23.12.77`, or whatever the
server window prints). It is a load generator for your own machine, nothing else.

## Before you start

1. Connect the attacker laptop to the **same phone hotspot** as the server.
2. Find the server IP: it is printed in the server window on laptop 1
   (`ON THE ATTACKER LAPTOP, aim the attack tool at: 10.23.12.77`).
3. For the strongest effect, turn off the attacker laptop's firewall and run as
   admin / root (see below). UDP and HTTP floods work even without that.

## Run it

**Windows**
- Double-click `attack.bat` and answer the two prompts (server IP, attack type), **or**
  from a terminal:
  ```bat
  attack.bat 10.23.12.77 udp
  ```
- For the ICMP or spoofed-SYN floods, right-click `attack.bat` → **Run as administrator**.
- Turn the firewall off for the test: Windows Security → Firewall & network protection →
  turn off for the active (hotspot) network. Turn it back on afterwards.

**Linux**
```bash
sudo bash attack.sh 10.23.12.77 udp
```
- `sudo` is needed for the fast ICMP flood and the spoofed-SYN flood.
- Disable the firewall for the test if one is on: `sudo ufw disable` (re-enable later).

## Attack types

| `--type` | What it sends | Needs admin/root | Notes |
|---|---|---|---|
| `udp`  | UDP flood to a port | no | **Recommended.** Highest packets/s, clearest on the chart. |
| `tcp`  | TCP SYN flood (half-open connections) | no | Each connection attempt is a real SYN. |
| `http` | Flood of HTTP GET requests | no | Application-layer load. |
| `icmp` | Ping flood | yes for the fast path | Falls back to the system `ping` flood without admin (slower). |
| `syn`  | Spoofed SYN flood, random fake source IPs | yes + scapy | Looks like a *distributed* attack from one machine. Windows also needs Npcap. |
| `all`  | UDP + TCP + HTTP (+ ICMP if privileged) at once | partial | Multi-vector. |

Options: `--port`, `--threads` (default 8), `--duration` (seconds; default: until Ctrl+C),
`--size` (UDP payload bytes). Examples:
```bash
python ddos_attack.py 10.23.12.77 --type udp --threads 16
python ddos_attack.py 10.23.12.77 --type all --duration 30
sudo python3 ddos_attack.py 10.23.12.77 --type icmp
```

The spoofed-SYN flood needs scapy: `pip install scapy` (and Npcap on Windows). Everything
else uses only the standard library, so plain Python 3 is enough.

## What you should see

- On the **attacker** screen: a rising count of packets sent per second.
- On the **server dashboard** (phone / projector): the banner turns red "UNDER ATTACK",
  this laptop's IP appears in the source table marked **Blocked**, and the "Dropped by XDP"
  line climbs. Within a couple of seconds most of your packets are being dropped in the
  kernel before they reach the server's network stack.
- Stop with **Ctrl+C**. After about 30 s the block expires and the server goes back to green.
