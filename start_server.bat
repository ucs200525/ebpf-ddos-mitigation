@echo off
rem Protection server for the review. Runs in WSL Kali on the real Linux kernel (live XDP),
rem attached to the Wi-Fi hotspot interface. The attacker laptop floods this machine and the
rem packets are dropped in the kernel. Connect this laptop to the phone hotspot first.
rem Ctrl+C in this window (or closing it) stops the server and removes XDP.
cd /d "%~dp0"
wsl.exe -d kali-linux -u root --cd "%~dp0" bash scripts/live_net_server.sh
pause
