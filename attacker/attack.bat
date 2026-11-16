@echo off
rem DDoS attack tool for the capstone demo (attacker laptop, Windows).
rem Usage:  attack.bat [server-ip] [type]      e.g.  attack.bat 10.23.12.77 udp
rem Run as Administrator for the ICMP/SYN floods. Disable the firewall for best effect.
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 ddos_attack.py %*
) else (
    python ddos_attack.py %*
)
pause
