@echo off
rem bot.py writes bot.log itself; only capture crashes here to avoid a file lock.
cd /d "%~dp0"
python bot.py 2>> crash.log
