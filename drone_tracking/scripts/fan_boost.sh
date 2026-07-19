#!/bin/bash
# fan_boost.sh on|off — switch the WINDOWS power plan from WSL.
# "on"  = High performance (raises boost + thermal envelope -> the MSI EC
#         ramps the fans harder under sim load)
# "off" = Balanced (normal quiet behavior)
# Called automatically: launch_stack.sh -> on (FAN_BOOST=0 disables),
# cleanup.sh -> off. Direct fan-RPM control isn't exposed to WSL (MSI
# Cooler Boost is MSI-Center/Fn-hotkey only); this is the scriptable lever.
HIGH_GUID="a62641af-b1ae-48fb-86e9-df431c074c74"   # created 2026-07-15
BAL_GUID="381b4222-f694-41f0-9685-ff5bb260df2e"    # Windows default Balanced
case "${1:-}" in
  on)  powercfg.exe /setactive "$HIGH_GUID" >/dev/null 2>&1 || true ;;
  off) powercfg.exe /setactive "$BAL_GUID"  >/dev/null 2>&1 || true ;;
  *)   echo "usage: fan_boost.sh on|off"; exit 1 ;;
esac
