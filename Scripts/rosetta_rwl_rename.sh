#!/bin/bash

### RWA ###
echo "Renaming RWA"

echo "Renaming est_frict_torque"
find ./ -name "*nacw0g05*" -exec rename nacw0g05 rwa '{}' \;

echo "Renaming frict_coeff"
find ./ -name "*naag0005*" -exec rename naag0005 rwa '{}' \;

echo "Renaming meas_ang_mom"
find ./ -name "*nacg0010*" -exec rename nacg0010 rwa '{}' \;

echo "Renaming wheel_direction"
find ./ -name "*naad6011*" -exec rename naad6011 rwa '{}' \;

echo "Renaming wheel speed"
find ./ -name "*nacg0014*" -exec rename nacg0014 rwa '{}' \;

### RWB ###
echo "Renaming RWB"

echo "Renaming est_frict_torque"
find ./ -name "*nacw0g0h*" -exec rename nacw0g0h rwb '{}' \;

echo "Renaming frict_coeff"
find ./ -name "*naag0006*" -exec rename naag0006 rwb '{}' \;

echo "Renaming meas_ang_mom"
find ./ -name "*nacg0011*" -exec rename nacg0011 rwb '{}' \;

echo "Renaming wheel_direction"
find ./ -name "*naad6021*" -exec rename naad6021 rwb '{}' \;

echo "Renaming wheel speed"
find ./ -name "*nacg0015*" -exec rename nacg0015 rwb '{}' \;

### RWC ###
echo "Renaming RWC"

echo "Renaming est_frict_torque"
find ./ -name "*nacw0g0t*" -exec rename nacw0g0t rwc '{}' \;

echo "Renaming frict_coeff"
find ./ -name "*naag0007*" -exec rename naag0007 rwc '{}' \;

echo "Renaming meas_ang_mom"
find ./ -name "*nacg0012*" -exec rename nacg0012 rwc '{}' \;

echo "Renaming wheel_direction"
find ./ -name "*naad6031*" -exec rename naad6031 rwc '{}' \;

echo "Renaming wheel speed"
find ./ -name "*nacg0016*" -exec rename nacg0016 rwc '{}' \;

### RWD ###
echo "Renaming RWD"

echo "Renaming est_frict_torque"
find ./ -name "*nacw0g15*" -exec rename nacw0g15 rwd '{}' \;

echo "Renaming frict_coeff"
find ./ -name "*naag0008*" -exec rename naag0008 rwd '{}' \;

echo "Renaming meas_ang_mom"
find ./ -name "*nacg0013*" -exec rename nacg0013 rwd '{}' \;

echo "Renaming wheel_direction"
find ./ -name "*naad6041*" -exec rename naad6041 rwd '{}' \;

echo "Renaming wheel speed"
find ./ -name "*nacg0017*" -exec rename nacg0017 rwd '{}' \;


