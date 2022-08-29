#!/bin/bash

### SAA ###
echo "Renaming SAA"

echo "Renaming PDU current"
find ./ -name "*npwd2260*" -exec rename npwd2260 saa '{}' \;

echo "Renaming PDU voltage"
find ./ -name "*npwd2268*" -exec rename npwd2268 saa '{}' \;

echo "Renaming bus voltage"
find ./ -name "*npwd1024*" -exec rename npwd1024 saa '{}' \;

echo "Renaming solar displacement error"
find ./ -name "*nacx0017*" -exec rename nacx0017 saa '{}' \;

echo "Renaming solar angular position deg"
find ./ -name "*nacx0021*" -exec rename nacx0021 saa '{}' \;

echo "Renaming solar angular position rad"
find ./ -name "*nacw1307*" -exec rename nacw1307 saa '{}' \;

echo "Renaming solar incidence"
find ./ -name "*nawg0026*" -exec rename nawg0026 saa '{}' \;

echo "Renaming PCU Voltage"
find ./ -name "*npwd1044*" -exec rename npwd1044 saa '{}' \;

### SAB ###
echo "Renaming SAB"

echo "Renaming PDU current"
find ./ -name "*npwd2920*" -exec rename npwd2920 sab '{}' \;

echo "Renaming PDU voltage"
find ./ -name "*npwd2928*" -exec rename npwd2928 sab '{}' \;

echo "Renaming bus voltage"
find ./ -name "*npwd102b*" -exec rename npwd102b sab '{}' \;

echo "Renaming solar displacement error"
find ./ -name "*nacx0016*" -exec rename nacx0016 sab '{}' \;

echo "Renaming solar angular position deg"
find ./ -name "*nacx0020*" -exec rename nacx0020 sab '{}' \;

echo "Renaming solar angular position rad"
find ./ -name "*nacw1306*" -exec rename nacw1306 sab '{}' \;

echo "Renaming solar incidence"
find ./ -name "*nawg0025*" -exec rename nawg0025 sab '{}' \;

echo "Renaming PCU Voltage"
find ./ -name "*npwd104b*" -exec rename npwd104b sab '{}' \;

### General ###
echo "Renaming General"

echo "Renaming PCU Current"
find ./ -name "*npwd1104*" -exec rename npwd1104 sag '{}' \;

echo "Renaming PCU Voltage"
find ./ -name "*npwd1704*" -exec rename npwd1704 sag '{}' \;

echo "Renaming solar misalignment"
find ./ -name "*nacx0022*" -exec rename nacx0022 sag '{}' \;

