#!/bin/bash
cmake -DCMAKE_BUILD_TYPE=Release -D num_cars=250000 CMakeLists.txt
make

ERR_SF=$((
(
bin/traffic ~/Downloads/graphml_urbanized/San_Francisco--Oakland__CA_Urbanized_Area_78904.graphml
) 1>/dev/null
) 2>&1)


ERR_LOS=$((
(
bin/traffic ~/Downloads/graphml/0644000_Los_Angeles.graphml
) 1>/dev/null
) 2>&1)


cmake -DCMAKE_BUILD_TYPE=Release -D num_cars=200000 CMakeLists.txt
make

ERR_NYC=$((
(
bin/traffic ~/Downloads/graphml/3651000_New_York.graphml
) 1>/dev/null
) 2>&1)

cmake -DCMAKE_BUILD_TYPE=Release -D num_cars=100000 CMakeLists.txt
make

ERR_SD=$((
(
bin/traffic ~/Downloads/graphml/0666000_San_Diego.graphml
) 1>/dev/null
) 2>&1)


echo "SF, LOS, NYC, SD"
echo $ERR_SF
echo $ERR_LOS
echo $ERR_NYC
echo $ERR_SD
