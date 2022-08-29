#!/bin/bash

#declare -a periods=("96" "288" "627" "1344" "2880")
declare -a periods=("35040")
declare -a files=("saa" "sab" "sag")
declare -a year=("2004" "2005" "2006" "2007" "2008" "2009" "2010")

for f in "${files[@]}"; do
#	for y in "${year[@]}"; do
		### Collect data ###
		if [ $1 -eq 1 ]
		then
			python rosetta_collect.py /home/mattis/storage-b/master-thesis/Sources/Data/ro-x-hk-3-solararray-v1.0/data $f 2011 11 ;
		fi

		### CSV to PNG ###
		if [ $1 -eq 2 ]
		then
			python csv_to_png.py $f.csv ;
		fi
									
		### Create meta info ###
		if [ $1 -eq 3 ]
		then
			python rosetta_meta.py $f.csv ;
		fi

		### Prune dataset ###
		if [ $1 -eq 4 ]
		then
			python rosetta_prune.py $f.csv &
		fi
					
		### Prune CSV to PNG ###
		if [ $1 -eq 5 ]
		then
			python csv_to_png.py $f\_prune.csv &
		fi
					
		### X11 ##
		if [ $1 -eq 6 ]
		then
			for p in "${periods[@]}"
				do python x11.py $f\_prune.csv $p &
			done
		fi
							
		### FFT ##
		if [ $1 -eq 7 ]
		then
			python fft.py $f\_prune.csv ;
		fi
#	done
done
