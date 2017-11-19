#!/bin/bash
set -e
extra_args="-march=native -fomit-frame-pointer"

mkdir -p bin
rm -f result_simple.txt

runs_aos=4
runs=12
echo "v_size: irka*${runs}, soa*${runs}, aos*${runs_aos}"
echo "Every second line shows minimum values for all ikra, soa, aos runs."

for v_size in `./size_generator.py`
do
	g++ -O3 ${extra_args} simple_ikra.cc -DkNumBodies=${v_size} \
      -std=c++11 -I../../../../../ikra -o bin/simple_ikra
	g++ -O3 ${extra_args} simple_soa.cc -DkNumBodies=${v_size} \
      -std=c++11 -o bin/simple_soa
	g++ -O3 ${extra_args} simple_aos.cc -DkNumBodies=${v_size} \
      -std=c++11 -o bin/simple_aos

	echo -n -e "\e[1m\e[91m"
	echo -n -e ".[${v_size}]"

	line="${runs},${v_size}"

	for ((i=0; i<$runs; i++))
	do
		result=`bin/simple_ikra`
		r_ikra[$i]=$result
		line="${line},${result}"
		echo -n "."
	done

	min_ikra=${r_ikra[0]}
	for (( i=1; i<${runs}; i++ ))
	do
		if (( $(echo "$min_ikra > ${r_ikra[$i]}" | bc -l) ))
		then
			min_ikra=${r_ikra[$i]}
		fi
	done
	echo -n "[ $min_ikra ]"



	for ((i=0; i<$runs; i++))
	do
		result=`bin/simple_soa`
		r_soa[$i]=$result
		line="${line},${result}"
		echo -n "."
	done

	min_soa=${r_soa[0]}
	for (( i=1; i<${runs}; i++ ))
	do
		if (( $(echo "$min_soa > ${r_soa[$i]}" | bc -l) ))
		then
			min_soa=${r_soa[$i]}
		fi
	done
	echo -n "[ $min_soa ]"



	for ((i=0; i<$runs_aos; i++))
	do
		result=`bin/simple_aos`
		r_aos[$i]=$result
		line="${line},${result}"
		echo -n "."
	done

	min_aos=${r_aos[0]}
	for (( i=1; i<${runs_aos}; i++ ))
	do
		if (( $(echo "$min_aos > ${r_aos[$i]}" | bc -l) ))
		then
			min_aos=${r_aos[$i]}
		fi
	done
	echo "[ $min_aos ]"


	echo -n -e "\e[0m\e[90m"
	echo $line >> result_simple.txt
	echo "   ---> $line"
done
