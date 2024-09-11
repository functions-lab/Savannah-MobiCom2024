#!/bin/bash

################################################################################
# Script Name: savannah.sh
# Description: This script is the entry point of Savannah. It is used to build,
#              and run the project. The script can also generate data for the
#              transmission. The transmission involves base station (BS, the
#              main program) and the usre equipment (UE). The script can run
#              the BS and UE in simulation mode or radio mode. It can also read
#              the log of the latest test run. For detailed usage, use -h or
#              --help.
#              The script will direct the output to a log file with a timestamp.
# Author     : @cstandy
################################################################################

################
# Path & Param
################

sudo=
ld_path=
bs_exe=./build/agora
sim_ue_exe=./build/sender
rru_ue_exe=./build/user
ue_exe=$sim_ue_exe
ue_arg=
data_gen_exe=./build/data_generator

sim_config=./files/config/ci/tddconfig-sim-ul-fr2.json
rru_config=./files/config/examples/ul-usrp.json
config=$sim_config
logpath=./log
logfile=$logpath/$(date +"%Y-%m-%d_%H-%M-%S").log

################
# Functions
################

# Function to display help
function display_help {
    echo "Usage: ./savannah.sh [option] [mode] [previlege]"
    echo ""
    echo "Options:"
    echo "  -b, --build   - Build the project"
    echo "  -g, --datagen - Generate the data from the config file"
    echo "  -x, --execute - Run the basestation"
    echo "  -u, --user    - Run the user"
    echo "  -r, --read    - Read the log of the latest test run"
    echo "  -c, --clean   - Clean the project"
    echo "  -a, --acc100  - Set the project to use ACC100 for LDPC, need rebuild"
    echo "  -f, --flexran - Set the project to use FlexRAN for LDPC, need rebuild"
    echo "  -d, --debug   - Set the project in debug mode, need to rebuild"
    echo "  -n, --normal  - Set the project in normal mode, need to rebuild"
    echo "  -h, --help    - Display this help message"
    echo ""
    echo "Mode (used with -g, -x, or -u flag):"
    echo "  -r, --radio   - Compile/generate/run with RRU mode"
    echo "  -s, --sim     - Compile/generate/run with Simulation mode"
    echo ""
    echo "Previlege (optional, used with -x):"
    echo "  -r, --root    - Run the command with root (sudo) privilege"
    echo ""
    echo "Common usage:"
    echo "./savannah.sh -a      : config cmake to use ACC100"
    echo "./savannah.sh -f      : config cmake to use FlexRAN"
    echo "./savannah.sh -b -r   : build in RRU mode"
    echo "./savannah.sh -b -s   : build in Simulation mode"
    echo "./savannah.sh -x -s -r: run bs in sim mode with root privilege"
    echo "./savannah.sh -x -r -r: run bs in rru mode with root privilege"
    echo "./savannah.sh -u -s   : run ue in sim mode"
    echo "./savannah.sh -u -r   : run ue in rru mode"
    echo "./savannah.sh -g -s   : generate data for sim mode"
    echo "./savannah.sh -g -r   : generate data for rru mode"
}

# Function to build the project
function build_project {
    echo "Building the project..."
    cd build
    make -j50
    cd ..
}

function set_debug {
    echo "Setting the project to debug mode..."
    cd build
    cmake .. -DDEBUG=true
    cd ..
}

function set_normal {
    echo "Setting the project to normal mode..."
    cd build
    cmake .. -DDEBUG=false
    cd ..
}

function set_sim {
    echo "Setting the project to simulation mode..."
    cd build
    cmake .. -DRADIO_TYPE=SIMULATION
    cd ..
}

function set_rru {
    echo "Setting the project to radio mode..."
    cd build
    cmake .. -DRADIO_TYPE=PURE_UHD
    cd ..
}

function set_decode_acc100 {
    echo "Setting the project to use ACC100 for LDPC..."
    cd build
    cmake .. -DLDPC_TYPE=ACC100
    cd ..
}

function set_decode_flexran {
    echo "Setting the project to use FlexRAN for LDPC..."
    cd build
    cmake .. -DLDPC_TYPE=FlexRAN
    cd ..
}

function gen_data {
    echo "Generating the data for simulation based on config file: $config"
    echo "$data_gen_exe --conf_file $config"
    $data_gen_exe --conf_file $config
}

# Function to run the project
function exe_bs {
    echo "Running the basestation..."
    echo "script -q -c "$sudo $ld_path $bs_exe --conf_file $config" $logfile"
    script -q -c "$sudo $ld_path $bs_exe --conf_file $config" $logfile
    # Use `cat log/2023-06-23_11-32-22.log | less -R` to read colored log file
}

function exe_user {
    echo "Running the user..."
    echo "$ue_exe --conf_file $config $ue_arg"
    $ue_exe --conf_file $config $ue_arg
    # script -q -c "$ue_exe --conf_file $config" $logfile
}

# Function to read the log
function read_log {
    # Find the latest log file
    latest_log=$(ls -t "$logpath"/*.log | head -1)

    # Read the contents
    if [ -f "$latest_log" ]; then
        echo "Reading the latest log: $latest_log with"
        echo "cat \`$latest_log | less -R\` to read colored log file"
        # tail "$latest_log"
        cat $latest_log | less -R
    else
        echo "No log files found in the directory"
    fi
}

# Function to clean the project
function clean_project {
    echo "Cleaning the project..."
    cd build
    make clean
    cd ..
}

################
# Handle Inputs
################

# Check the number of arguments
if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    display_help
    exit 1
fi

# Handle the argument

case "$3" in
    "-r" | "--root")
        echo "Running the project with root privilege (sudo)..."
        sudo=sudo
        ld_path="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
        ;;
    \?)
        echo "Invalid option."
        display_help
        ;;
esac

case "$2" in
    "-r" | "--radio")
        config=$rru_config
        ue_exe=$rru_ue_exe
        ue_arg=
        set_rru
        ;;
    "-s" | "--sim")
        config=$sim_config
        ue_exe=$sim_ue_exe
        ue_arg="--conf_file $config   \
                --num_threads=2       \
                --core_offset=10      \
                --enable_slow_start=0"
        set_sim
        ;;
    \?)
        echo "Invalid option."
        display_help
        ;;
esac

case "$1" in
    "-b" | "--build")
        build_project
        ;;
    "-d" | "--debug")
        set_debug
        ;;
    "-n" | "--normal")
        set_normal
        ;;
    "-g" | "--datagen")
        gen_data
        ;;
    "-x" | "--exe")
        exe_bs
        ;;
    "-u" | "--user")
        exe_user
        ;;
    "-r" | "--read")
        read_log
        ;;
    "-c" | "--clean")
        clean_project
        ;;
    "-a" | "--acc100")
        set_decode_acc100
        ;;
    "-f" | "--flexran")
        set_decode_flexran
        ;;
    "-h" | "--help")
        display_help
        ;;
    *)
        echo "Invalid option."
        display_help
        ;;
esac

