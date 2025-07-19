# -*- coding: gbk -*-
import os
import time

# Database connection and configuration
ip = "127.0.0.1"                # Database IP address
user = "postgres"               # Database user
password = "postgres"           # Database password
database = "tpcc"               # TPCC benchmark database name
database_ycsb = "ycsb"          # YCSB benchmark database name
database_wiki = "wikipedia"     # Wikipedia benchmark database name
i_port = 5432                   # Database port

# PostgreSQL control commands
start_PG_cmd = "systemctl restart postgresql-12"   # Command to start (restart) PostgreSQL service
stop_PG_cmd = "systemctl stop postgresql-12"       # Command to stop PostgreSQL service
status_PG_cmd = "systemctl status postgresql-12"   # Command to check PostgreSQL service status

# Commands for checking PostgreSQL status
port_open = "netstat -tulpn"                       # Command to check open network ports
port = ":5432"                                     # PostgreSQL port to check
PG_open = "/usr/pgsql-12/bin/postmaster"           # PostgreSQL process to check
pid_open = "ps -ef | grep postgres"                # Command to check if postgres processes are running

# Command to clear Linux file system cache (requires root privileges)
free_cache_cmd = "echo 3 > /proc/sys/vm/drop_caches"

# Workload execution commands for OLTPBench benchmarks
oltp_cmd = "cd /root/oltpbench; ./oltpbenchmark -b tpcc -c config/pg_tpcc.xml --execute=true -s 10 -o outputfile"
oltp_cmd_background = "cd /root/oltpbench; ./oltpbenchmark -b tpcc -c config/pg_tpcc.xml --execute=true -s 10 -o outputfile &"
oltp_cmd_ycsb = "cd /root/oltpbench; ./oltpbenchmark -b ycsb -c config/pg_ycsb.xml --execute=true -s 10 -o outputfile"
oltp_cmd_ycsb_background = "cd /root/oltpbench; ./oltpbenchmark -b ycsb -c config/pg_ycsb.xml --execute=true -s 10 -o outputfile &"
oltp_cmd_wiki = "cd /root/oltpbench; ./oltpbenchmark -b wikipedia -c config/pg_wk.xml --execute=true -s 10 -o outputfile"
oltp_cmd_wiki_background = "cd /root/oltpbench; ./oltpbenchmark -b wikipedia -c config/pg_wk.xml --execute=true -s 10 -o outputfile &"

throught_char = "requests/sec"                      # Throughput indicator for parsing benchmark results
del_result_file = "rm /root/oltpbench/results/output*"      # Command to delete previous result files
result_file_path = "/root/oltpbench/results/outputfile.summary" # Benchmark result summary file path

# Command to clean internal metrics files
clean_inner_metrics = "echo '' > /root/status/inner_metrics"

# Configuration file paths
path_my_cnf = "/var/lib/pgsql/12/data/postgresql.conf"      # PostgreSQL configuration file path

status_path = "/root/status/status"                  # Status file path
clean_status = "echo '' >  /root/status/status"      # Command to clean status file
clean_tuning_process = "echo '' > /root/status/tuning_process"  # Command to clean tuning process file
inner_metrics_path = "/root/status/inner_metrics"     # Path for inner metrics file
memory_path = "/root/status/memory_replay"            # Path for memory replay file

# Benchmark execution and system monitoring time parameters
run_time = 180                                       # Total time to collect monitoring information (in seconds)
start_time = None

# sar (System Activity Reporter) command settings for monitoring CPU and disk usage
sar_num = int(run_time/10)                           # Number of sar samples (1 sample every 10 seconds)
sar_u_com = "sar -u 1 " + str(sar_num) + "  > /root/status/sar_u &"   # Monitor CPU usage and output to file (background)
sar_d_com = "sar -d 1 " + str(sar_num) + " > /root/status/sar_d &"    # Monitor disk usage and output to file (background)
clean_sar_u_com = "echo '' > /root/status/sar_u"     # Command to clean sar CPU usage log
clean_sar_d_com = "echo '' > /root/status/sar_d"     # Command to clean sar disk usage log
path_sar_d = "/root/status/sar_d"                    # Path for sar disk usage log file
path_sar_u = "/root/status/sar_u"                    # Path for sar CPU usage log file

# Function to start the PostgreSQL database
def start_PG():
    """
    Restart the PostgreSQL database service.
    Prints success message if started successfully.
    """
    if os.system(start_PG_cmd) == 0:
        print("Database started successfully.")

# Function to stop the PostgreSQL database
def stop_PG():
    """
    Stop the PostgreSQL database service.
    Prints success message if stopped successfully.
    """
    if os.system(stop_PG_cmd) == 0:
        print("Database stopped successfully.")

# Function to check PostgreSQL status, port, and process, with automatic retries and delays.
def status_PG():
    """
    Continuously check if the PostgreSQL service is running:
    - Verify if the port is open
    - Check if the process ID exists
    - Check service status
    Includes automatic retries and wait periods to ensure service is fully up.
    """
    cnt = 0
    flag_start = False
    # Check if database port is open
    while True:
        cnt += 1
        if flag_start:
            break
        time.sleep(5)
        if cnt % 40 == 0:
            start_PG()
        tps = os.popen(port_open)
        for i_port in tps.readlines():
            if port in i_port and "postmaster" in i_port:
                print(i_port)
                flag_start = True
                break
    print("Database port is open and running.")

    flag_start = False
    cnt = 0
    # Check if database PID (process) is running
    while True:
        cnt += 1
        if flag_start:
            break
        time.sleep(5)
        if cnt % 40 == 0:
            start_PG()
        tps = os.popen(pid_open)
        list_pid = tps.readlines()
        for i_pid in list_pid:
            if PG_open in i_pid:
                print(i_pid)
                flag_start = True
    print("Database process ID is active.")

    cnt = 0
    flag_start = False
    # Check service status via systemctl
    while True:
        cnt += 1
        if flag_start:
            break
        time.sleep(5)
        if cnt % 40 == 0:
            start_PG()
        status = os.popen(status_PG_cmd)
        list_status = status.readlines()
        for i_status in list_status:
            if "active (running)" in i_status:
                flag_start = True
                print(i_status)
                break
    print("Database service status is active.")

    # Wait additional 60 seconds to ensure database is fully ready before operations
    print("Database will sleep for 60 seconds to ensure full startup.")
    time.sleep(60)
    print("Database sleep complete.")

# Function to free system cache (requires root privileges)
def free_cache():
    """
    Clear Linux file system cache to avoid caching effects before benchmark runs.
    """
    print("Clearing system cache...")
    if os.system(free_cache_cmd) == 0:
        print("System cache cleared successfully.")

if __name__ == "__main__":
    # Example main workflow for database and system preparation
    free_cache()         # Clear system cache
    stop_PG()            # Stop PostgreSQL service
    start_PG()           # Start PostgreSQL service
    status_PG()          # Check and ensure PostgreSQL is running

    # Example: sar monitoring (uncomment as needed)
    # print(sar_u_com)
    # os.system(sar_u_com)
    # print("Started CPU monitoring with sar.")
    # print(sar_d_com)
    # os.system(sar_d_com)
    # print("Started disk monitoring with sar.")
    # time.sleep(5)
    # os.system(clean_sar_d_com)    # Clean disk monitoring log file after monitoring
