# -*- coding: gbk -*-
import os
import time

# Basic connection and authentication parameters for MySQL
ip = "106.75.245.152"            # MySQL server IP address
user = "root"                    # MySQL username
password = "Yzw123456"           # MySQL password
database = "tpcc"                # Database name for TPCC benchmark
database_ycsb = "ycsb"           # Database name for YCSB benchmark
database_wiki = "wikipedia"      # Database name for Wikipedia benchmark
i_port = 3306                    # MySQL server port

# MySQL server control commands
start_mysql_cmd = "systemctl start mysqld"      # Start MySQL service
stop_mysql_cmd = "systemctl stop mysqld"        # Stop MySQL service
status_mysql_cmd = "systemctl status mysqld"    # Check status of MySQL service

# Commands for checking MySQL status
port_open = "netstat -tulpn"                    # Command to list all listening ports
port = ":3306"                                 # The port to check for MySQL
mysql_safe_open = "/usr/sbin/mysqld"           # MySQL server process to look for
pid_open = "ps -ef | grep mysqld"              # Command to check MySQL processes

# Command to clear the OS cache (requires root)
free_cache_cmd = "echo 3 > /proc/sys/vm/drop_caches"

# OLTPBench workload execution commands
oltp_cmd = "cd /root/oltpbench; ./oltpbenchmark -b tpcc -c config/sample_tpcc_config.xml --execute=true -s 10 -o outputfile"
oltp_cmd_background = "cd /root/oltpbench; ./oltpbenchmark -b tpcc -c config/sample_tpcc_config.xml --execute=true -s 10 -o outputfile &"
oltp_cmd_ycsb = "cd /root/oltpbench; ./oltpbenchmark -b ycsb -c config/sample_ycsb_config.xml --execute=true -s 10 -o outputfile"
oltp_cmd_ycsb_background = "cd /root/oltpbench; ./oltpbenchmark -b ycsb -c config/sample_ycsb_config.xml --execute=true -s 10 -o outputfile &"
oltp_cmd_wiki = "cd /root/oltpbench; ./oltpbenchmark -b wikipedia -c config/sample_wikipedia_config.xml --execute=true -s 10 -o outputfile"
oltp_cmd_wiki_background = "cd /root/oltpbench; ./oltpbenchmark -b wikipedia -c config/sample_wikipedia_config.xml --execute=true -s 10 -o outputfile &"

throught_char = "requests/sec"                  # Character string for throughput in output
del_result_file = "rm /root/oltpbench/results/output*"   # Remove previous output files
del_inner_file = "rm /root/status/inner_metrics"         # Remove inner metrics file
chown_result = "chown -R mysql:mysql /root/oltpbench/results" # Change result directory ownership

# MySQL recovery and log cleaning commands
mysql_path = "/var/lib/"                        # MySQL installation directory
data_path = "/var/lib/mysql"                    # Actual data directory for MySQL
data_cp_path = "/var/lib/cp_mysql_init"         # Path for initial data backup
del_data_cmd = "rm -rf /var/lib/mysql"          # Delete MySQL data directory
cp_data_cmd = "cp -rf /var/lib/cp_mysql_init /var/lib/mysql" # Restore data directory from backup
clean_log_err = "echo '' >  /var/log/mysqld.log"             # Clear MySQL error log
clean_inner_metrics = "echo '' > /root/status/inner_metrics" # Clear inner metrics file

# File paths for configuration and results
path_my_cnf = "/etc/my.cnf"                     # MySQL configuration file
result_file_path = "/root/oltpbench/results/outputfile.summary" # Output summary file path
status_path = "/root/status/status"             # File to record status
clean_status = "echo '' >  /root/status/status" # Clear status file
log_err_path = "/var/log/mysqld.log"            # MySQL error log path
tunning_process_path = "/root/status/tuning_process"          # File to record tuning process
clean_tuning_process = "echo '' > /root/status/tuning_process" # Clear tuning process file
inner_metrics_path = "/root/status/inner_metrics"             # File to record inner metrics
memory_path = "/root/status/memory_replay"       # File to record memory usage for replay
model_path = "/root/status/model"                # File to store model path

# Benchmark and monitoring timing parameters
run_time = 180                                  # Time for collecting monitoring info (in seconds)
start_time = None

# sar (System Activity Reporter) command settings for monitoring CPU and disk activity
sar_num = int(run_time / 10)                    # Number of sar data points (every 10 seconds)
sar_u_com = "sar -u 10 " + str(sar_num) + "  > /root/status/sar_u &" # Monitor CPU usage
sar_d_com = "sar -d 10 " + str(sar_num) + " > /root/status/sar_d &"  # Monitor disk activity
clean_sar_u_com = "echo '' > /root/status/sar_u" # Clear sar CPU usage log
clean_sar_d_com = "echo '' > /root/status/sar_d" # Clear sar disk usage log
path_sar_d = "/root/status/sar_d"                # File to store sar disk usage log
path_sar_u = "/root/status/sar_u"                # File to store sar CPU usage log

# Function to start MySQL server
def start_mysql():
    """
    Start the MySQL server using the system service command.
    """
    if os.system(start_mysql_cmd) == 0:
        print("MySQL server started successfully.")

# Function to stop MySQL server
def stop_mysql():
    """
    Stop the MySQL server using the system service command.
    """
    if os.system(stop_mysql_cmd) == 0:
        print("MySQL server stopped successfully.")

# Function to check MySQL server status, port, and process; will retry with waits for reliability.
def status_mysql():
    """
    Continuously check if the MySQL server is running:
    - Check if the port is open.
    - Check if the server process (mysqld) is running.
    - Check the system service status for active state.
    Includes retries and waits to ensure the server is fully operational.
    """
    cnt = 0
    flag_start = False
    # Step 1: Check if the MySQL port is open (i.e., server is listening)
    while True:
        cnt += 1
        if flag_start:
            break
        time.sleep(5)
        if cnt % 40 == 0:
            start_mysql()
        tps = os.popen(port_open)
        for i_port in tps.readlines():
            if port in i_port and "mysqld" in i_port:
                flag_start = True
                break
    print("MySQL port is open and listening.")

    flag_start = False
    cnt = 0
    # Step 2: Check if MySQL process (mysqld) is running
    while True:
        cnt += 1
        if flag_start:
            break
        time.sleep(5)
        if cnt % 40 == 0:
            start_mysql()
        tps = os.popen(pid_open)
        list_pid = tps.readlines()
        for i_pid in list_pid:
            if mysql_safe_open in i_pid:
                flag_start = True

    print("MySQL process is running.")

    cnt = 0
    flag_start = False
    # Step 3: Check systemctl service status for active (running)
    while True:
        cnt += 1
        if flag_start:
            break
        time.sleep(5)
        if cnt % 40 == 0:
            start_mysql()
        status = os.popen(status_mysql_cmd)
        list_status = status.readlines()
        for i_status in list_status:
            if "active (running)" in i_status:
                flag_start = True
                break
    print("MySQL service is active.")

    # Step 4: Wait 70 seconds after confirming active status to ensure the server is fully ready
    print("Waiting 60 seconds to ensure MySQL server is fully initialized.")
    time.sleep(70)
    print("Wait complete, MySQL server should be ready.")

# Function to clear system cache before benchmarking
def free_cache():
    """
    Clear the system file cache to avoid OS-level caching effects before running benchmarks.
    """
    print("Clearing system cache...")
    if os.system(free_cache_cmd) == 0:
        print("System cache cleared successfully.")

if __name__ == "__main__":
    # Example main execution flow
    free_cache()        # Clear system cache
    start_mysql()       # Start MySQL server
    status_mysql()      # Check MySQL server status
    stop_mysql()        # Stop MySQL server

    # # Uncomment the following lines to enable sar monitoring
    # print(sar_u_com)
    # os.system(sar_u_com)
    # time.sleep(100)
    # print("Finished sar CPU monitoring.")
    # os.system(clean_sar_u_com)   # Clear sar CPU usage log after monitoring
