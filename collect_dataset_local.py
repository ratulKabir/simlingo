"""
Generates a dataset for training locally.
Each route file is parallelized using Python's multiprocessing.
Monitors the data collection and continues crashed processes.
"""

from datetime import datetime
import os
import time
import glob
import json
import multiprocessing as mp
from pathlib import Path
import random
import re
import subprocess
import signal
import psutil

def run_carla_server(carla_root, port, streaming_port):
    """Start CARLA server with the given ports"""
    print(f"Starting CARLA server on port {port} with streaming port {streaming_port}")
    cmd = f"{carla_root}/CarlaUE4.sh --world-port={port} -RenderOffScreen -nosound -graphicsadapter=0 -quality-level=Low -carla-streaming-port={streaming_port}"
    print(f"Running command: {cmd}")
    process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    print("Waiting for CARLA to start...")
    time.sleep(10)  # Give CARLA time to start
    print("CARLA should be ready now")
    return process

def run_data_collection(args):
    """Run a single data collection process"""
    route_file, code_dir, agent_name, checkpoint_endpoint, save_path, seed, carla_root, town, repetition, ports = args
    
    print(f"\nStarting data collection for route: {route_file}")
    print(f"Using ports: streaming={ports[0]}, world={ports[1]}, tm={ports[2]}")
    
    # Set environment variables
    env = os.environ.copy()
    env.update({
        'SCENARIO_RUNNER_ROOT': f"{code_dir}/scenario_runner_autopilot",
        'LEADERBOARD_ROOT': f"{code_dir}/leaderboard_autopilot",
        'CARLA_ROOT': carla_root,
        'PYTHONPATH': f"{env.get('PYTHONPATH', '')}:{carla_root}/PythonAPI/carla:leaderboard_autopilot:scenario_runner_autopilot",
        'REPETITIONS': '1',
        'DEBUG_CHALLENGE': '0',
        'TEAM_AGENT': agent_name,
        'CHALLENGE_TRACK_CODENAME': 'MAP',
        'ROUTES': route_file,
        'TOWN': town,
        'REPETITION': str(repetition),
        'TM_SEED': str(seed),
        'CHECKPOINT_ENDPOINT': checkpoint_endpoint,
        'TEAM_CONFIG': route_file,
        'RESUME': '1',
        'DATAGEN': '1',
        'SAVE_PATH': save_path
    })

    print("Environment variables set")

    # Start CARLA server
    streaming_port, world_port, tm_port = ports
    print("\nStarting CARLA server...")
    server_process = run_carla_server(carla_root, world_port, streaming_port)
    print("CARLA server started")

    try:
        # Run the data collection
        cmd = f"python leaderboard/leaderboard/leaderboard_evaluator_local.py --port={world_port} \
            --traffic-manager-port={tm_port} --traffic-manager-seed={seed} --routes={route_file} \
            --repetitions=1 --track=MAP --checkpoint={checkpoint_endpoint} --agent={agent_name} \
            --agent-config={route_file} --debug=0 --resume=1 --timeout=600"
        
        print(f"\nRunning data collection command:\n{cmd}\n")
        process = subprocess.run(cmd, shell=True, env=env)
        success = process.returncode == 0
        print(f"Data collection {'succeeded' if success else 'failed'} with return code {process.returncode}")
        return success
    finally:
        print("\nCleaning up CARLA server...")
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        time.sleep(2)
        print("Cleanup complete")

def find_free_ports(n=3, start_range=10000):
    """Find n available ports starting from start_range"""
    def is_port_in_use(port):
        for conn in psutil.net_connections():
            if conn.laddr.port == port:
                return True
        return False

    ports = []
    current_port = start_range
    while len(ports) < n:
        if not is_port_in_use(current_port):
            ports.append(current_port)
        current_port += 1
    return ports

def main():
    # Configuration
    repetitions = 1
    repetition_start = 0
    num_parallel_instances = 1  # Set to 1 for single CARLA instance
    code_root = r"/home/ratul/Workstation/ratul/simlingo"
    carla_root = "/home/ratul/software/carla0915"
    date = datetime.today().strftime("%Y_%m_%d")
    dataset_name = "simlingo_v2_" + date
    root_folder = r"database/"  # With ending slash
    data_save_directory = root_folder + dataset_name

    route_folder = f"{code_root}/data/simlingo"

    # Comment out original route collection
    routes = glob.glob(f"{route_folder}/**/*balanced*/*.xml", recursive=True)
    routes_lb1 = glob.glob(f"{route_folder}/**/*lb1*/**/*.xml", recursive=True)
    routes = routes + routes_lb1
    
    # # Add single route for testing
    # routes = [f"{route_folder}/training_3_scenarios/routes_training/random_weather_seed_3_balanced_100/575.xml"]

    # Shuffle routes
    random.seed(42)
    random.shuffle(routes)

    # Create necessary directories
    Path(data_save_directory).mkdir(parents=True, exist_ok=True)

    # Prepare arguments for parallel processing
    process_args = []
    seed_counter = 1000000 * repetition_start - 1

    for repetition in range(repetition_start, repetitions):
        for route in routes:
            seed_counter += 1

            # Limit to 5 routes for testing
            if seed_counter > 50:  # Changed from 100 to 5
                break

            try:
                town = re.search('Town(\\d+)', route).group(0)
            except:
                if 'validation' in route:
                    town = 'Town13'
                elif 'training' in route:
                    town = 'Town12'
                else:
                    print(f"Town not found in route {route}")
                    continue

            scenario_type = route.split("/")[-5:-1]
            scenario_type = "/".join(scenario_type)
            routefile_number = route.split("/")[-1].split(".")[0]

            ckpt_endpoint = f"{code_root}/{data_save_directory}/results/{scenario_type}/{routefile_number}_result.json"
            save_path = f"{code_root}/{data_save_directory}/data/{scenario_type}"
            Path(save_path).mkdir(parents=True, exist_ok=True)
            
            agent = f"{code_root}/team_code/data_agent.py"
            
            # Find free ports for this process
            ports = find_free_ports(3, 10000 + len(process_args) * 3)
            
            args = (route, code_root, agent, ckpt_endpoint, save_path, seed_counter, 
                   carla_root, town, repetition, ports)
            process_args.append(args)

    # Run data collection in parallel
    max_parallel_processes = min(num_parallel_instances, mp.cpu_count() - 1)  # Use specified number but don't exceed CPU count - 1
    print(f"Starting data collection with {max_parallel_processes} parallel processes")
    
    with mp.Pool(max_parallel_processes) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(run_data_collection, process_args)):
            print(f"Completed {i+1}/{len(process_args)} routes. Success: {result}")
            results.append(result)

    # Print final statistics
    successful = sum(results)
    total = len(results)
    print(f"\nData collection completed:")
    print(f"Total routes: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")

if __name__ == "__main__":
    main() 