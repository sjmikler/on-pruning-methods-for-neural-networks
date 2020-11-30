import subprocess
from subprocess import PIPE
import threading
import time
import signal

hosts = open('hosts.txt', 'r').read().split('\n')


def print_host_outputs(process, host_name):
    while process.poll() is None:
        time.sleep(0.2)
        line = process.stderr.readline().decode().strip()
        if line:
            print(f"{host_name:^40} STDERR <<< {line}", end='\n')

        line = process.stdout.readline().decode().strip()
        if line:
            print(f"{host_name:^40} STDOUT <<< {line}", end='\n')

    for line in process.stderr.readlines():
        line = line.decode().strip()
        print(f"{host_name:^40} STDERR <<< {line}", end='\n')

    for line in process.stdout.readlines():
        line = line.decode().strip()
        print(f"{host_name:^40} STDOUT <<< {line}", end='\n')
    print(f"{host_name:^40} <<< FINISHED WITH CODE {process.poll()}")

processes = []

while True:
    command = 'ssh {h} ' + input()
    if '{h}' in command:
        for host in hosts:
            print(">>>", command.format(h=host))
            process = subprocess.Popen(command.format(h=host),)
            processes.append(process)
                                       # shell=True,
                                       # stderr=PIPE,
                                       # stdin=PIPE,
                                       # stdout=PIPE)
            # thread = threading.Thread(target=print_host_outputs, args=[process, host])
            # thread.start()
    else:
        print(">>>", command)
        process = subprocess.Popen(command, shell=True)