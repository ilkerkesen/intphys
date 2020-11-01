#!/bin/bash
#SBATCH --partition=debug                     	# Name of Partition
#SBATCH --ntasks=1                            	# Maximum CPU cores for job
#SBATCH --nodes=1                             	# Ensure all cores are from the same node
#SBATCH --time=2                              	# Job should run for up to 5 minutes (for example)
#SBATCH --mail-type=END                       	# Event(s) that triggers email notification (BEGIN,END,FAIL,ALL)
#SBATCH --mail-user=tayfunatesceng@gmail.com    # Destination email address

train    										# Replace with your application's commands