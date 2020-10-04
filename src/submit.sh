#!/bin/bash -l
# Time-stamp: <Sun 2017-06-11 22:13 svarrette>
##################################################################
#SBATCH -N 1
# Exclusive mode is recommended for all spark jobs
#SBATCH --exclusive
#SBATCH --ntasks-per-node 1
### -c, --cpus-per-task=<ncpus>
###     (multithreading) Request that ncpus be allocated per process
#SBATCH -c 28
#SBATCH --time=0-01:00:00   # 1 hour
#
#          Set the name of the job
#SBATCH -J SparkMaster
#          Passive jobs specifications
#SBATCH --partition=batch
#SBATCH --qos qos-batch>>

# Use the RESIF build modules
if [ -f  /etc/profile ]; then
       .  /etc/profile
fi

# Load the {intel | foss} toolchain and whatever module(s) you need
module purge
module use $HOME/.local/easybuild/modules/all
module load devel/Spark

export SPARK_HOME=$EBROOTSPARK

# sbin/start-master.sh - Starts a master instance on the machine the script is executed on.
$SPARK_HOME/sbin/start-all.sh

export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=3
export CORES_PER_WORKER=1
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
export TFoS_HOME="/home/users/yhoffmann/venv/tf-on-spark/lib/python3.6/site-packages/tensorflowonspark"

${SPARK_HOME}/sbin/start-master.sh; ${SPARK_HOME}/sbin/start-slave.sh -c $CORES_PER_WORKER -m 3G $ {MASTER}


# remove any old artifacts
rm -rf ${TFoS_HOME}/lstm_model
rm -rf ${TFoS_HOME}/prices_export

# train
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --conf spark.cores.max=${TOTAL_CORES} \
    --conf spark.task.cpus=${CORES_PER_WORKER} \
    ${TFoS_HOME}/examples/prices/keras/lstm_spark.py \
    --cluster_size ${SPARK_WORKER_INSTANCES} \
    --images_labels ${TFoS_HOME}/data/prices/csv/train \
    --model_dir ${TFoS_HOME}/lstm_model \
    --export_dir ${TFoS_HOME}/prices_export

# confirm model
ls -lR ${TFoS_HOME}/lstm_model
ls -lR ${TFoS_HOME}/lstm_export


# sbin/stop-master.sh - Stops the master that was started via the bin/start-master.sh script.
$SPARK_HOME/sbin/stop-all.sh
