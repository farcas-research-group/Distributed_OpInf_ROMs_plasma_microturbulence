#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH -A w7x
#SBATCH --partition=normal_q
#SBATCH --mem=32G

module reset
module load Miniconda3
source activate mpipython

MY_MPIEXEC=$(which mpiexec)

# force python to be the one in the same directory as mpiexec
BIN_DIR=$(dirname "$MY_MPIEXEC")
MY_PYTHON="$BIN_DIR/python"

echo "--------------------------------------------------"
echo "ENVIRONMENT DEBUG:"
echo "MPI Path:    $MY_MPIEXEC"
echo "Python Path: $MY_PYTHON"
echo "--------------------------------------------------"

RANKS_TO_TEST="1 2 8 13 73"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
# create slug
RANKS_FILENAME_SUFFIX=$(echo "$RANKS_TO_TEST" | tr ' ' '-')
RESULT_FILE="results_${TIMESTAMP}_ranks-${RANKS_FILENAME_SUFFIX}.txt"

echo "Type     | Ranks | Checksum Result" > "$RESULT_FILE"
echo "---------|-------|----------------------------------" >> "$RESULT_FILE"
echo "Results will be saved to: $RESULT_FILE"

echo ""
echo "=================================================="
echo "Running SERIAL Verification..."
echo "=================================================="

"$MY_PYTHON" src/tinkercliffs_serial_verify.py | tee serial.tmp
SERIAL_VAL=$(grep "SERIAL Checksum" serial.tmp | awk -F': ' '{print $2}' | tr -d '\r')

if [ -z "$SERIAL_VAL" ]; then
    SERIAL_VAL="Error/Not Found"
fi

echo "Serial   |   1   | $SERIAL_VAL" >> "$RESULT_FILE"
rm serial.tmp

for N in $RANKS_TO_TEST; do
    echo ""
    echo "=================================================="
    echo "Running PARALLEL with $N Ranks..."
    echo "=================================================="
    
    "$MY_MPIEXEC" --oversubscribe -n $N "$MY_PYTHON" src/read_parallel.py \
        --sim_index 1 \
        --base_dir "/scratch/jackkelley/experiments_CBC" \
        --output_dir "/scratch/jackkelley/experiments_CBC" \
        --precision float32 \
        --grid 96 16 16 32 8 | tee par_${N}.tmp
    
    # get checksum
    PAR_VAL=$(grep "PARALLEL Q" par_${N}.tmp | awk -F': ' '{print $2}' | tr -d '\r')
    
    if [ -z "$PAR_VAL" ]; then
        PAR_VAL="Error/Not Found"
    fi
    
    echo "Parallel |   $N   | $PAR_VAL" >> "$RESULT_FILE"
    rm par_${N}.tmp
done

echo ""
echo "##################################################"
echo "             FINAL CHECKSUM COMPARISON            "
echo "##################################################"
cat "$RESULT_FILE"
echo "##################################################"
echo "Saved to $RESULT_FILE"