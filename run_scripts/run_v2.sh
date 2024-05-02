output_dir=mdsine_v5
# Ensure the output directory exists
mkdir -p "$output_dir"


# Run the nominal GLV model (no training)
nohup python -u run_nominalGLV.py --run_id 0 --run_all 0 --project_name mdsine2_v5  > "$output_dir/output_nominal.log" &
echo $! > "$output_dir/save_nominal.txt"

# Train the A, r parameters of the GLV model (different learning rate configurations)
for i in {0..6}
do
    nohup python -u run_tuneGLV.py --run_id $i --run_all 0 --project_name mdsine2_v5  > "$output_dir/output_tune$i.log" &
    echo $! > "$output_dir/save_tune$i.txt"
done

# Train a pure Markovian Neural ODE model
for i in {0..17}
do
    nohup python -u run_pureMarkNODE.py --run_id $i --run_all 0 --project_name mdsine2_v5  > "$output_dir/output_pureMarkNODE$i.log" &
    echo $! > "$output_dir/save_pureMarkNODE$i.txt"
done

# Train a pure Non-Markovian Neural ODE model
for i in {0..35}
do
    nohup python -u run_pureNonMarkNODE.py --run_id $i --run_all 0 --project_name mdsine2_v5  > "$output_dir/output_pureNonMarkNODE$i.log" &
    echo $! > "$output_dir/save_pureNonMarkNODE$i.txt"
done