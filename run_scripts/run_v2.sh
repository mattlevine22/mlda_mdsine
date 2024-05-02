# Run the nominal GLV model (no training)
nohup python -u run_nominalGLV.py --run_id 0 --run_all 0 --project_name mdsine2_v5  > output_nominal.log &
echo $! > save_nominal.txt

# Train the A, r parameters of the GLV model (different learning rate configurations)
for i in {0..6}
do
    nohup python -u run_tuneGLV.py --run_id $i --run_all 0 --project_name mdsine2_v5  > output_tune$i.log &
    echo $! > save_tune$i.txt
done

# Train a pure Markovian Neural ODE model
for i in {0..17}
do
    nohup python -u run_pureMarkNODE.py --run_id $i --run_all 0 --project_name mdsine2_v5  > output_pureMarkNODE$i.log &
    echo $! > save_pureMarkNODE$i.txt
done

# Train a pure Non-Markovian Neural ODE model
for i in {0..35}
do
    nohup python -u run_pureNonMarkNODE.py --run_id $i --run_all 0 --project_name mdsine2_v5  > output_pureNonMarkNODE$i.log &
    echo $! > save_pureNonMarkNODE$i.txt
done