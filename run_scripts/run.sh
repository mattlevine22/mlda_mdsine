nohup python -u run.py --run_id 0 --run_all 0 --project_name mdsine2_v4  > output_mech$i.log &
echo $! > save_mech$i.txt
