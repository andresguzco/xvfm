# Generate sweeps for all yaml config files

python yaml_to_sh.py sweeps/beijing.yaml sweeps/beijing.sh --qos=normal --array_max_active=5 --array=40
python yaml_to_sh.py sweeps/news.yaml sweeps/news.sh --qos=m --array_max_active=5 --array=40
python yaml_to_sh.py sweeps/adult.yaml sweeps/adult.sh --qos=m2 --array_max_active=5 --array=40
python yaml_to_sh.py sweeps/default.yaml sweeps/default.sh --qos=m3 --array_max_active=5 --array=40
python yaml_to_sh.py sweeps/shoppers.yaml sweeps/shoppers.sh --qos=m4 --array_max_active=5 --array=40
python yaml_to_sh.py sweeps/magic.yaml sweeps/magic.sh --qos=m5 --array_max_active=5 --array=40

