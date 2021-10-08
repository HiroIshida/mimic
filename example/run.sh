SCRIPT_DIR=$(cd $(dirname $0); pwd)
python3 $SCRIPT_DIR/kuka_reaching.py -n 3
python3 -m mimic.scripts.train -pn kuka_reaching -n 3
