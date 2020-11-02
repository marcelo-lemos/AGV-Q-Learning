#!/bin/sh

# echo python3 src/qlearning.py --input "$1" --alpha "$2" --epsilon "$3" --gamma "$4" --episodes "$5"
python3 src/qlearning.py --input "$1" --alpha "$2" --epsilon "$3" --gamma "$4" --episodes "$5"
