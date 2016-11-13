#!/bin/bash

python mvo.py | grep -v Terminated 2>&1
