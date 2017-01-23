#!/bin/bash

cat $1 | csel , 2 | alz | st 1 o
cat $1 | csel , 1 3 | st 2 o
st p 1 2 | csel , 2 1 3 | st 3 o
st 3 > $1
