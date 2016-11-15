#!/bin/bash
source common.sh

CUR_PRICE_FILE="../current_prices.csv"
TMPFILE="$ETF_INFO_FOLDER/tmpfile"
cat /dev/null > $CUR_PRICE_FILE

for i in "HK" "US"
do
    if [[ $i == "HK" ]]
    then
        SYMBOL_LIST="$HK_SYMBOL_LIST"
        EXCHANGE="HKG"
    elif [[ $i == "US" ]]
    then
        SYMBOL_LIST="$US_SYMBOL_LIST"
        EXCHANGE="NYSEARCA"
    fi

    for SYMBOL in $SYMBOL_LIST
    do
        wget -O $TMPFILE "https://www.google.com/finance?q="$EXCHANGE"%3A"$SYMBOL"&hl=en"
        echo -n "$SYMBOL," >> $CUR_PRICE_FILE
        cat $TMPFILE | grep ref_ | head -n 1 | sed -e 's/<\/.*$//' | sed -e 's/^.*>//' >> $CUR_PRICE_FILE
    done
done

rm -f $TMPFILE
