#!/bin/bash

source common.sh
TMPFILE="/tmp/etnet_tmp"
TMPFILE2="/tmp/etnet_tmp2"

if [[ $# -eq 0 ]]
then
    echo "Usage: $(basename $0) [stock code]"
    exit
fi

###################################################
STOCKCODE=$1
STOCKCODE_ZEROPADDED=$(printf "%04d" $STOCKCODE)

###################################################
wget -O $TMPFILE "http://www.etnet.com.hk/www/eng/stocks/realtime/quote_ci_pl.php?code="$STOCKCODE"&quarter=final"

# drawBarChart('', 'np', 'Profit / (Loss) Attributable to Shareholders', ["2011","2012","2013","2014","2015"], [32210000000,17410000000,1.3291e+10,1.1069e+10,1.3429e+10], 'K','HKD');
EARNINGS_FILE=$FUNDL_INFO_FOLDER/historical_earnings.csv
cat $EARNINGS_FILE | grep -v ^$STOCKCODE_ZEROPADDED > $TMPFILE2
cat $TMPFILE2 > $EARNINGS_FILE
echo -n $STOCKCODE_ZEROPADDED >> $EARNINGS_FILE
echo -n "," >> $EARNINGS_FILE
cat $TMPFILE | grep drawBarChart | grep "Attributable to Shareholders" | sed -e 's/^.*\], \[//' | sed -e 's/\].*$//' >> $EARNINGS_FILE

# drawBarChart('', 'eps', 'EPS (cts)', ["2011","2012","2013","2014","2015"], [2141,1157,883,736,893], 'cts','HKD');
EPS_FILE=$FUNDL_INFO_FOLDER/historical_eps.csv
cat $EPS_FILE | grep -v ^$STOCKCODE_ZEROPADDED > $TMPFILE2
cat $TMPFILE2 > $EPS_FILE
echo -n $STOCKCODE_ZEROPADDED >> $EPS_FILE
echo -n "," >> $EPS_FILE
cat $TMPFILE | grep drawBarChart | grep "EPS" | sed -e 's/^.*\], \[//' | sed -e 's/\].*$//' >> $EPS_FILE

###################################################
wget -O $TMPFILE "http://www.etnet.com.hk/www/eng/stocks/realtime/quote_ci_bs.php?code="$STOCKCODE"&quarter=final"

# drawBarChart('', 'totnoncurasset', "Non-current Assets(total)", ["2011","2012","2013","2014","2015"], [272351000000,2.95717e+11,317309000000,325855000000,331136000000], 'K', 'HKD');
NONCURAST_FILE=$FUNDL_INFO_FOLDER/historical_noncurasset.csv
cat $NONCURAST_FILE | grep -v ^$STOCKCODE_ZEROPADDED > $TMPFILE2
cat $TMPFILE2 > $NONCURAST_FILE
echo -n $STOCKCODE_ZEROPADDED >> $NONCURAST_FILE
echo -n "," >> $NONCURAST_FILE
cat $TMPFILE | grep drawBarChart | grep "Non-current Assets" | sed -e 's/^.*\], \[//' | sed -e 's/\].*$//' | tr -d '\n' >> $NONCURAST_FILE
echo >> $NONCURAST_FILE

# drawBarChart('', 'totcurrassets', "Current Assets(total)", ["2011","2012","2013","2014","2015"], [20312000000,26461000000,31716000000,31480000000,31229000000], 'K', 'HKD');
CURAST_FILE=$FUNDL_INFO_FOLDER/historical_curasset.csv
cat $CURAST_FILE | grep -v ^$STOCKCODE_ZEROPADDED > $TMPFILE2
cat $TMPFILE2 > $CURAST_FILE
echo -n $STOCKCODE_ZEROPADDED >> $CURAST_FILE
echo -n "," >> $CURAST_FILE
cat $TMPFILE | grep drawBarChart | grep "Current Assets" | sed -e 's/^.*\], \[//' | sed -e 's/\].*$//' | tr -d '\n' >> $CURAST_FILE
echo >> $CURAST_FILE

###################################################
rm -f $TMPFILE $TMPFILE2
