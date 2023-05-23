task=$1
cat output/$task/sa/x_input-replace/warmup-False/origin/16-13/train_log | grep -v None | awk -F '[ }]' 'BEGIN {max = 0} {if ($13+0>max+0) max=$13 fi} END {print "Max=", max}'
cat output/$task/sa/x_input-replace/warmup-False/origin/16-21/train_log | grep -v None | awk -F '[ }]' 'BEGIN {max = 0} {if ($13+0>max+0) max=$13 fi} END {print "Max=", max}'
cat output/$task/sa/x_input-replace/warmup-False/origin/16-42/train_log | grep -v None | awk -F '[ }]' 'BEGIN {max = 0} {if ($13+0>max+0) max=$13 fi} END {print "Max=", max}'
cat output/$task/sa/x_input-replace/warmup-False/origin/16-87/train_log | grep -v None | awk -F '[ }]' 'BEGIN {max = 0} {if ($13+0>max+0) max=$13 fi} END {print "Max=", max}'
cat output/$task/sa/x_input-replace/warmup-False/origin/16-100/train_log | grep -v None | awk -F '[ }]' 'BEGIN {max = 0} {if ($13+0>max+0) max=$13 fi} END {print "Max=", max}'
