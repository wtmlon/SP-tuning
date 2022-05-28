#export CUDA_VISIBLE_DEVICES=0

#for task in SST-2 sst-5 mr cr mpqa subj trec CoLA MNLI MNLI-mm SNLI QNLI RTE-glue MRPC QQP
#do
#done
#task=$1

#python3 run.py   \
#    --encoder lstm \
#    --task ${task}  \
#    --extra_mask_rate 0.1   \
#    --x_input mix   \
#    --aug \
#    --learning_rate 2e-4 \
#    --weight_decay 0.05 \
#    --pet_per_gpu_train_batch_size 8


# connect the network
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com"
export WANDB_API_KEY=be340acd2d3396078ae82c556d462b42a457dfad
LC_ALL=en_US
export LC_ALL

task=$1
# test
#python3 run.py   \
#    --encoder inner \
#    --task ${task}  \
#    --learning_rate 5e-5 \
#    --weight_decay 0.01 \


# generating .yaml file
##python3 my_sweep.py \
##        --task ${task} \
##        --x_input mix \
##        --encoder lstm \
##        --aug \
##        --extra_mask_rate 0.1 \

# generating .yaml file
#python3 my_sweep.py   \
#        --encoder lstm \
#        --task ${task}  \
#        --extra_mask_rate 0.1   \
#        --x_input replace   \
#        --warmup    \
#        --warmup_lr 1e-4    \
#        --aug

# sweep
# mix
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/QQP_es2/pmaobq2d
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/QQP_es3/i15l66qg
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/mr_es2/8hclotdk
CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/mr_es4/whnqnmxs

# mix with warmup
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/SST-2_replace_es/qh10go9x
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/cr_replace_es/hn0uoy77
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/mr_replace_es/dqou4fwc
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/trec_replace_es/41ovtzvf
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/subj_replace_es/5xw6u4yp
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/QNLI_replace_es/sc32wa1i
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/MNLI_replace_es/5ohxb8o2
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/SNLI_replace_es/l4drn1ug
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/MRPC_replace_es/11y7tblf

# replace
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/replace_cr_es/h7rclumf
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/replace_mr_es/qxxro38h
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/replace_SST-2_es/1ab6xs3y
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/replace_subj_es/hn0g5ocd
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/replace_trec_es/a2988txy
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/replace_QNLI_es/zqvzfk6s
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/replace_SNLI_es/yd0rauwx
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/replace_MNLI_es/f7gnkb44
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/replace_MRPC_es/mk4s09ik
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/replace_QQP_es/p8riae66
#CUDA_VISIBLE_DEVICES=0 wandb agent szu_csse_bdi/replace_QQP_es2/6ledy52h