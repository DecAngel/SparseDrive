#!/bin/bash

#SBATCH --job-name=SparseDrive_t1    		# 作业在调度系统中的作业名
#SBATCH --partition=gpu_v100     		    # 作业提交的指定队列/分区
#SBATCH --nodes=1                   		# 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=1         		# 每节点任务数，GPU任务不需要修改
#SBATCH --cpus-per-task=3           		# V100一张卡默认配置3个CPU核心，gpuB一张卡默认配置12个CPU核心,MIG资源一张卡默认配置6个CPU核心(根据卡数自行调整)
#SBATCH --gres=gpu:4                		# 申请GPU卡
#SBATCH -o slurm/%J.out                   		# 脚本执行的输出将被保存在当 %J.out文件下，%j表示作业号
#SBATCH -e slurm/%J.err                   		# 脚本执行的错误日志将被保存在当 %J.err文件下，%j表示作业号
#SBATCH --mail-type=END,FAIL			# 任务结束，失败时邮件通知
#SBATCH --mail-user=230238536@seu.edu.cn 	# 邮件通知邮箱

module load anaconda3               		# 加载相关依赖
module load cuda-11.6
module load cudn-11.2
module load gcc-9.3.0

# shellcheck disable=SC2164
cd /seu_share/home/fuchenchen/230238536/projects/SparseDrive/
source activate sparse                 		# 如果已经在命令行中激活对应环境，提交脚本时需注释此行，推荐保留此行在base环境下提交任务

# shellcheck disable=SC2164
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../..
#export GPU_NUM=4
#export TOTAL_BATCH_SIZE=32
#sh scripts/train.sh                          	# 执行相关命令
