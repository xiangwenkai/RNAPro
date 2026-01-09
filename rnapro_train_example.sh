export LAYERNORM_TYPE=torch

python3 ./runner/train.py \
--model_name rnapro_mini \
--run_name rnapro_mini \
--seed 43 --base_dir ./results/rnapro_mini \
--dtype bf16 \
--project rnapro \
--use_wandb false \
--diffusion_batch_size 16 \
--eval_interval 50000 \
--log_interval 5 \
--checkpoint_interval 1000 \
--ema_decay 0.995 \
--train_crop_size 200 \
--max_steps 1000000 \
--warmup_steps 50 \
--lr 0.0001 \
--sample_diffusion.N_step 20 \
--load_checkpoint_path ./release_data/protenix_models/protenix_mini_default_v0.5.0.pt \
--load_ema_checkpoint_path ./release_data/protenix_models/protenix_mini_default_v0.5.0.pt \
--data.train_sets ./release_data/kaggle/ \
--load_strict False \
--use_msa True \
--msa_dir ./release_data/kaggle/MSA_v2 \
--model.use_RibonanzaNet2 True \
--model.ribonanza_net_path ./release_data/ribonanzanet2_checkpoint \
--use_template ca_precomputed \
--model.use_template ca_precomputed \
--lr_scheduler constant \
--load_params_only True \
--model.template_embedder.n_blocks 2 \
--triangle_attention cuequivariance \
--triangle_multiplicative cuequivariance

