## more attack baselines
model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'rpa' 'l2t')

# 174 20
for var1 in 0
do
    # for var2 in 0 1 2 3 4
    # for var2 in 1 2 3 4
    for var2 in 3
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=4 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar10     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar10/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar10/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar10/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to${model[$var1]}_v \
                      --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done





for var1 in 0
do
    # for var2 in 0 1 2 3 4
    # for var2 in 1 2 3 4
    for var2 in 3
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=5 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar10     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar10/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar10/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar10/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_jpeg30_to${model[$var1]}_v \
                     --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0       \
                      --jpeg           \
                      --purify_hp               75   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done

for var1 in 0
do
    # for var2 in 0 1 2 3 4
    for var2 in 3
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=5 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar10     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar10/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar10/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar10/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_jpeg20_to${model[$var1]}_v \
                     --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0       \
                      --jpeg           \
                      --purify_hp               50   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


for var1 in 0
do
    # for var2 in 0 1 2 3 4
    # for var2 in 1 2 3 4
    for var2 in 3
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=6 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar10     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar10/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar10/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar10/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_bdr_to${model[$var1]}_v \
                     --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0       \
                      --bdr           \
                      --purify_hp               20   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done

for var1 in 0
do
    # for var2 in 0 1 2 3 4
    for var2 in 3
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=6 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar10     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar10/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar10/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar10/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_gf3.0_to${model[$var1]}_v \
                     --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --gaussian_filter           \
                      --purify_hp               0.7  \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done




for var1 in 0
do
    # for var2 in 0 1 2 3 4
    # for var2 in 1 2 3 4
    for var2 in 3
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=7 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar10     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar10/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar10/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar10/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_gf1.2_to${model[$var1]}_v \
                     --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --gaussian_filter           \
                      --purify_hp               0.6  \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done

for var1 in 0
do
    # for var2 in 0 1 2 3 4
    for var2 in 3
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=7 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar10     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar10/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar10/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar10/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_rp1.1_to${model[$var1]}_v \
                     --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --resize_and_padding       \
                      --purify_hp               1.2  \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


# 176 4
for var1 in 0
do
    # for var2 in 0 1 2 3 4
    for var2 in 4
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=6 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar10     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar10/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar10/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar10/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_nrp_to${model[$var1]}_v \
                     --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --nrp       \
                      --purify_hp               1.1  \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


# 175 0
for var1 in 0
do
    # for var2 in 0 1 2 3 4
    for var2 in 1
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=2 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar10     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar10/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar10/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar10/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_diffpure_to${model[$var1]}_v \
                     --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --diffpure       \
                      --purify_hp               50  \
                      --t                       100 \
                      --score_type              score_sde \
                      --eval_batch_size         2 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


# var1=0
# var2=4
# var3=5

# CUDA_VISIBLE_DEVICES=9 nohup python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar10     \
#                       --version                 ${model[$var1]}                    \
#                       --version_weight_path     ./experiments/cifar10/clean/${model[$var1]}/checkpoints/${model[$var1]} \
#                       --version_s               ${model_s[$var2]}                    \
#                       --version_s_weight_path   ./experiments/cifar10/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
#                       --exp_name                rebuttal/experiments/cifar10/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_diffpure_to${model[$var1]}_v \
#                      --train_data_type         CIFAR10_Tensor                       \
#                       --test_data_type          CIFAR10_Tensor                    \
#                       --adv_type                ${adv_type[$var3]}                         \
#                       --epsilon                 8.0 \
#                       --diffpure       \
#                       --purify_hp               50  \
#                       --t                       100 \
#                       --score_type              score_sde \
#                       --eval_batch_size         64 &


# 176 7
model=('resnet18_madrys' 'resnet50_madrys' 'vgg19_madrys' 'mobilenet_v2_cifar_madrys' 'dense121_madrys' 'inc_v4_madrys')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'rpa' 'l2t')

for var1 in 0
do
    # for var2 in 0 1 2 3 4
    for var2 in 3
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=3 python -u eval_transfer_robustness_diffpure_cifar_madrys.py    --config_path configs/cifar10     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar10/AT/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar10/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar10/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to${model[$var1]} \
                     --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


model=('resnet18_madrys' 'resnet50_madrys' 'vgg19_madrys' 'mobilenet_v2_cifar_madrys' 'dense121_madrys' 'inc_v4_madrys')
model_s=('resnet18_madrys' 'resnet50_madrys' 'vgg19_madrys' 'mobilenet_v2_cifar_madrys' 'dense121_madrys' 'inc_v4_madrys')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'rpa' 'l2t')

for var1 in 0
do
    for var2 in 0 1 2 3 4
    # for var2 in 3 4 5
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=7 python -u eval_transfer_robustness_diffpure_cifar_madrys.py    --config_path configs/cifar10     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar10/AT/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar10/AT/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar10/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_to${model[$var1]} \
                     --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


# 176 8,8 175 3 178 4,5
model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'rpa' 'l2t')
epsilon_defense=(1 2 4 8 16 32 64)

# for var1 in 4
for var1 in 0 1
# for var1 in 3 4
do
    # for var2 in 0 1 2 3 4
    for var2 in 3
    do
        for var3 in 6
        do
            for var4 in 2
            do
                CUDA_VISIBLE_DEVICES=9 python -u eval_transfer_robustness_v2_cifar.py    --config_path configs/cifar10     \
                          --version                 ${model[$var1]}                    \
                          --version_weight_path     ./experiments/cifar10/transfer_robustness_v2/${model[$var1]}_learn_t_e${epsilon_defense[$var4]}/${model[$var1]}_e100/checkpoints/${model[$var1]}_e100 \
                          --version_s               ${model_s[$var2]}                    \
                          --version_s_weight_path   ./experiments/cifar10/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                          --exp_name                rebuttal/experiments/cifar10/eval_transfer_robustness_v2_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to_${model[$var1]}_learn_t_e${epsilon_defense[$var4]} \
                          --train_data_type         CIFAR10_Tensor                       \
                          --test_data_type          CIFAR10_Tensor                    \
                      	  --adv_type                ${adv_type[$var3]}                         \
                          --epsilon                 8.0 \
                          --eval_batch_size         1 \
                      	  --num_steps               10 \
                          --step_size               0.8  
            done
        done
    done
done

# for var1 in 4
for var1 in 2 3
do
    # for var2 in 0 1 2 3 4
    for var2 in 3
    do
        for var3 in 6
        do
            for var4 in 6
            do
                CUDA_VISIBLE_DEVICES=9 python -u eval_transfer_robustness_v2_cifar.py    --config_path configs/cifar10     \
                          --version                 ${model[$var1]}                    \
                          --version_weight_path     ./experiments/cifar10/transfer_robustness_v2/${model[$var1]}_fixed_t_e${epsilon_defense[$var4]}/${model[$var1]}_e100/checkpoints/${model[$var1]}_e100 \
                          --version_s               ${model_s[$var2]}                    \
                          --version_s_weight_path   ./experiments/cifar10/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                          --exp_name                rebuttal/experiments/cifar10/eval_transfer_robustness_v2_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to_${model[$var1]}_fixed_t_e${epsilon_defense[$var4]} \
                          --train_data_type         CIFAR10_Tensor                       \
                          --test_data_type          CIFAR10_Tensor                    \
                          --adv_type                ${adv_type[$var3]}                         \
                          --epsilon                 8.0 \
                          --eval_batch_size         1 \
                          --num_steps               10 \
                          --step_size               0.8  
            done
        done
    done
done





model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
adv_type=('pgd' 'ifgsm' 'mifgsm' 'difgsm' 'naa' 'rpa' 'l2t')
epsilon_defense=(1 2 4 8 16)

for var1 in 5
do
    # for var2 in 0 1 2
    for var2 in 3 4 5
    do
        # for var3 in 4 5
        for var3 in 6
        do
            for var4 in 2
            do
              for var5 in 3
              do
                  CUDA_VISIBLE_DEVICES=9 python -u eval_transfer_robustness_v3.py    --config_path configs/cifar10     \
                            --version                 ${model[$var1]}                    \
                            --version_weight_path     rebuttal/experiments/cifar10/transfer_robustness_v2/${model[$var1]}_learn_t_e${epsilon_defense[$var4]}/${model[$var1]}/checkpoints/${model[$var1]} \
                            --version_s               ${model_s[$var2]}                    \
                            --version_s_weight_path   rebuttal/experiments/cifar10/transfer_robustness_v2/${model_s[$var2]}_learn_t_e${epsilon_defense[$var5]}/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                            --exp_name                rebuttal/experiments/cifar10/eval_transfer_robustness_v5_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_learn_t_e${epsilon_defense[$var5]}_to_${model[$var1]}_learn_t_e${epsilon_defense[$var4]} \
                            --train_data_type         cifar10Mini                       \
                            --train_data_path         ../cifar10               \
                            --test_data_type          cifar10Mini                       \
                            --test_data_path          ../cifar10               \
                            --adv_type                ${adv_type[$var3]}                         \
                            --epsilon                 8.0 \
                          	--eval_batch_size         2
              done
            done
        done
    done
done



model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
adv_type=('pgd' 'ifgsm' 'mifgsm' 'difgsm' 'naa' 'rpa' 'l2t')
epsilon_defense=(1 2 4 8 16 32 64)

for var1 in 4
do
    # for var2 in 0 1 2
    for var2 in 3 4 5
    do
        # for var3 in 4 5
        for var3 in 6
        do
            for var4 in 6
            do
              for var5 in 5
              do
                  CUDA_VISIBLE_DEVICES=0 python -u eval_transfer_robustness_v3.py    --config_path configs/cifar10     \
                            --version                 ${model[$var1]}                    \
                            --version_weight_path     rebuttal/experiments/cifar10/transfer_robustness_v2/${model[$var1]}_fixed_t_e${epsilon_defense[$var4]}/${model[$var1]}/checkpoints/${model[$var1]} \
                            --version_s               ${model_s[$var2]}                    \
                            --version_s_weight_path   rebuttal/experiments/cifar10/transfer_robustness_v2/${model_s[$var2]}_fixed_t_e${epsilon_defense[$var5]}/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                            --exp_name                rebuttal/experiments/cifar10/eval_transfer_robustness_v5_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_fixed_t_e${epsilon_defense[$var5]}_to_${model[$var1]}_fixed_t_e${epsilon_defense[$var4]} \
                            --train_data_type         cifar10Mini                       \
                            --train_data_path         ../cifar10               \
                            --test_data_type          cifar10Mini                       \
                            --test_data_path          ../cifar10               \
                            --adv_type                ${adv_type[$var3]}                         \
                            --epsilon                 8.0 \
                                  --eval_batch_size         2
              done
            done
        done
    done
done



## cifar100
model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'rpa' 'l2t')

# 175 5 6 7 (1 3 7)
for var1 in 0
do
    for var2 in 0 1 2 3 4
    # for var2 in 1 2 3 4
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=1 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar100     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar100/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar100/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar100/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to${model[$var1]}_v \
                      --train_data_type         CIFAR100                       \
                      --test_data_type          CIFAR100                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done

for var1 in 0
do
    for var2 in 0 1 2 3 4
    # for var2 in 1 2 3 4
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=1 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar100     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar100/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar100/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar100/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_jpeg30_to${model[$var1]}_v \
                     --train_data_type         CIFAR100                       \
                      --test_data_type          CIFAR100                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0       \
                      --jpeg           \
                      --purify_hp               75   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done

for var1 in 0
do
    for var2 in 0 1 2 3 4
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=1 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar100     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar100/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar100/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar100/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_jpeg20_to${model[$var1]}_v \
                     --train_data_type         CIFAR100                       \
                      --test_data_type          CIFAR100                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0       \
                      --jpeg           \
                      --purify_hp               50   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


for var1 in 0
do
    for var2 in 0 1 2 3 4
    # for var2 in 1 2 3 4
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=3 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar100     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar100/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar100/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar100/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_bdr_to${model[$var1]}_v \
                     --train_data_type         CIFAR100                       \
                      --test_data_type          CIFAR100                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0       \
                      --bdr           \
                      --purify_hp               20   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done

for var1 in 0
do
    for var2 in 0 1 2 3 4
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=3 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar100     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar100/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar100/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar100/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_gf3.0_to${model[$var1]}_v \
                     --train_data_type         CIFAR100                       \
                      --test_data_type          CIFAR100                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --gaussian_filter           \
                      --purify_hp               0.7  \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


for var1 in 0
do
    for var2 in 0 1 2 3 4
    # for var2 in 1 2 3 4
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=3 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar100     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar100/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar100/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar100/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_gf1.2_to${model[$var1]}_v \
                     --train_data_type         CIFAR100                       \
                      --test_data_type          CIFAR100                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --gaussian_filter           \
                      --purify_hp               0.6  \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


for var1 in 0
do
    for var2 in 0 1 2 3 4
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=7 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar100     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar100/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar100/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar100/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_rp1.1_to${model[$var1]}_v \
                     --train_data_type         CIFAR100                       \
                      --test_data_type          CIFAR100                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --resize_and_padding       \
                      --purify_hp               1.2  \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done

for var1 in 0
do
    for var2 in 0 1 2 3 4
    # for var2 in 4
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=7 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar100     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar100/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar100/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar100/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_nrp_to${model[$var1]}_v \
                     --train_data_type         CIFAR100                       \
                      --test_data_type          CIFAR100                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --nrp       \
                      --purify_hp               1.1  \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


# 172 9 10 11 12
for var1 in 0
do
    for var2 in 0 1 2 3 4
    # for var2 in 1
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=0 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar100     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar100/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar100/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar100/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_diffpure_to${model[$var1]}_v \
                      --train_data_type         CIFAR100                       \
                      --test_data_type          CIFAR100                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --diffpure       \
                      --purify_hp               50  \
                      --t                       100 \
                      --score_type              score_sde \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


# 172 10 11 12
model=('resnet18_madrys' 'resnet50_madrys' 'vgg19_madrys' 'mobilenet_v2_cifar_madrys' 'dense121_madrys' 'inc_v4_madrys')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'rpa' 'l2t')

for var1 in 0
do
    for var2 in 0 1 2 3 4
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=1 python -u yuyi/model_key/eval_transfer_robustness_diffpure_cifar_madrys.py    --config_path yuyi/model_key/configs/cifar100     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     yuyi/model_key/./experiments/cifar100/AT/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   yuyi/model_key/./experiments/cifar100/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                yuyi/model_key/rebuttal/experiments/cifar100/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to${model[$var1]} \
                     --train_data_type         CIFAR100                       \
                      --test_data_type          CIFAR100                    \
                      --train_data_path         yuyi/model_key/../datasets  \
                      --test_data_path          yuyi/model_key/../datasets  \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


model=('resnet18_madrys' 'resnet50_madrys' 'vgg19_madrys' 'mobilenet_v2_cifar_madrys' 'dense121_madrys' 'inc_v4_madrys')
model_s=('resnet18_madrys' 'resnet50_madrys' 'vgg19_madrys' 'mobilenet_v2_cifar_madrys' 'dense121_madrys' 'inc_v4_madrys')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'rpa' 'l2t')

for var1 in 0
do
    for var2 in 0 1 2 3 4
    # for var2 in 3 4 5
    do
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=1 python -u yuyi/model_key/eval_transfer_robustness_diffpure_cifar_madrys.py    --config_path yuyi/model_key/configs/cifar100     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     yuyi/model_key/./experiments/cifar100/AT/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   yuyi/model_key/./experiments/cifar100/AT/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                yuyi/model_key/rebuttal/experiments/cifar100/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_to${model[$var1]} \
                     --train_data_type         CIFAR100                       \
                      --test_data_type          CIFAR100                    \
                      --train_data_path         yuyi/model_key/../datasets  \
                      --test_data_path          yuyi/model_key/../datasets  \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'rpa' 'l2t')
epsilon_defense=(1 2 4 8 16 32 64)

for var1 in 0 1 2 3 4
do
    for var2 in 0 1 2 3 4
    do
        for var3 in 6
        do
            for var4 in 2
            do
                CUDA_VISIBLE_DEVICES=2 python -u yuyi/model_key/eval_transfer_robustness_v2_cifar.py    --config_path yuyi/model_key/configs/cifar100     \
                          --version                 ${model[$var1]}                    \
                          --version_weight_path     yuyi/model_key/./experiments/cifar100/transfer_robustness_v2/${model[$var1]}_learn_t_e${epsilon_defense[$var4]}/${model[$var1]}/checkpoints/${model[$var1]} \
                          --version_s               ${model_s[$var2]}                    \
                          --version_s_weight_path   yuyi/model_key/./experiments/cifar100/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                          --exp_name                yuyi/model_key/rebuttal/experiments/cifar100/eval_transfer_robustness_v2_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to_${model[$var1]}_learn_t_e${epsilon_defense[$var4]} \
                          --train_data_type         CIFAR100                       \
                          --test_data_type          CIFAR100                    \
                      	  --train_data_path         yuyi/model_key/../datasets  \
                      --test_data_path          yuyi/model_key/../datasets  \
                      --adv_type                ${adv_type[$var3]}                         \
                          --epsilon                 8.0 \
                          --eval_batch_size         1 \
                      	  --num_steps               10 \
                          --step_size               0.8  
            done
        done
    done
done

for var1 in 4
do
    for var2 in 0 1 2 3 4
    do
        for var3 in 6
        do
            for var4 in 6
            do
                CUDA_VISIBLE_DEVICES=3 python -u yuyi/model_key/eval_transfer_robustness_v2_cifar.py    --config_path yuyi/model_key/configs/cifar100     \
                          --version                 ${model[$var1]}                    \
                          --version_weight_path     yuyi/model_key/./experiments/cifar100/transfer_robustness_v2/${model[$var1]}_fixed_t_e${epsilon_defense[$var4]}/${model[$var1]}/checkpoints/${model[$var1]} \
                          --version_s               ${model_s[$var2]}                    \
                          --version_s_weight_path   yuyi/model_key/./experiments/cifar100/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                          --exp_name                yuyi/model_key/rebuttal/experiments/cifar100/eval_transfer_robustness_v2_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to_${model[$var1]}_fixed_t_e${epsilon_defense[$var4]} \
                          --train_data_type         CIFAR100                       \
                          --test_data_type          CIFAR100                    \
                          --train_data_path         yuyi/model_key/../datasets  \
                      --test_data_path          yuyi/model_key/../datasets  \
                      --adv_type                ${adv_type[$var3]}                         \
                          --epsilon                 8.0 \
                          --eval_batch_size         1 \
                          --num_steps               10 \
                          --step_size               0.8  
            done
        done
    done
done


## imagenet
model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2' 'dense121' 'inc_v4')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2' 'dense121' 'inc_v4')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'l2t')


# 176 4 then 175 4
for var1 in 0
do
    # for var2 in 0 1 2 3 4
    # for var2 in 5
    for var2 in 2
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=1 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0   \
                      --num_steps               10   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


for var1 in 0
do
    # for var2 in 0 1 2 3 4
    # for var2 in 5
    for var2 in 2
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=2 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_jpeg30_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0       \
                      --num_steps               10   \
                      --jpeg           \
                      --purify_hp               30   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done

for var1 in 0
do
    # for var2 in 0 1 2 3 4
    # for var2 in 5
    for var2 in 2
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=4 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_jpeg20_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0       \
                      --num_steps               10   \
                      --jpeg           \
                      --purify_hp               20   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


for var1 in 0
do
    for var2 in 4
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=8 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0   \
                      --num_steps               10   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
            CUDA_VISIBLE_DEVICES=8 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_jpeg30_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0       \
                      --num_steps               10   \
                      --jpeg           \
                      --purify_hp               30   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
            CUDA_VISIBLE_DEVICES=8 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_jpeg20_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0       \
                      --num_steps               10   \
                      --jpeg           \
                      --purify_hp               20   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done



# 165 5 6 7 8 9 10 then 175 8
for var1 in 0
do
    # for var2 in 0 1 2 3 4
    # for var2 in 5
    for var2 in 1
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=0 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_bdr_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0       \
                      --num_steps               10   \
                      --bdr           \
                      --purify_hp               20   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done





for var1 in 0
do
    # for var2 in 0 1 2 3 4
    # for var2 in 5
    for var2 in 2
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=1 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_gf3.0_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --num_steps               10   \
                      --gaussian_filter           \
                      --purify_hp               3.0  \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done




for var1 in 0
do
    # for var2 in 0 1 2 3 4
    # for var2 in 5
    for var2 in 2
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=2 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_gf1.2_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --num_steps               10   \
                      --gaussian_filter           \
                      --purify_hp               1.2  \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


for var1 in 0
do
    # for var2 in 0 1 2 3 4
    # for var2 in 5
    for var2 in 2
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=3 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_rp1.1_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --num_steps               10   \
                      --resize_and_padding       \
                      --purify_hp               1.1  \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done



for var1 in 0
do
    # for var2 in 3 0 1 2 4
    # for var2 in 5
    for var2 in 2
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=4 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_nrp_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --num_steps               10   \
                      --nrp       \
                      --purify_hp               1.1  \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


# 174 20 175 1 174 21 22 24
model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2' 'dense121' 'inc_v4')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2' 'dense121' 'inc_v4')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'l2t')

# 171 174 21 23
for var1 in 0
do
    for var2 in 2
    # for var2 in 5
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=0 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_diffpure_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --num_steps               10   \
                      --diffpure       \
                      --purify_hp               1.1  \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done



model=('resnet18_madrys' 'resnet50_madrys' 'vgg19_madrys' 'mobilenet_v2_madrys' 'dense121_madrys' 'inc_v4_madrys')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2' 'dense121' 'inc_v4')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'l2t')

for var1 in 0
do
    # for var2 in 0 1 2 3 4
    # for var2 in 5
    for var2 in 2
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=1 python -u eval_transfer_robustness_diffpure_imagenet_madrys.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/AT/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to${model[$var1]} \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --num_steps               10   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


model=('resnet18_madrys' 'resnet50_madrys' 'vgg19_madrys' 'mobilenet_v2_madrys' 'dense121_madrys' 'inc_v4_madrys')
model_s=('resnet18_madrys' 'resnet50_madrys' 'vgg19_madrys' 'mobilenet_v2_madrys' 'dense121_madrys' 'inc_v4_madrys')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'l2t')

for var1 in 0
do
    for var2 in 0 1 2 3 4
    # for var2 in 5
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=6 python -u eval_transfer_robustness_diffpure_imagenet_madrys.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/AT/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/AT/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_to${model[$var1]} \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --num_steps               10   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


# 175 0 2
model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2' 'dense121' 'inc_v4')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2' 'dense121' 'inc_v4')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'l2t')
epsilon_defense=(1 2 4 8 16 32 64)

# for var1 in 0 1 2 3 4 5
for var1 in 4
do
    # for var2 in 0 1 2 3 4 5
    # for var2 in 1 4
    for var2 in 2 3
    do
        for var3 in 5
        do
            for var4 in 2
            do
                CUDA_VISIBLE_DEVICES=4 python -u eval_transfer_robustness_v2_imagenet.py    --config_path configs/imagenet-mini     \
                          --version                 ${model[$var1]}                    \
                          --version_weight_path     experiments/imagenet-mini/transfer_robustness_v2/${model[$var1]}_learn_t_e${epsilon_defense[$var4]}/${model[$var1]}/checkpoints/${model[$var1]} \
                          --version_s               ${model_s[$var2]}                    \
                          --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                          --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_v2_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to_${model[$var1]}_learn_t_e${epsilon_defense[$var4]} \
                          --train_data_type         ImageNetMini                       \
                          --train_data_path         ../imagenet               \
                          --test_data_type          ImageNetMini                       \
                          --test_data_path          ../imagenet               \
                          --adv_type                ${adv_type[$var3]}                         \
                          --epsilon                 8.0 \
                          --num_steps               10   \
                          --eval_batch_size         1 \
	                      --num_steps               10 \
	                      --step_size               0.8  
            done
        done
    done
done




# for var1 in 0 1 2 3 4 5
for var1 in 4
do
    # for var2 in 0 1 2 3 4 5
    for var2 in 2 4
    do
        for var3 in 5
        do
            for var4 in 6
            do
                CUDA_VISIBLE_DEVICES=2 python -u eval_transfer_robustness_v2_imagenet.py    --config_path configs/imagenet-mini     \
                          --version                 ${model[$var1]}                    \
                          --version_weight_path     experiments/imagenet-mini/transfer_robustness_v2/${model[$var1]}_fixed_t_e${epsilon_defense[$var4]}/${model[$var1]}/checkpoints/${model[$var1]} \
                          --version_s               ${model_s[$var2]}                    \
                          --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                          --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_v2_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to_${model[$var1]}_fixed_t_e${epsilon_defense[$var4]} \
                          --train_data_type         ImageNetMini                       \
                          --train_data_path         ../imagenet               \
                          --test_data_type          ImageNetMini                       \
                          --test_data_path          ../imagenet               \
                          --adv_type                ${adv_type[$var3]}                         \
                          --epsilon                 8.0 \
                          --num_steps               10   \
                          --eval_batch_size         1 \
	                      --num_steps               10 \
	                      --step_size               0.8  
            done
        done
    done
done



# diff_pgd
model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2' 'dense121' 'inc_v4')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2' 'dense121' 'inc_v4')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'diff_pgd')


# 175 3
for var1 in 0
do
    # for var2 in 0 1 2 3 4 5
    # for var2 in 5
    for var2 in 4
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=2 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0   \
                      --num_steps               10   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               1.6  
        done
    done
done


# 176 4 5 6 7 8 9
for var1 in 0
do
    for var2 in 0 1 2 3 4 5
    # for var2 in 5
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=4 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_jpeg30_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0       \
                      --num_steps               10   \
                      --jpeg           \
                      --purify_hp               30   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               1.6  
        done
    done
done

for var1 in 0
do
    for var2 in 0 1 2 3 4 5
    # for var2 in 5
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=5 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_jpeg20_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0       \
                      --num_steps               10   \
                      --jpeg           \
                      --purify_hp               20   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               1.6  
        done
    done
done


for var1 in 0
do
    for var2 in 0 1 2 3 4 5
    # for var2 in 5
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=6 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_bdr_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0       \
                      --num_steps               10   \
                      --bdr           \
                      --purify_hp               20   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               1.6  
        done
    done
done





for var1 in 0
do
    for var2 in 0 1 2 3 4 5
    # for var2 in 5
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=7 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_gf3.0_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --num_steps               10   \
                      --gaussian_filter           \
                      --purify_hp               3.0  \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               1.6  
        done
    done
done




for var1 in 0
do
    for var2 in 0 1 2 3 4 5
    # for var2 in 5
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=8 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_gf1.2_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --num_steps               10   \
                      --gaussian_filter           \
                      --purify_hp               1.2  \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               1.6  
        done
    done
done


for var1 in 0
do
    for var2 in 0 1 2 3 4 5
    # for var2 in 5
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=9 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_rp1.1_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --num_steps               10   \
                      --resize_and_padding       \
                      --purify_hp               1.1  \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               1.6  
        done
    done
done


# 178 4 5
for var1 in 0
do
    # for var2 in 0 3 4
    for var2 in 1 2 5
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=1 python -u eval_transfer_robustness_diffpure_imagenet.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_nrp_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --num_steps               10   \
                      --nrp       \
                      --purify_hp               1.1  \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               1.6  
        done
    done
done


# 175 4(5) 8(4) 177 4(0) 5(2) 172 11(3) 171
model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2' 'dense121' 'inc_v4')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2' 'dense121' 'inc_v4')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'diff_pgd')

# 175 4 5 6 
for var1 in 0
do
    for var2 in 2
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=3 python -u yuyi/model_key/eval_transfer_robustness_diffpure_imagenet.py    --config_path yuyi/model_key/configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     yuyi/model_key/./experiments/imagenet-mini/clean/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   yuyi/model_key/./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                yuyi/model_key/rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_diffpure_to${model[$var1]}_v \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         yuyi/model_key/../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          yuyi/model_key/../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --num_steps               10   \
                      --diffpure       \
                      --purify_hp               1.1  \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               1.6  
        done
    done
done



# 165 5 6 
model=('resnet18_madrys' 'resnet50_madrys' 'vgg19_madrys' 'mobilenet_v2_madrys' 'dense121_madrys' 'inc_v4_madrys')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2' 'dense121' 'inc_v4')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'diff_pgd')

for var1 in 0
do
    for var2 in 0 1 2 3 4 5
    # for var2 in 5
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=1 python -u eval_transfer_robustness_diffpure_imagenet_madrys.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/AT/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to${model[$var1]} \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --num_steps               10   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               1.6  
        done
    done
done


model=('resnet18_madrys' 'resnet50_madrys' 'vgg19_madrys' 'mobilenet_v2_madrys' 'dense121_madrys' 'inc_v4_madrys')
model_s=('resnet18_madrys' 'resnet50_madrys' 'vgg19_madrys' 'mobilenet_v2_madrys' 'dense121_madrys' 'inc_v4_madrys')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'diff_pgd')

for var1 in 0
do
    for var2 in 0 1 2 3 4 5
    # for var2 in 5
    do
        for var3 in 5
        do
            CUDA_VISIBLE_DEVICES=2 python -u eval_transfer_robustness_diffpure_imagenet_madrys.py    --config_path configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/imagenet-mini/AT/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/imagenet-mini/AT/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_to${model[$var1]} \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --num_steps               10   \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               1.6  
        done
    done
done


# 165 7 8 9 172 9 10 12
model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2' 'dense121' 'inc_v4')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2' 'dense121' 'inc_v4')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'diff_pgd')
epsilon_defense=(1 2 4 8 16 32 64)

for var1 in 0 1 2 3 4 5
do
    for var2 in 5
    do
        for var3 in 5
        do
            for var4 in 2
            do
                CUDA_VISIBLE_DEVICES=3 python -u yuyi/model_key/eval_transfer_robustness_v2_imagenet.py    --config_path yuyi/model_key/configs/imagenet-mini     \
                          --version                 ${model[$var1]}                    \
                          --version_weight_path     yuyi/model_key/experiments/imagenet-mini/transfer_robustness_v2/${model[$var1]}_learn_t_e${epsilon_defense[$var4]}/${model[$var1]}/checkpoints/${model[$var1]} \
                          --version_s               ${model_s[$var2]}                    \
                          --version_s_weight_path   yuyi/model_key/./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                          --exp_name                yuyi/model_key/rebuttal/experiments/imagenet-mini/eval_transfer_robustness_v2_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to_${model[$var1]}_learn_t_e${epsilon_defense[$var4]} \
                          --train_data_type         ImageNetMini                       \
                          --train_data_path         yuyi/model_key/../imagenet               \
                          --test_data_type          ImageNetMini                       \
                          --test_data_path          yuyi/model_key/../imagenet               \
                          --adv_type                ${adv_type[$var3]}                         \
                          --epsilon                 8.0 \
                          --num_steps               10   \
                          --eval_batch_size         1 \
	                      --num_steps               10 \
	                      --step_size               1.6  
            done
        done
    done
done


for var1 in 0 1 2 3 4 5
do
    for var2 in 5
    do
        for var3 in 5
        do
            for var4 in 6
            do
                CUDA_VISIBLE_DEVICES=3 python -u yuyi/model_key/eval_transfer_robustness_v2_imagenet.py    --config_path yuyi/model_key/configs/imagenet-mini     \
                          --version                 ${model[$var1]}                    \
                          --version_weight_path     yuyi/model_key/experiments/imagenet-mini/transfer_robustness_v2/${model[$var1]}_fixed_t_e${epsilon_defense[$var4]}/${model[$var1]}/checkpoints/${model[$var1]} \
                          --version_s               ${model_s[$var2]}                    \
                          --version_s_weight_path   yuyi/model_key/./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                          --exp_name                yuyi/model_key/rebuttal/experiments/imagenet-mini/eval_transfer_robustness_v2_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to_${model[$var1]}_fixed_t_e${epsilon_defense[$var4]} \
                          --train_data_type         ImageNetMini                       \
                          --train_data_path         yuyi/model_key/../imagenet               \
                          --test_data_type          ImageNetMini                       \
                          --test_data_path          yuyi/model_key/../imagenet               \
                          --adv_type                ${adv_type[$var3]}                         \
                          --epsilon                 8.0 \
                          --num_steps               10   \
                          --eval_batch_size         1 \
	                      --num_steps               10 \
	                      --step_size               1.6  
            done
        done
    done
done


# additional AT methods

CUDA_VISIBLE_DEVICES=0 python -u main.py    --config_path configs/cifar10     \
                      --version                 resnet18_RAT                    \
                      --exp_name                experiments/cifar10/RAT \
                      --train_data_type         CIFAR10           \
                      --test_data_type          CIFAR10                     \
                      --train                                                 

CUDA_VISIBLE_DEVICES=1 python -u main.py    --config_path configs/cifar10     \
                      --version                 vgg19_RAT                    \
                      --exp_name                experiments/cifar10/RAT \
                      --train_data_type         CIFAR10           \
                      --test_data_type          CIFAR10                     \
                      --train                                                 

CUDA_VISIBLE_DEVICES=7 python -u main.py    --config_path configs/cifar10     \
                      --version                 resnet50_RAT                    \
                      --exp_name                experiments/cifar10/RAT \
                      --train_data_type         CIFAR10           \
                      --test_data_type          CIFAR10                     \
                      --train_batch_size        64 \
                      --eval_batch_size         128 \
                      --train                                                 

CUDA_VISIBLE_DEVICES=0 python -u main.py    --config_path configs/cifar10     \
                      --version                 mobilenet_v2_cifar_RAT                    \
                      --exp_name                experiments/cifar10/RAT \
                      --train_data_type         CIFAR10           \
                      --test_data_type          CIFAR10                     \
                      --train_batch_size        64 \
                      --eval_batch_size         128 \
                      --train                                                 

CUDA_VISIBLE_DEVICES=0 nohup python -u main.py    --config_path configs/cifar10     \
                      --version                 dense121_RAT                    \
                      --exp_name                experiments/cifar10/RAT \
                      --train_data_type         CIFAR10           \
                      --test_data_type          CIFAR10                     \
                      --train_batch_size        64 \
                      --eval_batch_size         128 \
                      --train                     &                            



CUDA_VISIBLE_DEVICES=0 python -u main.py    --config_path configs/cifar100     \
                      --version                 resnet18_RAT                    \
                      --exp_name                experiments/cifar100/RAT \
                      --train_data_type         CIFAR100           \
                      --test_data_type          CIFAR100                     \
                      --train                                                 

CUDA_VISIBLE_DEVICES=0 python -u main.py    --config_path configs/cifar100     \
                      --version                 vgg19_RAT                    \
                      --exp_name                experiments/cifar100/RAT \
                      --train_data_type         CIFAR100           \
                      --test_data_type          CIFAR100                     \
                      --train                                                 

CUDA_VISIBLE_DEVICES=7 python -u main.py    --config_path configs/cifar100     \
                      --version                 resnet50_RAT                    \
                      --exp_name                experiments/cifar100/RAT \
                      --train_data_type         CIFAR100           \
                      --test_data_type          CIFAR100                     \
                      --train_batch_size        64 \
                      --eval_batch_size         128 \
                      --train                                                 

CUDA_VISIBLE_DEVICES=7 python -u main.py    --config_path configs/cifar100     \
                      --version                 dense121_RAT                    \
                      --exp_name                experiments/cifar100/RAT \
                      --train_data_type         CIFAR100           \
                      --test_data_type          CIFAR100                     \
                      --train_batch_size        64 \
                      --eval_batch_size         128 \
                      --train                                                 


CUDA_VISIBLE_DEVICES=0 python -u main.py    --config_path configs/cifar100     \
                      --version                 mobilenet_v2_cifar_RAT                    \
                      --exp_name                experiments/cifar100/RAT \
                      --train_data_type         CIFAR100           \
                      --test_data_type          CIFAR100                     \
                      --train_batch_size        64 \
                      --eval_batch_size         128 \
                      --train                                                 



# 3 days
CUDA_VISIBLE_DEVICES=0 python -u main.py    --config_path configs/imagenet-mini     \
                      --version                 resnet18_RAT                    \
                      --exp_name                experiments/imagenet-mini/RAT \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --train_batch_size        64 \
                      --eval_batch_size         128 \
                      --load_pretrained_model     \
                      --train                                                 

# 15 days * 0.3
CUDA_VISIBLE_DEVICES=1 python -u main.py    --config_path configs/imagenet-mini     \
                      --version                 vgg19_RAT                    \
                      --exp_name                experiments/imagenet-mini/RAT \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --train_batch_size        16 \
                      --eval_batch_size         32 \
                      --load_pretrained_model     \
                      --train                                                 

# 8 days * 0.48
CUDA_VISIBLE_DEVICES=7 python -u main.py    --config_path configs/imagenet-mini     \
                      --version                 resnet50_RAT                    \
                      --exp_name                experiments/imagenet-mini/RAT \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --train_batch_size        24 \
                      --eval_batch_size         48 \
                      --load_pretrained_model     \
                      --train                                                

# 4 days
CUDA_VISIBLE_DEVICES=7 nohup python -u main.py    --config_path configs/imagenet-mini     \
                      --version                 mobilenet_v2_RAT                    \
                      --exp_name                experiments/imagenet-mini/RAT \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --train_batch_size        16 \
                      --eval_batch_size         24 \
                      --load_pretrained_model     \
                      --train                      &                          

# 16 days * 0.3 (4090)
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py    --config_path configs/imagenet-mini     \
                      --version                 dense121_RAT                    \
                      --exp_name                experiments/imagenet-mini/RAT \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         /home/hdd_disk/Dataset/ImageNet/imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          /home/hdd_disk/Dataset/ImageNet/imagenet               \
                      --train_batch_size        16 \
                      --eval_batch_size         32 \
                      --load_pretrained_model     \
                      --train                     &                         


# 23 days * 0.2
CUDA_VISIBLE_DEVICES=0 python -u main.py    --config_path configs/imagenet-mini     \
                      --version                 inc_v4_RAT                    \
                      --exp_name                experiments/imagenet-mini/RAT \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --train_batch_size        24 \
                      --eval_batch_size         48 \
                      --load_pretrained_model     \
                      --train                                                



# 178 4 5
# 177 5
model=('resnet18_RAT' 'resnet50_RAT' 'vgg19_RAT' 'mobilenet_v2_cifar_RAT' 'dense121_RAT' 'inc_v4_RAT')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'rpa' 'l2t' 'ifgsm' 'pgd')

for var1 in 0
do
    # for var2 in 0 1 2 3 4
    for var2 in 3
    do
        # for var3 in 4 5 6
        # for var3 in 0 1 2 3
        # for var3 in 7 8
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=3 python -u eval_transfer_robustness_diffpure_cifar_RAT.py    --config_path configs/cifar10     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar10/RAT/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar10/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar10/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to${model[$var1]} \
                     --train_data_type          CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


# 165 5 6
# 177 4
model=('resnet18_RAT' 'resnet50_RAT' 'vgg19_RAT' 'mobilenet_v2_cifar_RAT' 'dense121_RAT' 'inc_v4_RAT')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'rpa' 'l2t' 'ifgsm' 'pgd')

for var1 in 0
do
    # for var2 in 0 1 2 3 4
    for var2 in 3
    do
        # for var3 in 4 5 6
        # for var3 in 0 1 2 3
        # for var3 in 7 8
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=2 python -u eval_transfer_robustness_diffpure_cifar_RAT.py    --config_path configs/cifar100     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar100/RAT/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar100/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar100/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to${model[$var1]} \
                     --train_data_type          CIFAR100                       \
                      --test_data_type          CIFAR100                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --eval_batch_size         1 \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


# 176 4 174 21 22 172 9 10 11 12 177 6 175 7 8
model=('resnet18_RAT' 'resnet50_RAT' 'vgg19_RAT' 'mobilenet_v2_RAT' 'dense121_RAT' 'inc_v4_RAT')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2' 'dense121' 'inc_v4')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'l2t' 'diff_pgd' 'ifgsm' 'pgd' 'rpa')

for var1 in 0
do
    for var2 in 5 1 2 3 4
    do
        for var3 in 9
        do
            CUDA_VISIBLE_DEVICES=7 python -u yuyi/model_key/eval_transfer_robustness_diffpure_imagenet_RAT.py    --config_path yuyi/model_key/configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     yuyi/model_key/./experiments/imagenet-mini/RAT/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   yuyi/model_key/./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                yuyi/model_key/rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to${model[$var1]} \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         yuyi/model_key/../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          yuyi/model_key/../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --num_steps               10   \
                      --eval_batch_size         4 \
                      --num_steps               10 \
                      --step_size               1.6  
        done
    done
done


# TDAT
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py    --config_path configs/cifar10     \
                      --version                 resnet18_TDAT                    \
                      --exp_name                experiments/cifar10/TDAT \
                      --train_data_type         CIFAR10           \
                      --test_data_type          CIFAR10                     \
                      --train                            &                     

CUDA_VISIBLE_DEVICES=2 nohup python -u main.py    --config_path configs/cifar10     \
                      --version                 vgg19_TDAT                    \
                      --exp_name                experiments/cifar10/TDAT \
                      --train_data_type         CIFAR10           \
                      --test_data_type          CIFAR10                     \
                      --train                          &                       

CUDA_VISIBLE_DEVICES=3 nohup python -u main.py    --config_path configs/cifar10     \
                      --version                 resnet50_TDAT                    \
                      --exp_name                experiments/cifar10/TDAT \
                      --train_data_type         CIFAR10           \
                      --test_data_type          CIFAR10                     \
                      --train_batch_size        64 \
                      --eval_batch_size         128 \
                      --train                            &                     

CUDA_VISIBLE_DEVICES=0 nohup python -u main.py    --config_path configs/cifar10     \
                      --version                 mobilenet_v2_cifar_TDAT                    \
                      --exp_name                experiments/cifar10/TDAT \
                      --train_data_type         CIFAR10           \
                      --test_data_type          CIFAR10                     \
                      --train_batch_size        64 \
                      --eval_batch_size         128 \
                      --train                            &                     

CUDA_VISIBLE_DEVICES=1 nohup python -u main.py    --config_path configs/cifar10     \
                      --version                 dense121_TDAT                    \
                      --exp_name                experiments/cifar10/TDAT \
                      --train_data_type         CIFAR10           \
                      --test_data_type          CIFAR10                     \
                      --train_batch_size        64 \
                      --eval_batch_size         128 \
                      --train                     &                            


CUDA_VISIBLE_DEVICES=2 nohup python -u main.py    --config_path configs/cifar100     \
                      --version                 resnet18_TDAT                    \
                      --exp_name                experiments/cifar100/TDAT \
                      --train_data_type         CIFAR100           \
                      --test_data_type          CIFAR100                     \
                      --train                             &                    

CUDA_VISIBLE_DEVICES=3 nohup python -u main.py    --config_path configs/cifar100     \
                      --version                 vgg19_TDAT                    \
                      --exp_name                experiments/cifar100/TDAT \
                      --train_data_type         CIFAR100           \
                      --test_data_type          CIFAR100                     \
                      --train                        &                         

CUDA_VISIBLE_DEVICES=4 nohup python -u main.py    --config_path configs/cifar100     \
                      --version                 resnet50_TDAT                    \
                      --exp_name                experiments/cifar100/TDAT \
                      --train_data_type         CIFAR100           \
                      --test_data_type          CIFAR100                     \
                      --train_batch_size        64 \
                      --eval_batch_size         128 \
                      --train                            &                     

CUDA_VISIBLE_DEVICES=5 nohup python -u main.py    --config_path configs/cifar100     \
                      --version                 dense121_TDAT                    \
                      --exp_name                experiments/cifar100/TDAT \
                      --train_data_type         CIFAR100           \
                      --test_data_type          CIFAR100                     \
                      --train_batch_size        64 \
                      --eval_batch_size         128 \
                      --train                           &                      


CUDA_VISIBLE_DEVICES=6 nohup python -u main.py    --config_path configs/cifar100     \
                      --version                 mobilenet_v2_cifar_TDAT                    \
                      --exp_name                experiments/cifar100/TDAT \
                      --train_data_type         CIFAR100           \
                      --test_data_type          CIFAR100                     \
                      --train_batch_size        64 \
                      --eval_batch_size         128 \
                      --train                         &                        



# 3 days
CUDA_VISIBLE_DEVICES=7 nohup python -u main.py    --config_path configs/imagenet-mini     \
                      --version                 resnet18_TDAT                    \
                      --exp_name                experiments/imagenet-mini/TDAT \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --train_batch_size        64 \
                      --eval_batch_size         128 \
                      --load_pretrained_model     \
                      --train                 &                                

# 15 days * 0.3
CUDA_VISIBLE_DEVICES=4 nohup python -u main.py    --config_path configs/imagenet-mini     \
                      --version                 vgg19_TDAT                    \
                      --exp_name                experiments/imagenet-mini/TDAT \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --train_batch_size        16 \
                      --eval_batch_size         32 \
                      --load_pretrained_model     \
                      --train                  &                               

# 8 days * 0.48
CUDA_VISIBLE_DEVICES=7 nohup python -u main.py    --config_path configs/imagenet-mini     \
                      --version                 resnet50_TDAT                    \
                      --exp_name                experiments/imagenet-mini/TDAT \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --train_batch_size        24 \
                      --eval_batch_size         48 \
                      --load_pretrained_model     \
                      --train                         &                       

# 4 days
CUDA_VISIBLE_DEVICES=8 nohup python -u main.py    --config_path configs/imagenet-mini     \
                      --version                 mobilenet_v2_TDAT                    \
                      --exp_name                experiments/imagenet-mini/TDAT \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         ../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          ../imagenet               \
                      --train_batch_size        32 \
                      --eval_batch_size         64 \
                      --load_pretrained_model     \
                      --train                      &                          

# 16 days * 0.3 
CUDA_VISIBLE_DEVICES=1 nohup python -u yuyi/model_key/main.py    --config_path yuyi/model_key/configs/imagenet-mini     \
                      --version                 dense121_TDAT                    \
                      --exp_name                yuyi/model_key/experiments/imagenet-mini/TDAT \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         yuyi/model_key/../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          yuyi/model_key/../imagenet               \
                      --train_batch_size        64 \
                      --eval_batch_size         128 \
                      --load_pretrained_model     \
                      --train                     &                         


# 23 days * 0.2
CUDA_VISIBLE_DEVICES=2 nohup python -u yuyi/model_key/main.py    --config_path yuyi/model_key/configs/imagenet-mini     \
                      --version                 inc_v4_TDAT                    \
                      --exp_name                yuyi/model_key/experiments/imagenet-mini/TDAT \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         yuyi/model_key/../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          yuyi/model_key/../imagenet               \
                      --train_batch_size        64 \
                      --eval_batch_size         128 \
                      --load_pretrained_model     \
                      --train                           &                     


# 177 7 8
model=('resnet18_TDAT' 'resnet50_TDAT' 'vgg19_TDAT' 'mobilenet_v2_cifar_TDAT' 'dense121_TDAT' 'inc_v4_TDAT')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'rpa' 'l2t' 'ifgsm' 'pgd')
bs=(128 128 128 128 16 16 1 128 128)

for var1 in 0
do
    # for var2 in 0 1 2 3 4
    for var2 in 3
    do
        # for var3 in 4 5 6
        # for var3 in 0 1 2 3
        # for var3 in 7 8
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=4 python -u eval_transfer_robustness_diffpure_cifar_TDAT.py    --config_path configs/cifar10     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar10/TDAT/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar10/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar10/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to${model[$var1]} \
                     --train_data_type          CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --eval_batch_size         ${bs[$var3]}  \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


# 177 9 10 
model=('resnet18_TDAT' 'resnet50_TDAT' 'vgg19_TDAT' 'mobilenet_v2_cifar_TDAT' 'dense121_TDAT' 'inc_v4_TDAT')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'rpa' 'l2t' 'ifgsm' 'pgd')
bs=(128 128 128 128 16 16 1 128 128)

for var1 in 0
do
    # for var2 in 0 1 2 3 4
    for var2 in 3
    do
        # for var3 in 4 5 6
        # for var3 in 0 1 2 3
        # for var3 in 7 8
        for var3 in 6
        do
            CUDA_VISIBLE_DEVICES=1 python -u eval_transfer_robustness_diffpure_cifar_TDAT.py    --config_path configs/cifar100     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar100/TDAT/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar100/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                rebuttal/experiments/cifar100/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to${model[$var1]} \
                     --train_data_type          CIFAR100                       \
                      --test_data_type          CIFAR100                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --eval_batch_size         ${bs[$var3]}  \
                      --num_steps               10 \
                      --step_size               0.8  
        done
    done
done


# 178 4 5 176 4 174 22 172 9 10 11 178 4 5 175 8
model=('resnet18_TDAT' 'resnet50_TDAT' 'vgg19_TDAT' 'mobilenet_v2_TDAT' 'dense121_TDAT' 'inc_v4_TDAT')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2' 'dense121' 'inc_v4')
adv_type=('mifgsm' 'difgsm' 'bsr' 'pgn' 'naa' 'l2t' 'diff_pgd' 'ifgsm' 'pgd' 'rpa')

for var1 in 0
do
    for var2 in 0 1 2 3 4 5
    do
        for var3 in 9
        do
            CUDA_VISIBLE_DEVICES=7 python -u yuyi/model_key/eval_transfer_robustness_diffpure_imagenet_TDAT.py    --config_path yuyi/model_key/configs/imagenet-mini     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     yuyi/model_key/./experiments/imagenet-mini/TDAT/${model[$var1]}/checkpoints/${model[$var1]} \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   yuyi/model_key/./experiments/imagenet-mini/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                yuyi/model_key/rebuttal/experiments/imagenet-mini/eval_transfer_robustness_new_add_attack/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to${model[$var1]} \
                      --train_data_type         ImageNetMini                       \
                      --train_data_path         yuyi/model_key/../imagenet               \
                      --test_data_type          ImageNetMini                       \
                      --test_data_path          yuyi/model_key/../imagenet               \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0 \
                      --num_steps               10   \
                      --eval_batch_size         2 \
                      --num_steps               10 \
                      --step_size               1.6  
        done
    done
done


#
model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
# for var1 in 0 1 2 3 4
for var1 in 4
do
    CUDA_VISIBLE_DEVICES=6 nohup python -u main_transfer_robustness_v2_madrys.py    --config_path configs/cifar10     \
                          --version                 ${model[$var1]}_e100                    \
                          --version_s               ${model[$var1]}                    \
                          --exp_name                rebuttal/experiments/cifar10/transfer_robustness_v2_madrys/${model[$var1]}_learn_t_e4 \
                          --train_data_type         CIFAR10_Tensor                       \
                          --test_data_type          CIFAR10_Tensor                     \
                          --train                                                 \
                          --learn_t                                              \
                          --epsilon                 4  &
done

for var1 in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python -u yuyi/model_key/main_transfer_robustness_v2_madrys.py    --config_path yuyi/model_key/configs/cifar10     \
                          --version                 ${model[$var1]}_e100                    \
                          --version_s               ${model[$var1]}                    \
                          --exp_name                yuyi/model_key/rebuttal/experiments/cifar10/transfer_robustness_v2_madrys/${model[$var1]}_fixed_t_e64 \
                          --train_data_type         CIFAR10_Tensor                       \
                          --test_data_type          CIFAR10_Tensor                     \
                          --train                                                 \
                          --train_data_path         yuyi/model_key/../datasets \
                          --test_data_path          yuyi/model_key/../datasets \
                          --epsilon                 64
done


#
model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
for var1 in 0 1 2 3 4
# for var1 in 4
do
    CUDA_VISIBLE_DEVICES=3 python -u yuyi/model_key/main_transfer_robustness_v2_madrys.py    --config_path yuyi/model_key/configs/cifar100     \
                          --version                 ${model[$var1]}                    \
                          --version_s               ${model[$var1]}                    \
                          --exp_name                yuyi/model_key/rebuttal/experiments/cifar100/transfer_robustness_v2_madrys/${model[$var1]}_learn_t_e4 \
                          --train_data_type         CIFAR100                       \
                          --test_data_type          CIFAR100                     \
                          --train                                                 \
                          --learn_t                                              \
                          --train_data_path         yuyi/model_key/../datasets \
                          --test_data_path          yuyi/model_key/../datasets \
                          --epsilon                 4
done

for var1 in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=1 python -u yuyi/model_key/main_transfer_robustness_v2_madrys.py    --config_path yuyi/model_key/configs/cifar100     \
                          --version                 ${model[$var1]}                    \
                          --version_s               ${model[$var1]}                    \
                          --exp_name                yuyi/model_key/rebuttal/experiments/cifar100/transfer_robustness_v2_madrys/${model[$var1]}_fixed_t_e64 \
                          --train_data_type         CIFAR100                       \
                          --test_data_type          CIFAR100                     \
                          --train                                                 \
                          --train_data_path         yuyi/model_key/../datasets \
                          --test_data_path          yuyi/model_key/../datasets \
                          --epsilon                 64
done



#
# 165 7 174 21 22 10 11 178 4 172 11
model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2' 'dense121' 'inc_v4')
for var1 in 5
do
    CUDA_VISIBLE_DEVICES=1 python -u yuyi/model_key/main_transfer_robustness_v2_madrys.py    --config_path yuyi/model_key/configs/imagenet-mini     \
                          --version                 ${model[$var1]}                    \
                          --version_s               ${model[$var1]}                    \
                          --exp_name                yuyi/model_key/rebuttal/experiments/imagenet-mini/transfer_robustness_v2_madrys/${model[$var1]}_learn_t_e4 \
                          --train_data_type         ImageNetMini                       \
	                      --train_data_path         yuyi/model_key/../imagenet               \
	                      --test_data_type          ImageNetMini                       \
	                      --test_data_path          yuyi/model_key/../imagenet               \
	                      --train_batch_size        64 \
	                      --eval_batch_size         64 \
	                      --train                                                 \
                          --learn_t                                              \
                          --load_pretrained_model        \
                          --epsilon                 4
done

# 165 8 172 9 10 9 178 5 172 12
for var1 in 5
do
    CUDA_VISIBLE_DEVICES=3 python -u yuyi/model_key/main_transfer_robustness_v2_madrys.py    --config_path yuyi/model_key/configs/imagenet-mini     \
                          --version                 ${model[$var1]}                    \
                          --version_s               ${model[$var1]}                    \
                          --exp_name                yuyi/model_key/rebuttal/experiments/imagenet-mini/transfer_robustness_v2_madrys/${model[$var1]}_fixed_t_e64 \
                          --train_data_type         ImageNetMini                       \
	                      --train_data_path         yuyi/model_key/../imagenet               \
	                      --test_data_type          ImageNetMini                       \
	                      --test_data_path          yuyi/model_key/../imagenet               \
	                      --train_batch_size        64 \
	                      --eval_batch_size         64 \
	                      --train                                                 \
                          --load_pretrained_model        \
                          --epsilon                 64
done


# reviewer 1.1
# 175 0 1 2 3 4 
model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
epsilon_defense=(1 2 4 8 16 32 64)

for var1 in 0 1 2 3 4
do
# for var2 in 5 10 15 20 25
# for var2 in 30 35 40 45 50
for var2 in 55 60 65 70 75
# for var2 in 80 85 90 95 100
do
for var3 in 6
do
for var4 in 2
do
CUDA_VISIBLE_DEVICES=3 python -u eval_v2_cifar.py    --config_path configs/cifar10     \
          --version                 ${model[$var1]}                    \
          --version_weight_path     ./experiments/cifar10/transfer_robustness_v2/${model[$var1]}_learn_t_e${epsilon_defense[$var4]}/${model[$var1]}_e100/checkpoints/${model[$var1]}_e100 \
          --exp_name                reviewer1.1/cifar10/diffpure_t$var2/${model[$var1]}_learn_t_e${epsilon_defense[$var4]} \
          --train_data_type         CIFAR10_Tensor                       \
          --test_data_type          CIFAR10_Tensor                    \
      	  --eval_batch_size         128  \
      	  --diffpure           \
      	  --t                       $var2 \
      	  --score_type              score_sde
done
done
done
done

for var1 in 0 1 2 3 4
do
# for var2 in 5 10 15 20 25
# for var2 in 30 35 40 45 50
# for var2 in 55 60 65 70 75
for var2 in 80 85 90 95 100
do
for var3 in 6
do
for var4 in 6
do
CUDA_VISIBLE_DEVICES=9 python -u eval_v2_cifar.py    --config_path configs/cifar10     \
          --version                 ${model[$var1]}                    \
          --version_weight_path     ./experiments/cifar10/transfer_robustness_v2/${model[$var1]}_fixed_t_e${epsilon_defense[$var4]}/${model[$var1]}_e100/checkpoints/${model[$var1]}_e100 \
          --exp_name                reviewer1.1/cifar10/diffpure_t$var2/${model[$var1]}_fixed_t_e${epsilon_defense[$var4]} \
          --train_data_type         CIFAR10_Tensor                       \
          --test_data_type          CIFAR10_Tensor                    \
          --eval_batch_size         128 \
      	  --diffpure           \
      	  --t                       $var2 \
      	  --score_type              score_sde 
done
done
done
done


for var1 in 0 1 2 3 4
do
for var2 in 10 20 30 40 50 60 70 80 90 100
do
for var3 in 6
do
for var4 in 2
do
CUDA_VISIBLE_DEVICES=2 python -u eval_v2_cifar.py    --config_path configs/cifar10     \
          --version                 ${model[$var1]}                    \
          --version_weight_path     ./experiments/cifar10/transfer_robustness_v2/${model[$var1]}_learn_t_e${epsilon_defense[$var4]}/${model[$var1]}_e100/checkpoints/${model[$var1]}_e100 \
          --exp_name                reviewer1.1/cifar10/jpeg$var2/${model[$var1]}_learn_t_e${epsilon_defense[$var4]} \
          --train_data_type         CIFAR10_Tensor                       \
          --test_data_type          CIFAR10_Tensor                    \
      	  --eval_batch_size         512  \
      	  --jpeg           \
      	  --purify_hp                       $var2 
done
done
done
done

for var1 in 0 1 2 3 4
do
for var2 in 10 20 30 40 50 60 70 80 90 100
do
for var3 in 6
do
for var4 in 6
do
CUDA_VISIBLE_DEVICES=3 python -u eval_v2_cifar.py    --config_path configs/cifar10     \
          --version                 ${model[$var1]}                    \
          --version_weight_path     ./experiments/cifar10/transfer_robustness_v2/${model[$var1]}_fixed_t_e${epsilon_defense[$var4]}/${model[$var1]}_e100/checkpoints/${model[$var1]}_e100 \
          --exp_name                reviewer1.1/cifar10/jpeg$var2/${model[$var1]}_fixed_t_e${epsilon_defense[$var4]} \
          --train_data_type         CIFAR10_Tensor                       \
          --test_data_type          CIFAR10_Tensor                    \
          --eval_batch_size         512 \
      	  --jpeg           \
      	  --purify_hp                       $var2 
done
done
done
done


# reviewer 1.3
model=('resnet18')
for var1 in 0
do
    CUDA_VISIBLE_DEVICES=2 python -u main_transfer_robustness_v2_finetune.py    --config_path configs/cifar10     \
                          --version                 ${model[$var1]}_e100                    \
                          --version_s               ${model[$var1]}                    \
                          --exp_name                experiments/cifar10/transfer_robustness_v2_finetune/${model[$var1]}_learn_t_e4 \
                          --train_data_type         CIFAR10_Tensor                       \
                          --test_data_type          CIFAR10_Tensor                     \
                          --train                                                 \
                          --load_model             \
                          --learn_t                                              \
                          --epsilon                 4         
done


for var1 in 0
do
    CUDA_VISIBLE_DEVICES=6 python -u main_transfer_robustness_v2_finetune.py    --config_path configs/cifar10     \
                          --version                 ${model[$var1]}_e100                    \
                          --version_s               ${model[$var1]}                    \
                          --exp_name                experiments/cifar10/transfer_robustness_v2_finetune/${model[$var1]}_fixed_t_e64 \
                          --train_data_type         CIFAR10_Tensor                       \
                          --test_data_type          CIFAR10_Tensor                     \
                          --train                                                 \
                          --load_model             \
                          --epsilon                 64       
done




# reviewer 2.3
CUDA_VISIBLE_DEVICES=4 python -u main_ml.py    --config_path configs/cifar10     \
                      --version                 resnet18_e100                    \
                      --exp_name                reviewer2.3/cifar10/LogisticRegressionModel \
                      --train_data_type         CIFAR10_Tensor           \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                                


CUDA_VISIBLE_DEVICES=4 python -u main_transfer_robustness_v2_ml.py    --config_path configs/cifar10     \
                      --version                 resnet18_e100                    \
                      --version_s               resnet18_e100                    \
                      --exp_name                reviewer2.3/cifar10/transfer_robustness_v2/LogisticRegressionModel_fixed_t64 \
                      --train_data_type         CIFAR10_Tensor           \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                  \
                      --epsilon                 64              


CUDA_VISIBLE_DEVICES=4 python -u main_transfer_robustness_v2_ml.py    --config_path configs/cifar10     \
                      --version                 resnet18_e100                    \
                      --version_s               resnet18_e100                    \
                      --exp_name                reviewer2.3/cifar10/transfer_robustness_v2/LogisticRegressionModel_learn_t4 \
                      --train_data_type         CIFAR10_Tensor           \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                  \
                      --learn_t                 \
                      --epsilon                 4              


CUDA_VISIBLE_DEVICES=5 python -u main.py    --config_path configs/cifar10     \
                      --version                 vit_cifar                    \
                      --exp_name                experiments/cifar10/clean \
                      --train_data_type         CIFAR10_Tensor           \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                                


CUDA_VISIBLE_DEVICES=5 python -u main_transfer_robustness_v2.py    --config_path configs/cifar10     \
                      --version                 vit_cifar_e100                    \
                      --exp_name                reviewer2.3/cifar10/transfer_robustness_v2/vit_cifar_fixed_t64 \
                      --train_data_type         CIFAR10_Tensor           \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                          \
                      --epsilon                 64      


CUDA_VISIBLE_DEVICES=6 python -u main_transfer_robustness_v2.py    --config_path configs/cifar10     \
                      --version                 vit_cifar_e100                    \
                      --exp_name                reviewer2.3/cifar10/transfer_robustness_v2/vit_cifar_learn_t4 \
                      --train_data_type         CIFAR10_Tensor           \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                          \
                      --learn_t                  \
                      --epsilon                 4      




# reviewer 2.5
model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
for var1 in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=2 python -u main_transfer_robustness_v2_DA.py    --config_path configs/cifar10     \
                          --version                 ${model[$var1]}_e100                    \
                          --version_s               ${model[$var1]}                    \
                          --exp_name                experiments/cifar10/transfer_robustness_v2_DA/${model[$var1]}_fixed_t_e64 \
                          --train_data_type         CIFAR10_Tensor                       \
                          --test_data_type          CIFAR10_Tensor                     \
                          --train                                                 \
                          --epsilon                 64       
done

model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
CUDA_VISIBLE_DEVICES=0 python -u main_transfer_robustness_v2_DA_random.py    --config_path configs/cifar10     \
                          --version                 resnet18_e100                    \
                          --version_s               resnet18                    \
                          --exp_name                experiments/cifar10/transfer_robustness_v2_DA_random/resnet18_fixed_t_e64 \
                          --train_data_type         CIFAR10_Tensor                       \
                          --test_data_type          CIFAR10_Tensor                     \
                          --train                                                 \
                          --epsilon                 64 

CUDA_VISIBLE_DEVICES=0 python -u main_transfer_robustness_v2_DA_random.py    --config_path configs/cifar10     \
                          --version                 resnet50_e100                    \
                          --version_s               resnet50                    \
                          --exp_name                experiments/cifar10/transfer_robustness_v2_DA_random/resnet50_fixed_t_e64 \
                          --train_data_type         CIFAR10_Tensor                       \
                          --test_data_type          CIFAR10_Tensor                     \
                          --train                                                 \
                          --epsilon                 64 

CUDA_VISIBLE_DEVICES=0 python -u main_transfer_robustness_v2_DA_random.py    --config_path configs/cifar10     \
                          --version                 vgg19_e100                    \
                          --version_s               vgg19                    \
                          --exp_name                experiments/cifar10/transfer_robustness_v2_DA_random/vgg19_fixed_t_e64 \
                          --train_data_type         CIFAR10_Tensor                       \
                          --test_data_type          CIFAR10_Tensor                     \
                          --train                                                 \
                          --epsilon                 64 

CUDA_VISIBLE_DEVICES=1 python -u main_transfer_robustness_v2_DA_random.py    --config_path configs/cifar10     \
                          --version                 mobilenet_v2_cifar_e100                    \
                          --version_s               mobilenet_v2_cifar                    \
                          --exp_name                experiments/cifar10/transfer_robustness_v2_DA_random/mobilenet_v2_cifar_fixed_t_e64 \
                          --train_data_type         CIFAR10_Tensor                       \
                          --test_data_type          CIFAR10_Tensor                     \
                          --train                                                 \
                          --epsilon                 64 

CUDA_VISIBLE_DEVICES=1 python -u main_transfer_robustness_v2_DA_random.py    --config_path configs/cifar10     \
                          --version                 dense121_e100                    \
                          --version_s               dense121                    \
                          --exp_name                experiments/cifar10/transfer_robustness_v2_DA_random/dense121_fixed_t_e64 \
                          --train_data_type         CIFAR10_Tensor                       \
                          --test_data_type          CIFAR10_Tensor                     \
                          --train                                                 \
                          --epsilon                 64 


CUDA_VISIBLE_DEVICES=0 python -u main_transfer_robustness_v2_DA.py    --config_path configs/cifar10     \
                          --version                 resnet18_e100                    \
                          --version_s               resnet18                    \
                          --exp_name                experiments/cifar10/transfer_robustness_v2_DA/resnet18_learn_t_e4 \
                          --train_data_type         CIFAR10_Tensor                       \
                          --test_data_type          CIFAR10_Tensor                     \
                          --train                                                 \
                          --epsilon                 4 \
                          --learn_t

CUDA_VISIBLE_DEVICES=0 python -u main_transfer_robustness_v2_DA.py    --config_path configs/cifar10     \
                          --version                 resnet50_e100                    \
                          --version_s               resnet50                    \
                          --exp_name                experiments/cifar10/transfer_robustness_v2_DA/resnet50_learn_t_e4 \
                          --train_data_type         CIFAR10_Tensor                       \
                          --test_data_type          CIFAR10_Tensor                     \
                          --train                                                 \
                          --epsilon                 4 \
                          --learn_t

CUDA_VISIBLE_DEVICES=0 python -u main_transfer_robustness_v2_DA.py    --config_path configs/cifar10     \
                          --version                 vgg19_e100                    \
                          --version_s               vgg19                    \
                          --exp_name                experiments/cifar10/transfer_robustness_v2_DA/vgg19_learn_t_e4 \
                          --train_data_type         CIFAR10_Tensor                       \
                          --test_data_type          CIFAR10_Tensor                     \
                          --train                                                 \
                          --epsilon                 4 \
                          --learn_t

CUDA_VISIBLE_DEVICES=1 python -u main_transfer_robustness_v2_DA.py    --config_path configs/cifar10     \
                          --version                 mobilenet_v2_cifar_e100                    \
                          --version_s               mobilenet_v2_cifar                    \
                          --exp_name                experiments/cifar10/transfer_robustness_v2_DA/mobilenet_v2_cifar_learn_t_e4 \
                          --train_data_type         CIFAR10_Tensor                       \
                          --test_data_type          CIFAR10_Tensor                     \
                          --train                                                 \
                          --epsilon                 4 \
                          --learn_t

CUDA_VISIBLE_DEVICES=1 python -u main_transfer_robustness_v2_DA.py    --config_path configs/cifar10     \
                          --version                 dense121_e100                    \
                          --version_s               dense121                    \
                          --exp_name                experiments/cifar10/transfer_robustness_v2_DA/dense121_learn_t_e4 \
                          --train_data_type         CIFAR10_Tensor                       \
                          --test_data_type          CIFAR10_Tensor                     \
                          --train                                                 \
                          --epsilon                 4 \
                          --learn_t

# reviewer 3.2

model=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
model_s=('resnet18' 'resnet50' 'vgg19' 'mobilenet_v2_cifar' 'dense121')
adv_type=('mifgsm' 'difgsm' 'ifgsm' 'pgd' 'bsr' 'pgn' 'naa' 'rpa' 'l2t')

# 175 1 174 21
for var1 in 0
do
    for var2 in 0 1 2 3 4
    do
        for var3 in 0 1 2 3
        do
            CUDA_VISIBLE_DEVICES=6 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar10     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar10/transfer_robustness_v2/${model[$var1]}_learn_t_e4/${model[$var1]}_e100/checkpoints/${model[$var1]}_e100 \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar10/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                reviewer3.2/cifar10/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to${model[$var1]}_learn_t_e4 \
                      --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0   \
                      --eval_batch_size         128 \
                      --num_steps               10 \
                      --step_size               1.6  
        done
    done
done


for var1 in 0
do
    for var2 in 0 1 2 3 4
    do
        for var3 in 0 1 2 3
        do
            CUDA_VISIBLE_DEVICES=4 python -u eval_transfer_robustness_diffpure_cifar.py    --config_path configs/cifar10     \
                      --version                 ${model[$var1]}                    \
                      --version_weight_path     ./experiments/cifar10/transfer_robustness_v2/${model[$var1]}_fixed_t_e64/${model[$var1]}_e100/checkpoints/${model[$var1]}_e100 \
                      --version_s               ${model_s[$var2]}                    \
                      --version_s_weight_path   ./experiments/cifar10/clean/${model_s[$var2]}/checkpoints/${model_s[$var2]} \
                      --exp_name                reviewer3.2/cifar10/${adv_type[$var3]}_e8/${model_s[$var2]}_v_to${model[$var1]}_fixed_t_e64 \
                      --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                    \
                      --adv_type                ${adv_type[$var3]}                         \
                      --epsilon                 8.0   \
                      --eval_batch_size         128 \
                      --num_steps               10 \
                      --step_size               1.6  
        done
    done
done



CUDA_VISIBLE_DEVICES=0 python -u main.py    --config_path configs/cifar10     \
                      --version                 resnet18_madrys                    \
                      --exp_name                experiments/cifar10/AT \
                      --train_data_type         CIFAR10_Tensor           \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                                 


CUDA_VISIBLE_DEVICES=0 python -u main.py    --config_path configs/cifar10     \
                      --version                 resnet50_madrys                    \
                      --exp_name                experiments/cifar10/AT \
                      --train_data_type         CIFAR10_Tensor           \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                                 

CUDA_VISIBLE_DEVICES=0 python -u main.py    --config_path configs/cifar10     \
                      --version                 vgg19_madrys                    \
                      --exp_name                experiments/cifar10/AT \
                      --train_data_type         CIFAR10_Tensor           \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                                 

CUDA_VISIBLE_DEVICES=1 python -u main.py    --config_path configs/cifar10     \
                      --version                 mobilenet_v2_cifar_madrys                    \
                      --exp_name                experiments/cifar10/AT \
                      --train_data_type         CIFAR10_Tensor           \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                                 

CUDA_VISIBLE_DEVICES=1 python -u main.py    --config_path configs/cifar10     \
                      --version                 dense121_madrys                    \
                      --exp_name                experiments/cifar10/AT \
                      --train_data_type         CIFAR10_Tensor           \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                                 



##
CUDA_VISIBLE_DEVICES=0 python -u main_transfer_robustness_v10.py    --config_path configs/cifar10     \
                      --version                 resnet18_e100                    \
                      --version_s               resnet18                    \
                      --exp_name                experiments/cifar10/transfer_robustness_v10/resnet18_learn_t_e4 \
                      --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                                 \
                      --learn_t                                              \
                      --epsilon                 4

CUDA_VISIBLE_DEVICES=0 python -u main_transfer_robustness_v10.py    --config_path configs/cifar10     \
                      --version                 resnet18_e100                    \
                      --version_s               resnet18                    \
                      --exp_name                experiments/cifar10/transfer_robustness_v10/resnet18_fixed_t_e64 \
                      --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                                 \
                      --epsilon                 64



CUDA_VISIBLE_DEVICES=0 python -u main_transfer_robustness_v10.py    --config_path configs/cifar10     \
                      --version                 resnet50_e100                    \
                      --version_s               resnet50                    \
                      --exp_name                experiments/cifar10/transfer_robustness_v10/resnet50_learn_t_e4 \
                      --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                                 \
                      --learn_t                                              \
                      --epsilon                 4

CUDA_VISIBLE_DEVICES=0 python -u main_transfer_robustness_v10.py    --config_path configs/cifar10     \
                      --version                 resnet50_e100                    \
                      --version_s               resnet50                    \
                      --exp_name                experiments/cifar10/transfer_robustness_v10/resnet50_fixed_t_e64 \
                      --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                                 \
                      --epsilon                 64



CUDA_VISIBLE_DEVICES=0 python -u main_transfer_robustness_v10.py    --config_path configs/cifar10     \
                      --version                 vgg19_e100                    \
                      --version_s               vgg19                    \
                      --exp_name                experiments/cifar10/transfer_robustness_v10/vgg19_learn_t_e4 \
                      --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                                 \
                      --learn_t                                              \
                      --epsilon                 4

CUDA_VISIBLE_DEVICES=0 python -u main_transfer_robustness_v10.py    --config_path configs/cifar10     \
                      --version                 vgg19_e100                    \
                      --version_s               vgg19                    \
                      --exp_name                experiments/cifar10/transfer_robustness_v10/vgg19_fixed_t_e64 \
                      --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                                 \
                      --epsilon                 64



CUDA_VISIBLE_DEVICES=1 python -u main_transfer_robustness_v10.py    --config_path configs/cifar10     \
                      --version                 mobilenet_v2_cifar_e100                    \
                      --version_s               mobilenet_v2_cifar                    \
                      --exp_name                experiments/cifar10/transfer_robustness_v10/mobilenet_v2_cifar_learn_t_e4 \
                      --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                                 \
                      --learn_t                                              \
                      --epsilon                 4

CUDA_VISIBLE_DEVICES=1 python -u main_transfer_robustness_v10.py    --config_path configs/cifar10     \
                      --version                 mobilenet_v2_cifar_e100                    \
                      --version_s               mobilenet_v2_cifar                    \
                      --exp_name                experiments/cifar10/transfer_robustness_v10/mobilenet_v2_cifar_fixed_t_e64 \
                      --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                                 \
                      --epsilon                 64



CUDA_VISIBLE_DEVICES=1 python -u main_transfer_robustness_v10.py    --config_path configs/cifar10     \
                      --version                 dense121_e100                    \
                      --version_s               dense121                    \
                      --exp_name                experiments/cifar10/transfer_robustness_v10/dense121_learn_t_e4 \
                      --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                                 \
                      --learn_t                                              \
                      --epsilon                 4

CUDA_VISIBLE_DEVICES=1 python -u main_transfer_robustness_v10.py    --config_path configs/cifar10     \
                      --version                 dense121_e100                    \
                      --version_s               dense121                    \
                      --exp_name                experiments/cifar10/transfer_robustness_v10/dense121_fixed_t_e64 \
                      --train_data_type         CIFAR10_Tensor                       \
                      --test_data_type          CIFAR10_Tensor                     \
                      --train                                                 \
                      --epsilon                 64



