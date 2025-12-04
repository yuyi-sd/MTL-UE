# CelebA
attr_num=40
## train on Clean Data
CUDA_VISIBLE_DEVICES=1 python -u main_MTL_binary.py    --config_path configs/CelebA     \
                  --version                 resnet18                    \
                  --exp_name                experiments_MTL/CelebA/clean_MTL_f${attr_num} \
                  --train_data_type         CelebA                       \
                  --test_data_type          CelebA                       \
                  --train_batch_size        512                            \
                  --eval_batch_size         512                            \
                  --train_data_path         ./datasets           \
                  --test_data_path          ./datasets          \
                  --load_model                                          \
                  --n_tasks                 ${attr_num}               \
                  --save_frequency          4   


## Generate perturbations (MTL-UE + SEP/TAP/EM)
CUDA_VISIBLE_DEVICES=0 python perturbation_MTL_binary_DAE.py --config_path             configs/CelebA                \
                        --exp_name                experiments_MTL/CelebA/DAE_ER_CER/sep/MTL_f${attr_num} \
                        --version                 resnet18                       \
                        --train_data_type         CelebA                       \
                        --test_data_type          CelebA                       \
                        --train_batch_size        1024                            \
                        --eval_batch_size         1024                            \
                        --train_data_path         ./datasets          \
                        --test_data_path          ./datasets          \
                        --epsilon                 8                              \
                        --num_steps               20                             \
                        --step_size               0.8                            \
                        --attack_type             sep                       \
                        --perturb_type            samplewise                      \
                        --universal_stop_error    0.99                                \
                        --train_step              10  \
                        --load_model                \
                        --load_model_path         experiments_MTL/CelebA/clean_MTL_f${attr_num}/resnet18/checkpoints/resnet18 \
                        --n_tasks                 ${attr_num}    \
                        --embedding_regularization  \
                        --cross_embedding_regularization \
                        --seed                    0  \
                        --stop_epoch              200

CUDA_VISIBLE_DEVICES=0 python perturbation_MTL_binary_DAE.py --config_path             configs/CelebA                \
                        --exp_name                experiments_MTL/CelebA/DAE_ER_CER/tap/MTL_f${attr_num} \
                        --version                 resnet18                       \
                        --train_data_type         CelebA                       \
                        --test_data_type          CelebA                       \
                        --train_batch_size        512                            \
                        --eval_batch_size         512                            \
                        --train_data_path         ./datasets          \
                        --test_data_path          ./datasets          \
                        --epsilon                 8                              \
                        --num_steps               20                             \
                        --step_size               0.8                            \
                        --attack_type             min-max                        \
                        --perturb_type            samplewise                      \
                        --universal_stop_error    0.9                                \
                        --train_step              10  \
                        --load_model                \
                        --load_model_path         experiments_MTL/CelebA/clean_MTL_f${attr_num}/resnet18/checkpoints/resnet18 \
                        --embedding_regularization  \
                        --cross_embedding_regularization \
                        --stop_epoch              100 \
                        --n_tasks                 ${attr_num}  

CUDA_VISIBLE_DEVICES=0 python perturbation_MTL_binary_DAE.py --config_path             configs/CelebA                \
                          --exp_name                experiments_MTL/CelebA/DAE_ER_CER/em/MTL_f${attr_num} \
                          --version                 resnet18                       \
                          --train_data_type         CelebA                       \
                          --test_data_type          CelebA                       \
                          --train_batch_size        512                            \
                          --eval_batch_size         512                            \
                          --train_data_path         ./datasets          \
                          --test_data_path          ./datasets          \
                          --epsilon                 8                              \
                          --num_steps               20                             \
                          --step_size               0.8                            \
                          --attack_type             min-min                        \
                          --perturb_type            samplewise                      \
                          --universal_stop_error    0.01                               \
                          --train_step              2  \
                          --n_tasks                 ${attr_num}   \
                          --embedding_regularization  \
                          --cross_embedding_regularization \
                          --stop_epoch              100  

## train on MTL-UE
CUDA_VISIBLE_DEVICES=0 python -u main_MTL_binary.py    --config_path configs/CelebA     \
                  --version                 resnet18                    \
                  --exp_name                experiments_MTL/CelebA/DAE_ER_CER/tap/DAE_ER_CER_MTL_f${attr_num}_MTL_f${attr_num} \
                  --train_data_type         PoisonCelebA                       \
                  --test_data_type          CelebA                       \
                  --train_batch_size        512                            \
                  --eval_batch_size         512                            \
                  --train_data_path         ./datasets           \
                  --test_data_path          ./datasets          \
                  --load_model                                          \
                  --perturb_tensor_filepath ./experiments_MTL/CelebA/DAE_ER_CER/tap/MTL_f${attr_num}/perturbation.pt    \
                  --poison_rate             1.0                            \
                  --perturb_type            samplewise           \
                  --n_tasks                 ${attr_num}               

CUDA_VISIBLE_DEVICES=0 python -u main_STL_binary.py    --config_path configs/CelebA     \
                        --version                 resnet18_stl                    \
                        --exp_name                experiments_MTL/CelebA/DAE_ER_CER/tap/DAE_ER_CER_MTL_f${attr_num}_STL \
                        --train_data_type         PoisonCelebA                       \
                        --test_data_type          CelebA                       \
                        --train_batch_size        512                            \
                        --eval_batch_size         512                            \
                        --train_data_path         ./datasets           \
                        --test_data_path          ./datasets          \
                        --train                                          \
                        --perturb_tensor_filepath ./experiments_MTL/CelebA/DAE_ER_CER/tap/MTL_f${attr_num}/perturbation.pt    \
                        --poison_rate             1.0                            \
                        --perturb_type            samplewise     \
                        --n_tasks                 ${attr_num}   


CUDA_VISIBLE_DEVICES=0 python -u main_MTL_binary.py    --config_path configs/CelebA     \
                      --version                 resnet18                    \
                      --exp_name                experiments_MTL/CelebA/DAE_ER_CER/sep/DAE_ER_CER_MTL_f${attr_num}_MTL_f${attr_num} \
                      --train_data_type         PoisonCelebA                       \
                      --test_data_type          CelebA                       \
                      --train_batch_size        512                            \
                      --eval_batch_size         512                            \
                      --train_data_path         ./datasets           \
                      --test_data_path          ./datasets          \
                      --load_model                                          \
                      --perturb_tensor_filepath ./experiments_MTL/CelebA/DAE_ER_CER/sep/MTL_f${attr_num}/perturbation.pt    \
                      --poison_rate             1.0                            \
                      --perturb_type            samplewise           \
                      --n_tasks                 ${attr_num}   

CUDA_VISIBLE_DEVICES=0 python -u main_STL_binary.py    --config_path configs/CelebA     \
                        --version                 resnet18_stl                    \
                        --exp_name                experiments_MTL/CelebA/DAE_ER_CER/sep/DAE_ER_CER_MTL_f${attr_num}_STL \
                        --train_data_type         PoisonCelebA                       \
                        --test_data_type          CelebA                       \
                        --train_batch_size        512                            \
                        --eval_batch_size         512                            \
                        --train_data_path         ./datasets           \
                        --test_data_path          ./datasets          \
                        --train                                          \
                        --perturb_tensor_filepath ./experiments_MTL/CelebA/DAE_ER_CER/sep/MTL_f${attr_num}/perturbation.pt    \
                        --poison_rate             1.0                            \
                        --perturb_type            samplewise     \
                        --n_tasks                 ${attr_num}   

CUDA_VISIBLE_DEVICES=0 python -u main_MTL_binary.py    --config_path configs/CelebA     \
                      --version                 resnet18                    \
                      --exp_name                experiments_MTL/CelebA/DAE_ER_CER/em/DAE_ER_CER_MTL_f${attr_num}_MTL_f${attr_num} \
                      --train_data_type         PoisonCelebA                       \
                      --test_data_type          CelebA                       \
                      --train_batch_size        512                            \
                      --eval_batch_size         512                            \
                      --train_data_path         ./datasets           \
                      --test_data_path          ./datasets          \
                      --load_model                                          \
                      --perturb_tensor_filepath ./experiments_MTL/CelebA/DAE_ER_CER/em/MTL_f${attr_num}/perturbation.pt    \
                      --poison_rate             1.0                            \
                      --perturb_type            samplewise           \
                      --n_tasks                 ${attr_num}  


CUDA_VISIBLE_DEVICES=0 python -u main_STL_binary.py    --config_path configs/CelebA     \
                        --version                 resnet18_stl                    \
                        --exp_name                experiments_MTL/CelebA/DAE_ER_CER/em/DAE_ER_CER_MTL_f${attr_num}_STL \
                        --train_data_type         PoisonCelebA                       \
                        --test_data_type          CelebA                       \
                        --train_batch_size        512                            \
                        --eval_batch_size         512                            \
                        --train_data_path         ./datasets           \
                        --test_data_path          ./datasets          \
                        --train                                          \
                        --perturb_tensor_filepath ./experiments_MTL/CelebA/DAE_ER_CER/em/MTL_f${attr_num}/perturbation.pt    \
                        --poison_rate             1.0                            \
                        --perturb_type            samplewise     \
                        --n_tasks                 ${attr_num}   



# UTKFace
attr_num=3
## train on Clean Data
CUDA_VISIBLE_DEVICES=0 python -u main_MTL.py    --config_path configs/UTKFace     \
                      --version                 resnet18                    \
                      --exp_name                experiments_MTL/UTKFace/clean_MTL_f3 \
                      --train_data_type         UTKFace                       \
                      --test_data_type          UTKFace                       \
                      --train_batch_size        128                            \
                      --eval_batch_size         128                            \
                      --train_data_path         ./datasets/UTKFace           \
                      --test_data_path          ./datasets/UTKFace          \
                      --train                                     \
                      --save_frequency          4        \
                      --n_tasks                 3   

## Generate perturbations (MTL-UE + SEP/TAP/EM)
CUDA_VISIBLE_DEVICES=0 python perturbation_MTL_DAE.py --config_path             configs/UTKFace                \
                          --exp_name                experiments_MTL/UTKFace/DAE_ER_CER/sep/MTL_f${attr_num} \
                          --version                 resnet18                       \
                          --train_data_type         UTKFace                       \
                          --test_data_type          UTKFace                       \
                          --train_batch_size        128                            \
                          --eval_batch_size         128                            \
                          --train_data_path         ./datasets/UTKFace          \
                          --test_data_path          ./datasets/UTKFace          \
                          --epsilon                 8                              \
                          --num_steps               20                             \
                          --step_size               0.8                            \
                          --attack_type             sep                        \
                          --perturb_type            samplewise                      \
                          --universal_stop_error    0.0001                                \
                          --train_step              10  \
                          --load_model                \
                          --load_model_path         experiments_MTL/UTKFace/clean_MTL_f${attr_num}/resnet18/checkpoints/resnet18 \
                          --n_tasks                 ${attr_num}  \
                          --embedding_regularization  \
                          --cross_embedding_regularization \
                          --stop_epoch              100

CUDA_VISIBLE_DEVICES=0 python perturbation_MTL_DAE.py --config_path             configs/UTKFace                \
                          --exp_name                experiments_MTL/UTKFace/DAE_ER_CER/tap/MTL_f${attr_num} \
                          --version                 resnet18                       \
                          --train_data_type         UTKFace                       \
                          --test_data_type          UTKFace                       \
                          --train_batch_size        128                            \
                          --eval_batch_size         128                            \
                          --train_data_path         ./datasets/UTKFace          \
                          --test_data_path          ./datasets/UTKFace          \
                          --epsilon                 8                              \
                          --num_steps               20                             \
                          --step_size               0.8                            \
                          --attack_type             min-max                        \
                          --perturb_type            samplewise                      \
                          --universal_stop_error    0.0001                                \
                          --train_step              10  \
                          --load_model                \
                          --load_model_path         experiments_MTL/UTKFace/clean_MTL_f${attr_num}/resnet18/checkpoints/resnet18 \
                          --n_tasks                 ${attr_num}  \
                          --embedding_regularization  \
                          --cross_embedding_regularization \
                          --stop_epoch              100

CUDA_VISIBLE_DEVICES=0 python perturbation_MTL_DAE.py --config_path             configs/UTKFace                \
                          --exp_name                experiments_MTL/UTKFace/DAE_ER_CER/em/MTL_f${attr_num} \
                          --version                 resnet18                       \
                          --train_data_type         UTKFace                       \
                          --test_data_type          UTKFace                       \
                          --train_batch_size        128                            \
                          --eval_batch_size         128                            \
                          --train_data_path         ./datasets/UTKFace          \
                          --test_data_path          ./datasets/UTKFace          \
                          --epsilon                 8                              \
                          --num_steps               20                             \
                          --step_size               0.8                            \
                          --attack_type             min-min                        \
                          --perturb_type            samplewise                      \
                          --universal_stop_error    0.01                                \
                          --train_step              2  \
                          --n_tasks                 ${attr_num}  \
                          --embedding_regularization  \
                          --cross_embedding_regularization \
                          --stop_epoch              100

## train on MTL-UE
CUDA_VISIBLE_DEVICES=0 python -u main_MTL.py    --config_path configs/UTKFace     \
                    --version                 resnet18                    \
                    --exp_name                experiments_MTL/UTKFace/DAE_ER_CER/em/DAE_ER_CER_MTL_f${attr_num}_MTL_f${attr_num} \
                    --train_data_type         PoisonUTKFace                       \
                    --test_data_type          UTKFace                       \
                    --train_batch_size        128                            \
                    --eval_batch_size         128                            \
                    --train_data_path         ./datasets/UTKFace           \
                    --test_data_path          ./datasets/UTKFace          \
                    --train                                          \
                    --perturb_tensor_filepath ./experiments_MTL/UTKFace/DAE_ER_CER/em/MTL_f${attr_num}/perturbation.pt    \
                    --poison_rate             1.0                            \
                    --perturb_type            samplewise       \
                    --n_tasks                 ${attr_num} 
CUDA_VISIBLE_DEVICES=0 python -u main_STL.py    --config_path configs/UTKFace     \
                    --version                 resnet18_stl                    \
                    --exp_name                experiments_MTL/UTKFace/DAE_ER_CER/em/DAE_ER_CER_MTL_f${attr_num}_STL \
                    --train_data_type         PoisonUTKFace                       \
                    --test_data_type          UTKFace                       \
                    --train_batch_size        128                            \
                    --eval_batch_size         128                            \
                    --train_data_path         ./datasets/UTKFace           \
                    --test_data_path          ./datasets/UTKFace          \
                    --train                                          \
                    --perturb_tensor_filepath ./experiments_MTL/UTKFace/DAE_ER_CER/em/MTL_f${attr_num}/perturbation.pt     \
                    --poison_rate             1.0                            \
                    --perturb_type            samplewise      \
                    --n_tasks                 ${attr_num}   


CUDA_VISIBLE_DEVICES=0 python -u main_MTL.py    --config_path configs/UTKFace     \
                    --version                 resnet18                    \
                    --exp_name                experiments_MTL/UTKFace/DAE_ER_CER/tap/DAE_ER_CER_MTL_f${attr_num}_MTL_f${attr_num} \
                    --train_data_type         PoisonUTKFace                       \
                    --test_data_type          UTKFace                       \
                    --train_batch_size        128                            \
                    --eval_batch_size         128                            \
                    --train_data_path         ./datasets/UTKFace           \
                    --test_data_path          ./datasets/UTKFace          \
                    --train                                          \
                    --perturb_tensor_filepath ./experiments_MTL/UTKFace/DAE_ER_CER/tap/MTL_f${attr_num}/perturbation.pt    \
                    --poison_rate             1.0                            \
                    --perturb_type            samplewise       \
                    --n_tasks                 ${attr_num} 
CUDA_VISIBLE_DEVICES=0 python -u main_STL.py    --config_path configs/UTKFace     \
                    --version                 resnet18_stl                    \
                    --exp_name                experiments_MTL/UTKFace/DAE_ER_CER/tap/DAE_ER_CER_MTL_f${attr_num}_STL \
                    --train_data_type         PoisonUTKFace                       \
                    --test_data_type          UTKFace                       \
                    --train_batch_size        128                            \
                    --eval_batch_size         128                            \
                    --train_data_path         ./datasets/UTKFace           \
                    --test_data_path          ./datasets/UTKFace          \
                    --train                                          \
                    --perturb_tensor_filepath ./experiments_MTL/UTKFace/DAE_ER_CER/tap/MTL_f${attr_num}/perturbation.pt     \
                    --poison_rate             1.0                            \
                    --perturb_type            samplewise      \
                    --n_tasks                 ${attr_num}   



CUDA_VISIBLE_DEVICES=0 python -u main_MTL.py    --config_path configs/UTKFace     \
                    --version                 resnet18                    \
                    --exp_name                experiments_MTL/UTKFace/DAE_ER_CER/sep/DAE_ER_CER_MTL_f${attr_num}_MTL_f${attr_num} \
                    --train_data_type         PoisonUTKFace                       \
                    --test_data_type          UTKFace                       \
                    --train_batch_size        128                            \
                    --eval_batch_size         128                            \
                    --train_data_path         ./datasets/UTKFace           \
                    --test_data_path          ./datasets/UTKFace          \
                    --train                                          \
                    --perturb_tensor_filepath ./experiments_MTL/UTKFace/DAE_ER_CER/sep/MTL_f${attr_num}/perturbation.pt    \
                    --poison_rate             1.0                            \
                    --perturb_type            samplewise       \
                    --n_tasks                 ${attr_num} 
CUDA_VISIBLE_DEVICES=0 python -u main_STL.py    --config_path configs/UTKFace     \
                    --version                 resnet18_stl                    \
                    --exp_name                experiments_MTL/UTKFace/DAE_ER_CER/sep/DAE_ER_CER_MTL_f${attr_num}_STL \
                    --train_data_type         PoisonUTKFace                       \
                    --test_data_type          UTKFace                       \
                    --train_batch_size        128                            \
                    --eval_batch_size         128                            \
                    --train_data_path         ./datasets/UTKFace           \
                    --test_data_path          ./datasets/UTKFace          \
                    --train                                          \
                    --perturb_tensor_filepath ./experiments_MTL/UTKFace/DAE_ER_CER/sep/MTL_f${attr_num}/perturbation.pt     \
                    --poison_rate             1.0                            \
                    --perturb_type            samplewise      \
                    --n_tasks                 ${attr_num}   

