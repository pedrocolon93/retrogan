cd ../Code

SEED=$1
echo "Starting run with $SEED"
CUDA_VISIBLE_DEVICES=0 python retrogan_trainer.py --device cuda --iters 312500 ../Data/ft_all_unseen.txt ../Data/ft_all_unseen_retrofitted.txt ft_new_full_all_pytorch models/trained_retrogan/ft_new_full_all_pytorch_$SEED &
CUDA_VISIBLE_DEVICES=1 python retrogan_trainer.py --device cuda --iters 312500 ../Data/ft_all_unseen.txt ../Data/nb_retrofitted_attractrepelretrofitted.txt nb_new_full_all_pytorch models/trained_retrogan/nb_new_full_all_pytorch_$SEED &
CUDA_VISIBLE_DEVICES=2 python retrogan_trainer.py --device cuda --iters 312500 ../Data/ft_all_unseen.txt ../Data/ft_all_retrofitted_ook_unseen.txt ft_ook_new_full_all_pytorch models/trained_retrogan/ft_ook_new_full_all_pytorch_$SEED &
CUDA_VISIBLE_DEVICES=3 python retrogan_trainer.py --device cuda --iters 312500 ../Data/ft_all_unseen.txt ../Data/nb_retrofitted_ook.txt nb_ook_new_full_all_pytorch models/trained_retrogan/nb_ook_new_full_all_pytorch_$SEED &
