#!/bin/bash
#
# if [ -f sample_results_output.log ]
# then
#   rm sample_results_output.log
# fi
# touch sample_results_output.log
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
do
    if [ -d ../sample_results/sample_softplus_"$i" ]
    then
      rm -r ../sample_results/sample_softplus_"$i"
    fi
    mkdir ../sample_results/sample_softplus_"$i"
    if [ -f ../sample_results/sample_softplus_"$i"/output.log ]
    then
      rm ../sample_results/sample_softplus_"$i"/output.log
    fi
    touch ../sample_results/sample_softplus_"$i"/output.log
    python run_attack_softplus.py --cuda --img ../../Spatial_transform/ST_ADV_exp_imagenet/sample_imagenet/sample_"$i".jpg \
    --target_img ../../Spatial_transform/ST_ADV_exp_imagenet/sample_imagenet/sample_"$i"_target.jpg \
    --output_dir ../sample_results/sample_softplus_"$i"/ --lr 0.0003| tee -a "sample_results_output.log"
    echo "\n" | tee -a ../sample_results/sample_softplus_"$i"/output.log
done
