mkdir -p output
txt_file="output/result.txt"
seeds="100000-149999"
model="s-fid"

## Table 2 ##
guidance=1.0

steps=32
for w_stiff in 0.5 1.0 1.5 2.0 2.5; do \
    for w_con in 0.5; do \
        echo "###### model: ${model}, guidance: ${guidance}, steps: ${steps}, w_stiff: ${w_stiff}, w_con: ${w_con} ######" >> ${txt_file}
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_images.py \
                --preset=edm2-img512-${model} --outdir=output/sample \
                --subdirs --seeds=${seeds} --steps=${steps} --batch=125 \
                --w_stiff=${w_stiff} --w_con=${w_con}
        CUDA_VISIBLE_DEVICES=0 python calculate_metrics.py calc --images=output/sample \
            --ref=dataset-refs/img512.pkl --num=50000 >> ${txt_file}
        CUDA_VISIBLE_DEVICES=0 python calculate_precision_recall_is.py output/sample >> ${txt_file}
        echo "###############################" >> ${txt_file}
        echo "" >> ${txt_file}
        rm -r output/sample/
    done
done

steps=16
for w_stiff in 0.75; do \
    for w_con in 0.3; do \
        echo "###### model: ${model}, guidance: ${guidance}, steps: ${steps}, w_stiff: ${w_stiff}, w_con: ${w_con} ######" >> ${txt_file}
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_images.py \
                --preset=edm2-img512-${model} --outdir=output/sample \
                --subdirs --seeds=${seeds} --steps=${steps} --batch=125 \
                --w_stiff=${w_stiff} --w_con=${w_con}
        CUDA_VISIBLE_DEVICES=0 python calculate_metrics.py calc --images=output/sample \
            --ref=dataset-refs/img512.pkl --num=50000 >> ${txt_file}
        CUDA_VISIBLE_DEVICES=0 python calculate_precision_recall_is.py output/sample >> ${txt_file}
        echo "###############################" >> ${txt_file}
        echo "" >> ${txt_file}
        rm -r output/sample/
    done
done

steps=8
for w_stiff in 0.5; do \
    for w_con in 0.1; do \
        echo "###### model: ${model}, guidance: ${guidance}, steps: ${steps}, w_stiff: ${w_stiff}, w_con: ${w_con} ######" >> ${txt_file}
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_images.py \
                --preset=edm2-img512-${model} --outdir=output/sample \
                --subdirs --seeds=${seeds} --steps=${steps} --batch=125 \
                --w_stiff=${w_stiff} --w_con=${w_con}
        CUDA_VISIBLE_DEVICES=0 python calculate_metrics.py calc --images=output/sample \
            --ref=dataset-refs/img512.pkl --num=50000 >> ${txt_file}
        CUDA_VISIBLE_DEVICES=0 python calculate_precision_recall_is.py output/sample >> ${txt_file}
        echo "###############################" >> ${txt_file}
        echo "" >> ${txt_file}
        rm -r output/sample/
    done
done

## Table 3 ##

steps=32

model="s-guid-fid"
# guidance weight : default
for w_stiff in 1.0; do \
    for w_con in 0.5; do \
        echo "###### model: ${model}, guidance: ${guidance}, steps: ${steps}, w_stiff: ${w_stiff}, w_con: ${w_con} ######" >> ${txt_file}
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_images.py \
                --preset=edm2-img512-${model} --outdir=output/sample \
                --subdirs --seeds=${seeds} --steps=${steps} --batch=125 \
                --w_stiff=${w_stiff} --w_con=${w_con}
        CUDA_VISIBLE_DEVICES=0 python calculate_metrics.py calc --images=output/sample \
            --ref=dataset-refs/img512.pkl --num=50000 >> ${txt_file}
        CUDA_VISIBLE_DEVICES=0 python calculate_precision_recall_is.py output/sample >> ${txt_file}
        echo "###############################" >> ${txt_file}
        echo "" >> ${txt_file}
        rm -r output/sample/
    done
done

model="s-autog-fid"
# guidance weight : default
for w_stiff in 1.0; do \
    for w_con in 0.5; do \
        echo "###### model: ${model}, guidance: ${guidance}, steps: ${steps}, w_stiff: ${w_stiff}, w_con: ${w_con} ######" >> ${txt_file}
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_images.py \
                --preset=edm2-img512-${model} --outdir=output/sample \
                --subdirs --seeds=${seeds} --steps=${steps} --batch=125 \
                --w_stiff=${w_stiff} --w_con=${w_con}
        CUDA_VISIBLE_DEVICES=0 python calculate_metrics.py calc --images=output/sample \
            --ref=dataset-refs/img512.pkl --num=50000 >> ${txt_file}
        CUDA_VISIBLE_DEVICES=0 python calculate_precision_recall_is.py output/sample >> ${txt_file}
        echo "###############################" >> ${txt_file}
        echo "" >> ${txt_file}
        rm -r output/sample/
    done
done

steps=16

model="s-guid-fid"
guidance=1.2
for w_stiff in 0.75; do \
    for w_con in 0.5; do \
        echo "###### model: ${model}, guidance: ${guidance}, steps: ${steps}, w_stiff: ${w_stiff}, w_con: ${w_con} ######" >> ${txt_file}
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_images.py \
                --preset=edm2-img512-${model} --outdir=output/sample \
                --subdirs --seeds=${seeds} --steps=${steps} --batch=125 \
                --w_stiff=${w_stiff} --w_con=${w_con} --guidance=${guidance}
        CUDA_VISIBLE_DEVICES=0 python calculate_metrics.py calc --images=output/sample \
            --ref=dataset-refs/img512.pkl --num=50000 >> ${txt_file}
        CUDA_VISIBLE_DEVICES=0 python calculate_precision_recall_is.py output/sample >> ${txt_file}
        echo "###############################" >> ${txt_file}
        echo "" >> ${txt_file}
        rm -r output/sample/
    done
done

model="s-autog-fid"
guidance=1.55
for w_stiff in 0.75; do \
    for w_con in 0.5; do \
        echo "###### model: ${model}, guidance: ${guidance}, steps: ${steps}, w_stiff: ${w_stiff}, w_con: ${w_con} ######" >> ${txt_file}
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 generate_images.py \
                --preset=edm2-img512-${model} --outdir=output/sample \
                --subdirs --seeds=${seeds} --steps=${steps} --batch=125 \
                --w_stiff=${w_stiff} --w_con=${w_con} --guidance=${guidance}
        CUDA_VISIBLE_DEVICES=0 python calculate_metrics.py calc --images=output/sample \
            --ref=dataset-refs/img512.pkl --num=50000 >> ${txt_file}
        CUDA_VISIBLE_DEVICES=0 python calculate_precision_recall_is.py output/sample >> ${txt_file}
        echo "###############################" >> ${txt_file}
        echo "" >> ${txt_file}
        rm -r output/sample/
    done
done