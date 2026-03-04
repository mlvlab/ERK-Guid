import argparse
import torch_fidelity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gen_dir")
    args = parser.parse_args()

    metrics = torch_fidelity.calculate_metrics(
        input1=args.gen_dir,
        input2="ILSVRC2012_img_val_resize_centercrop",
        cuda=True,
        batch_size=500,
        isc=True,
        prc=True,
        samples_find_deep=True,
        verbose=False,
    )
    print(f"precision = {metrics['precision']:.3f} \nrecall = {metrics['recall']:.3f} \ninception score = {metrics['inception_score_mean']:.3f}")

if __name__ == "__main__":
    main()