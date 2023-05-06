from eyeq import MOTBenchmark

gt_file = "../data/MOT20/train/MOT20-01/gt/gt.txt"
det_file = "../data/MOT20/train/MOT20-01/sort/det.txt"

mot_eval = MOTBenchmark(gt_file=gt_file, det_file=det_file)
mot_eval.load_data()
summary = mot_eval.run_benchmark()
print(summary)

