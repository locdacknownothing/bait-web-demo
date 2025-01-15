train_dataset_dir = "/mnt/data/src/bait/train/sim/data_sim_31k/Data_Sim_Aug_Split/train"
test_dataset_dir = "/mnt/data/src/bait/train/sim/data_sim_31k/Data_Sim_Aug_Split/val"

labels_per_batch = 64
samples_per_label = 6
batch_size = labels_per_batch * samples_per_label
eval_batch_size = 64
num_workers = 8

epochs = 100  # 20
lr = 0.0001  # 0.001
margin = 0.2  # 1.0
print_freq = 20

save_dir = "./weights/batch_hard"
resume = "./weights/batch_hard/epoch_75__ckpt.pth"  # if resume, set weight file path
