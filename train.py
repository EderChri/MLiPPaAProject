from dataset import TrajectoryDataset

dataset = TrajectoryDataset("output.txt", "parameter.txt", to_tensor=True)
test = dataset.__getitem__(1)