from dataset_rendered_2 import DatasetRendered2
import cv2
dataset_path = "/home/simon/datasets/structure_core_unity"

dataset = DatasetRendered2(dataset_path, 0, 100)


for data in dataset:
    ir, gt, mask = data
    cv2.imshow("ir", ir)
    cv2.imshow("gt", gt)
    cv2.waitKey()
    print(ir)

