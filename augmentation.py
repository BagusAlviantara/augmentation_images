from argparse import ArgumentParser
import os
import shutil
import albumentations as A
import cv2
import tqdm

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, help='input dir')
    parser.add_argument('--output', type=str, help='output dir')

    args = parser.parse_args()

    base_pipeline = [
        A.Resize(p=1.0, height=512, width=512),
    ]
    
    # pipeline_1 = [
    #     A.HorizontalFlip(p=1)
    # ]

    # pipeline_2 = [
    #     A.VerticalFlip(always_apply=False, p=1.0)
    # ]

    # pipeline_3 = [
    #     A.RandomBrightnessContrast(p=1)
    # ]

    # pipeline_4 = [
    #     A.CLAHE(always_apply=False, p=1.0, clip_limit=(1, 1), tile_grid_size=(30, 5))
    # ]

    # pipeline_5 = [
    #     A.GaussNoise(p=1.0, var_limit=(10.0, 100.0), per_channel=True, mean=20.0)
    # ]

    # pipeline_6 = [
    #     A.RandomRotate90(always_apply=True, p=1.0)
    # ]

    # used_pipelines = [[], pipeline_1, pipeline_2, pipeline_3, pipeline_4, pipeline_5, pipeline_6]
    used_pipelines = [[]]
    
    pipelines = []
    
    # for pipeline in used_pipelines:
    #     pipelines.append(A.Compose(base_pipeline + pipeline))

    for pipeline in used_pipelines:
        pipelines.append(A.Compose(base_pipeline))

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    
    os.makedirs(args.output)
    
    for folder in tqdm.tqdm(os.listdir(args.input)):
        class_dir = os.path.join(args.input, folder)
        output_class_dir = os.path.join(args.output, folder)
        os.makedirs(output_class_dir)
        for file in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file)
            output_file_path = os.path.join(output_class_dir, file)
            
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            filename = output_file_path[:-(len(file_path.split('.')[-1])+1)]
            
            for i, pipeline in enumerate(pipelines):
                result = pipeline(image=image)["image"]
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{filename}_{i}.jpg", result)