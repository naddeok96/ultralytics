from ultralytics import YOLO
from ultralytics.data import YOLOMultiFrameDataset

# Example usage of the multiframe model and dataset

def main():
    model = YOLO('ultralytics/cfg/models/12/yolo12-mf.yaml')
    data = {
        'train': '/path/to/train/images',
        'val': '/path/to/val/images',
        'nc': 1,
        'names': {0: 'object'},
        'channels': 3,
    }
    dataset = YOLOMultiFrameDataset(img_path=data['train'], data=data, task='detect')
    batch = next(iter(dataset))
    print('stack shape:', batch['img'].shape)

if __name__ == '__main__':
    main()
