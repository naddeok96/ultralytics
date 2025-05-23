import pytest
from ultralytics.nn.tasks import yaml_model_load, parse_model
from ultralytics.nn.modules import Conv3d


def test_conv3d_yaml_loading():
    cfg_path = 'ultralytics/cfg/models/12/yolo12-mf.yaml'
    cfg = yaml_model_load(cfg_path)
    model, _ = parse_model(cfg.copy(), ch=3, verbose=False)
    assert any(isinstance(m, Conv3d) for m in model.modules())
